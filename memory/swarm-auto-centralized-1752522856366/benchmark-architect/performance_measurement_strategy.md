# Performance Measurement Strategy for Prime Factorization

## Overview

This document outlines the comprehensive strategy for measuring and comparing CPU vs CUDA performance for prime factorization, with focus on accuracy, reliability, and statistical significance.

## Timing Methodology

### 1. High-Precision Timing

```rust
use std::time::{Instant, Duration};
use cudarc::driver::CudaEvent;

pub struct PrecisionTimer {
    cpu_timer: CpuTimer,
    gpu_timer: Option<GpuTimer>,
}

pub struct CpuTimer {
    start: Option<Instant>,
    measurements: Vec<Duration>,
}

pub struct GpuTimer {
    start_event: CudaEvent,
    stop_event: CudaEvent,
    measurements: Vec<f32>, // milliseconds
}

impl PrecisionTimer {
    /// CPU timing with nanosecond precision
    pub fn time_cpu<F, R>(&mut self, f: F) -> (R, Duration)
    where F: FnOnce() -> R {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        self.cpu_timer.measurements.push(duration);
        (result, duration)
    }
    
    /// GPU timing using CUDA events
    pub fn time_gpu<F, R>(&mut self, f: F) -> Result<(R, f32)>
    where F: FnOnce() -> Result<R> {
        self.gpu_timer.as_mut()
            .ok_or(anyhow!("GPU timer not initialized"))?
            .start_event.record()?;
        
        let result = f()?;
        
        self.gpu_timer.as_mut().unwrap().stop_event.record()?;
        self.gpu_timer.as_mut().unwrap().stop_event.synchronize()?;
        
        let elapsed_ms = self.gpu_timer.as_ref().unwrap()
            .start_event.elapsed_time(&self.gpu_timer.as_ref().unwrap().stop_event)?;
        
        self.gpu_timer.as_mut().unwrap().measurements.push(elapsed_ms);
        Ok((result, elapsed_ms))
    }
}
```

### 2. Warmup Strategy

```rust
pub struct WarmupConfig {
    cpu_iterations: u32,
    gpu_iterations: u32,
    discard_outliers: bool,
    outlier_threshold: f64, // z-score
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            cpu_iterations: 10,
            gpu_iterations: 20, // More for GPU due to JIT compilation
            discard_outliers: true,
            outlier_threshold: 3.0,
        }
    }
}

pub fn warmup_cpu<F>(config: &WarmupConfig, mut f: F) -> Vec<Duration>
where F: FnMut() {
    let mut timings = Vec::with_capacity(config.cpu_iterations as usize);
    
    for _ in 0..config.cpu_iterations {
        let start = Instant::now();
        f();
        timings.push(start.elapsed());
    }
    
    if config.discard_outliers {
        remove_outliers(&mut timings, config.outlier_threshold);
    }
    
    timings
}
```

### 3. Statistical Analysis

```rust
pub struct PerformanceStatistics {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub percentile_95: f64,
    pub percentile_99: f64,
    pub coefficient_of_variation: f64,
}

impl PerformanceStatistics {
    pub fn from_measurements(measurements: &[Duration]) -> Self {
        let times_ms: Vec<f64> = measurements.iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect();
        
        let mean = statistical::mean(&times_ms);
        let median = statistical::median(&times_ms);
        let std_dev = statistical::standard_deviation(&times_ms, Some(mean));
        let min = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let percentile_95 = statistical::percentile(&times_ms, 95.0);
        let percentile_99 = statistical::percentile(&times_ms, 99.0);
        let cv = std_dev / mean;
        
        Self {
            mean, median, std_dev, min, max,
            percentile_95, percentile_99,
            coefficient_of_variation: cv,
        }
    }
    
    pub fn report(&self) -> String {
        format!(
            "Performance Statistics:\n\
             Mean: {:.3} ms\n\
             Median: {:.3} ms\n\
             Std Dev: {:.3} ms\n\
             Min: {:.3} ms\n\
             Max: {:.3} ms\n\
             95th percentile: {:.3} ms\n\
             99th percentile: {:.3} ms\n\
             CV: {:.2}%",
            self.mean, self.median, self.std_dev,
            self.min, self.max, self.percentile_95,
            self.percentile_99, self.coefficient_of_variation * 100.0
        )
    }
}
```

## Memory Profiling

### 1. CPU Memory Tracking

```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct TrackingAllocator {
    current: AtomicUsize,
    peak: AtomicUsize,
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ret = System.alloc(layout);
        if !ret.is_null() {
            self.current.fetch_add(layout.size(), Ordering::SeqCst);
            let current = self.current.load(Ordering::SeqCst);
            self.peak.fetch_max(current, Ordering::SeqCst);
        }
        ret
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        self.current.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator {
    current: AtomicUsize::new(0),
    peak: AtomicUsize::new(0),
};

pub fn get_memory_stats() -> MemoryStats {
    MemoryStats {
        current_bytes: ALLOCATOR.current.load(Ordering::SeqCst),
        peak_bytes: ALLOCATOR.peak.load(Ordering::SeqCst),
    }
}
```

### 2. GPU Memory Profiling

```rust
use nvml_wrapper::{Nvml, Device};

pub struct GpuMemoryProfiler {
    nvml: Nvml,
    device: Device,
}

impl GpuMemoryProfiler {
    pub fn new() -> Result<Self> {
        let nvml = Nvml::init()?;
        let device = nvml.device_by_index(0)?;
        Ok(Self { nvml, device })
    }
    
    pub fn get_memory_info(&self) -> Result<GpuMemoryInfo> {
        let mem_info = self.device.memory_info()?;
        Ok(GpuMemoryInfo {
            total_bytes: mem_info.total,
            used_bytes: mem_info.used,
            free_bytes: mem_info.free,
        })
    }
    
    pub fn profile_operation<F, R>(&self, f: F) -> Result<(R, GpuMemoryDelta)>
    where F: FnOnce() -> Result<R> {
        let before = self.get_memory_info()?;
        let result = f()?;
        let after = self.get_memory_info()?;
        
        Ok((result, GpuMemoryDelta {
            allocated_bytes: after.used_bytes.saturating_sub(before.used_bytes),
            peak_usage_bytes: after.used_bytes,
        }))
    }
}
```

## Benchmark Execution Framework

### 1. Comprehensive Benchmark Suite

```rust
pub struct BenchmarkSuite {
    config: BenchmarkConfig,
    timer: PrecisionTimer,
    memory_profiler: MemoryProfiler,
    results: BenchmarkResults,
}

pub struct BenchmarkConfig {
    pub warmup_iterations: u32,
    pub measurement_iterations: u32,
    pub target_number: u64,
    pub algorithms: Vec<Algorithm>,
    pub enable_gpu: bool,
    pub enable_memory_profiling: bool,
    pub output_format: OutputFormat,
}

pub enum Algorithm {
    CpuTrialDivision,
    CpuPollardRho,
    CpuParallel,
    GpuTrialDivision,
    GpuPollardRho,
}

impl BenchmarkSuite {
    pub fn run(&mut self) -> Result<BenchmarkReport> {
        // Phase 1: Warmup
        self.warmup_phase()?;
        
        // Phase 2: CPU Benchmarks
        let cpu_results = self.benchmark_cpu_algorithms()?;
        
        // Phase 3: GPU Benchmarks (if enabled)
        let gpu_results = if self.config.enable_gpu {
            Some(self.benchmark_gpu_algorithms()?)
        } else {
            None
        };
        
        // Phase 4: Analysis
        let analysis = self.analyze_results(&cpu_results, &gpu_results)?;
        
        Ok(BenchmarkReport {
            cpu_results,
            gpu_results,
            analysis,
            config: self.config.clone(),
        })
    }
}
```

### 2. Result Analysis

```rust
pub struct PerformanceAnalysis {
    pub speedup_factors: HashMap<String, f64>,
    pub efficiency_metrics: EfficiencyMetrics,
    pub bottleneck_analysis: BottleneckAnalysis,
    pub recommendations: Vec<String>,
}

pub struct EfficiencyMetrics {
    pub cpu_efficiency: f64, // Operations per second per core
    pub gpu_efficiency: f64, // Operations per second per CUDA core
    pub memory_bandwidth_utilization: f64,
    pub compute_utilization: f64,
}

pub struct BottleneckAnalysis {
    pub is_memory_bound: bool,
    pub is_compute_bound: bool,
    pub limiting_factor: String,
    pub optimization_potential: f64, // 0.0 to 1.0
}

impl PerformanceAnalysis {
    pub fn analyze(cpu_results: &[AlgorithmResult], 
                   gpu_results: &Option<Vec<AlgorithmResult>>) -> Self {
        let mut speedup_factors = HashMap::new();
        
        // Calculate speedups
        if let Some(gpu_res) = gpu_results {
            for gpu in gpu_res {
                if let Some(cpu) = cpu_results.iter()
                    .find(|c| comparable(&c.algorithm, &gpu.algorithm)) {
                    let speedup = cpu.mean_time_ms / gpu.mean_time_ms;
                    speedup_factors.insert(gpu.algorithm.to_string(), speedup);
                }
            }
        }
        
        // Analyze efficiency
        let efficiency_metrics = calculate_efficiency(cpu_results, gpu_results);
        
        // Identify bottlenecks
        let bottleneck_analysis = identify_bottlenecks(&efficiency_metrics);
        
        // Generate recommendations
        let recommendations = generate_recommendations(&bottleneck_analysis);
        
        Self {
            speedup_factors,
            efficiency_metrics,
            bottleneck_analysis,
            recommendations,
        }
    }
}
```

## Output and Reporting

### 1. JSON Report Format

```json
{
  "benchmark_report": {
    "timestamp": "2024-01-14T10:30:00Z",
    "target_number": 100822548703,
    "factors": [317213, 317879],
    "system_info": {
      "cpu": "AMD Ryzen 9 5900X",
      "gpu": "NVIDIA GeForce RTX 2070",
      "memory": "32GB DDR4",
      "os": "Linux 5.15"
    },
    "cpu_results": {
      "trial_division": {
        "mean_ms": 823.45,
        "median_ms": 820.12,
        "std_dev_ms": 15.23,
        "min_ms": 798.34,
        "max_ms": 865.67,
        "memory_peak_mb": 2.34
      },
      "pollard_rho": {
        "mean_ms": 67.89,
        "median_ms": 66.45,
        "std_dev_ms": 3.21,
        "min_ms": 63.12,
        "max_ms": 75.43,
        "memory_peak_mb": 1.56
      }
    },
    "gpu_results": {
      "cuda_trial_division": {
        "mean_ms": 12.34,
        "median_ms": 11.98,
        "std_dev_ms": 0.89,
        "min_ms": 11.02,
        "max_ms": 14.56,
        "memory_peak_mb": 45.67,
        "speedup": 66.75
      }
    },
    "analysis": {
      "best_cpu_algorithm": "pollard_rho",
      "best_gpu_algorithm": "cuda_trial_division",
      "overall_speedup": 66.75,
      "gpu_efficiency": 0.78,
      "recommendations": [
        "GPU implementation shows excellent speedup",
        "Consider batch processing for better GPU utilization",
        "Memory usage is well within limits"
      ]
    }
  }
}
```

### 2. Visualization Components

```rust
pub struct BenchmarkVisualizer {
    results: BenchmarkReport,
}

impl BenchmarkVisualizer {
    pub fn generate_speedup_chart(&self) -> String {
        // Generate ASCII or plotters-based chart
        // showing speedup factors
    }
    
    pub fn generate_timing_comparison(&self) -> String {
        // Bar chart comparing all algorithms
    }
    
    pub fn generate_memory_usage_chart(&self) -> String {
        // Memory usage comparison
    }
    
    pub fn generate_efficiency_heatmap(&self) -> String {
        // Efficiency metrics visualization
    }
}
```

## Continuous Monitoring

### 1. Performance Regression Detection

```rust
pub struct RegressionDetector {
    baseline_results: HashMap<String, PerformanceBaseline>,
    threshold: f64, // Percentage change to trigger alert
}

pub struct PerformanceBaseline {
    algorithm: String,
    mean_time_ms: f64,
    std_dev_ms: f64,
    recorded_date: DateTime<Utc>,
}

impl RegressionDetector {
    pub fn check_regression(&self, current: &AlgorithmResult) -> RegressionStatus {
        if let Some(baseline) = self.baseline_results.get(&current.algorithm) {
            let change_percent = (current.mean_time_ms - baseline.mean_time_ms) 
                / baseline.mean_time_ms * 100.0;
            
            if change_percent > self.threshold {
                RegressionStatus::Regression {
                    baseline: baseline.mean_time_ms,
                    current: current.mean_time_ms,
                    change_percent,
                }
            } else if change_percent < -self.threshold {
                RegressionStatus::Improvement {
                    baseline: baseline.mean_time_ms,
                    current: current.mean_time_ms,
                    change_percent: change_percent.abs(),
                }
            } else {
                RegressionStatus::NoChange
            }
        } else {
            RegressionStatus::NoBaseline
        }
    }
}
```

## Key Performance Indicators (KPIs)

1. **Primary KPIs**:
   - Execution time (mean, median, p95)
   - Speedup factor (GPU vs best CPU)
   - Memory usage (peak)
   - Accuracy (100% required)

2. **Secondary KPIs**:
   - Throughput (factorizations/second)
   - Efficiency (utilization percentage)
   - Scalability (performance vs problem size)
   - Energy efficiency (operations/watt)

3. **Quality Metrics**:
   - Result consistency
   - Performance stability (CV < 5%)
   - Memory leak detection
   - Error rate (must be 0%)

## Benchmark Automation

```rust
pub struct AutomatedBenchmarkRunner {
    schedule: CronSchedule,
    suite: BenchmarkSuite,
    reporter: BenchmarkReporter,
    regression_detector: RegressionDetector,
}

impl AutomatedBenchmarkRunner {
    pub async fn run_scheduled(&mut self) -> Result<()> {
        loop {
            if self.schedule.should_run_now() {
                let report = self.suite.run()?;
                
                // Check for regressions
                let regressions = self.regression_detector.check_all(&report);
                
                // Generate and send report
                self.reporter.send_report(&report, &regressions).await?;
                
                // Update baselines if needed
                if regressions.is_empty() {
                    self.regression_detector.update_baselines(&report);
                }
            }
            
            tokio::time::sleep(Duration::from_secs(60)).await;
        }
    }
}
```