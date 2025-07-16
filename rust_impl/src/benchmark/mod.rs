/// Benchmark framework for comparing CPU vs CUDA performance
/// Specifically designed to test prime factorization and genomic operations

pub mod prime_factorization;
pub mod runner;

use std::time::{Duration, Instant};
use anyhow::Result;

/// Benchmark result for a single operation
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub cpu_time: Duration,
    pub cuda_time: Option<Duration>,
    pub speedup: Option<f64>,
    pub iterations: usize,
    pub error_rate: f64,
}

/// Trait for benchmarkable operations
pub trait Benchmarkable {
    /// Run the benchmark on CPU
    fn run_cpu(&self) -> Result<Vec<u64>>;
    
    /// Run the benchmark on CUDA (if available)
    fn run_cuda(&self) -> Result<Vec<u64>>;
    
    /// Get benchmark name
    fn name(&self) -> &str;
    
    /// Number of iterations to run
    fn iterations(&self) -> usize {
        100
    }
}

/// Run a benchmark and collect results
pub fn run_benchmark<B: Benchmarkable>(benchmark: &B, warm_up: bool) -> Result<BenchmarkResult> {
    let iterations = benchmark.iterations();
    
    // Warm up if requested
    if warm_up {
        for _ in 0..5 {
            let _ = benchmark.run_cpu()?;
            #[cfg(feature = "cuda")]
            {
                let _ = benchmark.run_cuda();
            }
        }
    }
    
    // Run CPU benchmark
    let cpu_start = Instant::now();
    let mut cpu_results = Vec::new();
    for _ in 0..iterations {
        cpu_results.extend(benchmark.run_cpu()?);
    }
    let cpu_time = cpu_start.elapsed();
    
    // Run CUDA benchmark if available
    let (cuda_time, cuda_results) = {
        #[cfg(feature = "cuda")]
        {
            if crate::cuda::cuda_available() {
                let cuda_start = Instant::now();
                let mut results = Vec::new();
                for _ in 0..iterations {
                    match benchmark.run_cuda() {
                        Ok(r) => results.extend(r),
                        Err(e) => {
                            log::warn!("CUDA benchmark failed: {}", e);
                            (None, Vec::new())
                        }
                    }
                }
                let time = cuda_start.elapsed();
                if !results.is_empty() {
                    (Some(time), results)
                } else {
                    (None, Vec::new())
                }
            } else {
                (None, Vec::new())
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            (None, Vec::new())
        }
    };
    
    // Calculate error rate if CUDA results available
    let error_rate = if !cuda_results.is_empty() {
        let errors = cpu_results.iter().zip(cuda_results.iter())
            .filter(|(a, b)| a != b)
            .count();
        errors as f64 / cpu_results.len().min(cuda_results.len()) as f64
    } else {
        0.0
    };
    
    // Calculate speedup
    let speedup = cuda_time.map(|ct| {
        cpu_time.as_secs_f64() / ct.as_secs_f64()
    });
    
    Ok(BenchmarkResult {
        name: benchmark.name().to_string(),
        cpu_time,
        cuda_time,
        speedup,
        iterations,
        error_rate,
    })
}

/// Format benchmark results for display
pub fn format_results(results: &[BenchmarkResult]) -> String {
    use std::fmt::Write;
    
    let mut output = String::new();
    writeln!(&mut output, "\n{:=<80}", "").unwrap();
    writeln!(&mut output, "{:^80}", "BENCHMARK RESULTS").unwrap();
    writeln!(&mut output, "{:=<80}", "").unwrap();
    writeln!(&mut output, "{:<30} {:>15} {:>15} {:>10} {:>8}", 
             "Benchmark", "CPU Time", "CUDA Time", "Speedup", "Error").unwrap();
    writeln!(&mut output, "{:-<80}", "").unwrap();
    
    for result in results {
        let cpu_time = format!("{:.3}ms", result.cpu_time.as_secs_f64() * 1000.0);
        let cuda_time = result.cuda_time
            .map(|t| format!("{:.3}ms", t.as_secs_f64() * 1000.0))
            .unwrap_or_else(|| "N/A".to_string());
        let speedup = result.speedup
            .map(|s| format!("{:.2}x", s))
            .unwrap_or_else(|| "N/A".to_string());
        let error = format!("{:.2}%", result.error_rate * 100.0);
        
        writeln!(&mut output, "{:<30} {:>15} {:>15} {:>10} {:>8}", 
                 result.name, cpu_time, cuda_time, speedup, error).unwrap();
    }
    
    writeln!(&mut output, "{:=<80}", "").unwrap();
    
    // Summary statistics
    let total_speedup = results.iter()
        .filter_map(|r| r.speedup)
        .sum::<f64>() / results.iter().filter(|r| r.speedup.is_some()).count().max(1) as f64;
    
    writeln!(&mut output, "\nAverage Speedup: {:.2}x", total_speedup).unwrap();
    
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    
    struct TestBenchmark;
    
    impl Benchmarkable for TestBenchmark {
        fn run_cpu(&self) -> Result<Vec<u64>> {
            Ok(vec![1, 2, 3, 4, 5])
        }
        
        fn run_cuda(&self) -> Result<Vec<u64>> {
            Ok(vec![1, 2, 3, 4, 5])
        }
        
        fn name(&self) -> &str {
            "Test Benchmark"
        }
        
        fn iterations(&self) -> usize {
            10
        }
    }
    
    #[test]
    fn test_benchmark_framework() {
        let benchmark = TestBenchmark;
        let result = run_benchmark(&benchmark, false).unwrap();
        
        assert_eq!(result.name, "Test Benchmark");
        assert_eq!(result.iterations, 10);
        assert!(result.cpu_time.as_nanos() > 0);
        assert_eq!(result.error_rate, 0.0);
    }
}