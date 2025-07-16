# Prime Factorization Benchmark Framework Design

## Overview
This document outlines the comprehensive benchmark framework for comparing CPU vs CUDA prime factorization performance, specifically designed to factorize the target number 100822548703 (product of two 6-digit primes).

## Architecture Components

### 1. Core Benchmark Structure

```rust
// Location: rust_impl/src/prime_factorization/mod.rs

pub mod cpu_factorizer;
pub mod cuda_factorizer;
pub mod benchmark;
pub mod types;

// Location: rust_impl/src/prime_factorization/types.rs
#[derive(Debug, Clone, PartialEq)]
pub struct FactorizationResult {
    pub number: u64,
    pub factors: Vec<u64>,
    pub is_prime: bool,
    pub execution_time_ms: f64,
    pub memory_usage_bytes: usize,
    pub algorithm_used: String,
}

#[derive(Debug)]
pub struct BenchmarkMetrics {
    pub cpu_time_ms: f64,
    pub gpu_time_ms: f64,
    pub speedup: f64,
    pub cpu_memory_peak: usize,
    pub gpu_memory_peak: usize,
    pub accuracy_verified: bool,
    pub iterations: u32,
}
```

### 2. CPU Implementation Strategy

```rust
// Location: rust_impl/src/prime_factorization/cpu_factorizer.rs

pub trait Factorizer {
    fn factorize(&self, n: u64) -> Result<FactorizationResult>;
}

pub struct CpuFactorizer {
    algorithm: FactorizationAlgorithm,
}

pub enum FactorizationAlgorithm {
    TrialDivision,
    PollardRho,
    QuadraticSieve,
    ParallelTrialDivision,
}

impl CpuFactorizer {
    // Trial Division (baseline)
    pub fn trial_division(&self, n: u64) -> Vec<u64>;
    
    // Pollard's Rho (optimized for semi-primes)
    pub fn pollard_rho(&self, n: u64) -> Vec<u64>;
    
    // Parallel Trial Division using Rayon
    pub fn parallel_trial_division(&self, n: u64) -> Vec<u64>;
}
```

### 3. CUDA Implementation Strategy

```rust
// Location: rust_impl/src/prime_factorization/cuda_factorizer.rs

pub struct CudaFactorizer {
    accelerator: CudaAccelerator,
    block_size: u32,
    grid_size: u32,
}

impl CudaFactorizer {
    // GPU-accelerated trial division
    pub fn cuda_trial_division(&mut self, n: u64) -> Result<Vec<u64>>;
    
    // Parallel modular arithmetic on GPU
    pub fn cuda_pollard_rho(&mut self, n: u64) -> Result<Vec<u64>>;
    
    // Batch factorization for multiple numbers
    pub fn batch_factorize(&mut self, numbers: &[u64]) -> Result<Vec<FactorizationResult>>;
}

// CUDA Kernel Design
// Location: rust_impl/src/cuda/kernels/prime_factorizer.rs
/*
__global__ void trial_division_kernel(
    unsigned long long n,
    unsigned long long* candidates,
    int num_candidates,
    unsigned long long* result,
    int* found
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_candidates) {
        unsigned long long candidate = candidates[idx];
        if (n % candidate == 0) {
            atomicExch((unsigned long long*)result, candidate);
            atomicExch(found, 1);
        }
    }
}
*/
```

### 4. Benchmark Framework Design

```rust
// Location: rust_impl/src/prime_factorization/benchmark/mod.rs

pub struct PrimeFactorizationBenchmark {
    cpu_factorizer: CpuFactorizer,
    cuda_factorizer: Option<CudaFactorizer>,
    target_number: u64,
    warmup_iterations: u32,
    benchmark_iterations: u32,
}

impl PrimeFactorizationBenchmark {
    pub fn new(target: u64) -> Self;
    
    // Run complete benchmark suite
    pub fn run_full_benchmark(&mut self) -> BenchmarkMetrics;
    
    // Individual benchmarks
    pub fn benchmark_cpu_trial_division(&self) -> (f64, usize);
    pub fn benchmark_cpu_pollard_rho(&self) -> (f64, usize);
    pub fn benchmark_cpu_parallel(&self) -> (f64, usize);
    pub fn benchmark_cuda_trial_division(&mut self) -> Result<(f64, usize)>;
    pub fn benchmark_cuda_pollard_rho(&mut self) -> Result<(f64, usize)>;
    
    // Memory profiling
    pub fn profile_memory_usage(&self) -> MemoryProfile;
    
    // Accuracy verification
    pub fn verify_results(&self, result: &FactorizationResult) -> bool;
}
```

### 5. Performance Measurement Strategy

```rust
// Location: rust_impl/src/prime_factorization/benchmark/metrics.rs

pub struct PerformanceProfiler {
    cpu_timer: CpuTimer,
    gpu_timer: Option<CudaTimer>,
    memory_tracker: MemoryTracker,
}

pub struct MemoryTracker {
    baseline_memory: usize,
    peak_memory: usize,
    allocations: Vec<AllocationEvent>,
}

pub struct CudaTimer {
    start_event: CudaEvent,
    stop_event: CudaEvent,
}

impl PerformanceProfiler {
    // High-precision timing
    pub fn time_cpu_operation<F, R>(&mut self, op: F) -> (R, f64)
        where F: FnOnce() -> R;
    
    pub fn time_gpu_operation<F, R>(&mut self, op: F) -> Result<(R, f64)>
        where F: FnOnce() -> Result<R>;
    
    // Memory tracking
    pub fn track_memory_usage<F, R>(&mut self, op: F) -> (R, MemoryStats)
        where F: FnOnce() -> R;
}
```

### 6. Test Case Specifications

```rust
// Location: rust_impl/src/prime_factorization/benchmark/test_cases.rs

pub struct TestCase {
    pub input: u64,
    pub expected_factors: Vec<u64>,
    pub category: TestCategory,
}

pub enum TestCategory {
    SmallPrime,      // < 1000
    MediumPrime,     // < 1_000_000
    LargePrime,      // < 10^12
    SemiPrime,       // Product of two primes
    Composite,       // Multiple factors
    TargetCase,      // 100822548703
}

pub fn generate_test_suite() -> Vec<TestCase> {
    vec![
        // Edge cases
        TestCase { input: 2, expected_factors: vec![2], category: SmallPrime },
        TestCase { input: 97, expected_factors: vec![97], category: SmallPrime },
        
        // Semi-primes of increasing difficulty
        TestCase { input: 15, expected_factors: vec![3, 5], category: SemiPrime },
        TestCase { input: 9999991, expected_factors: vec![9999991], category: LargePrime },
        
        // Target case
        TestCase { 
            input: 100822548703, 
            expected_factors: vec![317213, 317879], 
            category: TargetCase 
        },
        
        // Batch processing cases
        // ... more test cases
    ]
}
```

### 7. Regression Test Framework

```rust
// Location: rust_impl/tests/prime_factorization_tests.rs

#[cfg(test)]
mod prime_factorization_tests {
    use super::*;
    
    #[test]
    fn test_cpu_correctness() {
        let factorizer = CpuFactorizer::new(FactorizationAlgorithm::TrialDivision);
        for test_case in generate_test_suite() {
            let result = factorizer.factorize(test_case.input).unwrap();
            assert_eq!(result.factors, test_case.expected_factors);
        }
    }
    
    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_correctness() {
        let mut factorizer = CudaFactorizer::new().unwrap();
        for test_case in generate_test_suite() {
            let result = factorizer.cuda_trial_division(test_case.input).unwrap();
            assert_eq!(result, test_case.expected_factors);
        }
    }
    
    #[test]
    fn test_performance_regression() {
        let mut benchmark = PrimeFactorizationBenchmark::new(100822548703);
        let metrics = benchmark.run_full_benchmark();
        
        // Assert performance thresholds
        assert!(metrics.cpu_time_ms < 1000.0, "CPU performance regression");
        if metrics.gpu_time_ms > 0.0 {
            assert!(metrics.speedup > 10.0, "GPU speedup regression");
        }
    }
    
    #[test]
    fn test_memory_usage() {
        let benchmark = PrimeFactorizationBenchmark::new(100822548703);
        let memory_profile = benchmark.profile_memory_usage();
        
        // Assert memory bounds
        assert!(memory_profile.cpu_peak_bytes < 100_000_000, "Excessive CPU memory");
        assert!(memory_profile.gpu_peak_bytes < 500_000_000, "Excessive GPU memory");
    }
}
```

### 8. Benchmark Runner CLI

```rust
// Location: rust_impl/src/bin/prime_benchmark.rs

use clap::Parser;

#[derive(Parser)]
struct Args {
    /// Number to factorize
    #[arg(long, default_value = "100822548703")]
    number: u64,
    
    /// Number of benchmark iterations
    #[arg(long, default_value = "100")]
    iterations: u32,
    
    /// Enable CUDA benchmarks
    #[arg(long)]
    cuda: bool,
    
    /// Output format (json, table, csv)
    #[arg(long, default_value = "table")]
    format: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    let mut benchmark = PrimeFactorizationBenchmark::new(args.number);
    benchmark.set_iterations(args.iterations);
    
    let results = benchmark.run_full_benchmark();
    
    match args.format.as_str() {
        "json" => println!("{}", serde_json::to_string_pretty(&results)?),
        "csv" => export_csv(&results)?,
        _ => print_table(&results),
    }
    
    Ok(())
}
```

### 9. Integration with Existing Codebase

```toml
# Addition to Cargo.toml
[dependencies]
num-integer = "0.1"  # For GCD and number theory
primal = "0.3"       # For prime number utilities

[[bench]]
name = "prime_factorization"
harness = false
```

### 10. Expected Performance Characteristics

Based on the algorithmic complexity and GPU parallelization potential:

1. **CPU Trial Division**: O(√n) - Expected ~500-1000ms for 100822548703
2. **CPU Pollard's Rho**: O(n^(1/4)) - Expected ~50-100ms
3. **CPU Parallel Trial**: O(√n/p) where p = cores - Expected ~100-200ms
4. **CUDA Trial Division**: O(√n/p) where p = CUDA cores - Expected ~5-20ms
5. **CUDA Pollard's Rho**: Complex parallelization - Expected ~10-30ms

Expected speedup: 20-100x depending on algorithm and GPU utilization.

## Memory Optimizations

1. **CPU Side**:
   - Use bit-packed prime sieves
   - Memory-mapped candidate lists
   - Cache-friendly data structures

2. **GPU Side**:
   - Shared memory for frequently accessed data
   - Coalesced memory access patterns
   - Texture memory for prime tables

## Deliverables Checklist

- [x] Benchmark framework architecture design
- [x] Test case specifications for 100822548703
- [x] Performance measurement strategy with timing and memory profiling
- [x] Regression test framework design
- [x] Integration plan with existing CUDA infrastructure
- [x] CLI tool specification for running benchmarks
- [x] Expected performance characteristics documentation

## Next Steps

1. Implement CPU factorization algorithms
2. Implement CUDA kernels for prime factorization
3. Create benchmark harness with criterion
4. Add regression tests to CI pipeline
5. Document performance results and optimization opportunities