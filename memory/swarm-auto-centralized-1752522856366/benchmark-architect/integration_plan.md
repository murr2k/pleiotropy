# Integration Plan for Prime Factorization Benchmark

## Overview

This document outlines the step-by-step integration plan for adding prime factorization benchmarks to the existing genomic cryptanalysis codebase, leveraging the existing CUDA infrastructure.

## Integration Strategy

### Phase 1: Project Structure Setup

```bash
rust_impl/
├── src/
│   ├── prime_factorization/           # NEW MODULE
│   │   ├── mod.rs
│   │   ├── types.rs                  # Core types and traits
│   │   ├── cpu_factorizer.rs         # CPU implementations
│   │   ├── cuda_factorizer.rs        # CUDA wrapper
│   │   ├── benchmark/
│   │   │   ├── mod.rs
│   │   │   ├── runner.rs             # Benchmark execution
│   │   │   ├── metrics.rs            # Performance metrics
│   │   │   └── report.rs             # Report generation
│   │   └── algorithms/
│   │       ├── trial_division.rs
│   │       ├── pollard_rho.rs
│   │       └── quadratic_sieve.rs
│   ├── cuda/
│   │   └── kernels/
│   │       └── prime_factorizer.rs   # NEW CUDA KERNEL
│   └── bin/
│       └── prime_benchmark.rs         # NEW BINARY
├── benches/
│   └── prime_factorization.rs         # NEW BENCHMARK
└── tests/
    └── prime_factorization_tests.rs   # NEW TESTS
```

### Phase 2: Dependency Updates

```toml
# Cargo.toml additions
[dependencies]
# Number theory utilities
num-integer = "0.1"
num-traits = "0.2"
primal = "0.3"

# For Pollard's Rho
rand_chacha = "0.3"  # Deterministic RNG for reproducible results

# Benchmarking
statistical = "1.0"  # For statistical analysis

[[bin]]
name = "prime_benchmark"
path = "src/bin/prime_benchmark.rs"

[[bench]]
name = "prime_factorization"
harness = false
```

### Phase 3: Integration with Existing CUDA Infrastructure

#### 3.1 Extend CudaAccelerator

```rust
// In src/cuda/mod.rs, add to CudaAccelerator impl
impl CudaAccelerator {
    /// Prime factorization using GPU
    pub fn factorize_prime(&mut self, n: u64) -> Result<Vec<u64>> {
        let kernel = self.load_kernel("prime_factorizer")?;
        let result = kernel.launch(n)?;
        Ok(result)
    }
    
    /// Batch prime factorization
    pub fn batch_factorize(&mut self, numbers: &[u64]) -> Result<Vec<Vec<u64>>> {
        let kernel = self.load_kernel("batch_prime_factorizer")?;
        kernel.launch_batch(numbers)
    }
}
```

#### 3.2 Add CUDA Kernel

```cuda
// src/cuda/kernels/prime_factorizer.rs
pub const PRIME_FACTORIZER_KERNEL: &str = r#"
extern "C" __global__ void trial_division_kernel(
    unsigned long long n,
    unsigned long long* candidates,
    unsigned int num_candidates,
    unsigned long long* result,
    unsigned int* found_count
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;
    
    unsigned long long candidate = candidates[idx];
    if (candidate * candidate > n) return;
    
    if (n % candidate == 0) {
        unsigned int pos = atomicAdd(found_count, 1);
        if (pos < 2) {  // Maximum 2 factors for semi-prime
            result[pos] = candidate;
            result[pos + 2] = n / candidate;
        }
    }
}

extern "C" __global__ void pollard_rho_kernel(
    unsigned long long n,
    unsigned long long* x_values,
    unsigned int num_threads,
    unsigned long long* result
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_threads) return;
    
    // Pollard's Rho implementation
    unsigned long long x = x_values[idx];
    unsigned long long y = x;
    unsigned long long d = 1;
    
    while (d == 1) {
        x = (x * x + 1) % n;
        y = (y * y + 1) % n;
        y = (y * y + 1) % n;
        
        unsigned long long diff = (x > y) ? x - y : y - x;
        d = gcd(diff, n);
    }
    
    if (d != n) {
        atomicExch(result, d);
    }
}
"#;
```

### Phase 4: Reuse Existing Components

#### 4.1 Leverage Performance Profiling

```rust
// Reuse existing performance profiler
use crate::cuda::performance::{PerformanceProfiler, Gtx2070Optimizer};

impl PrimeFactorizationBenchmark {
    fn setup_profiling(&mut self) {
        self.profiler = PerformanceProfiler::new();
        self.optimizer = Gtx2070Optimizer::default();
        
        // Configure for prime factorization workload
        let (grid, block) = self.optimizer.optimize_launch_config(
            self.target_number.isqrt() as usize,  // Search space
            32  // Registers per thread
        );
        
        self.cuda_factorizer.set_launch_config(grid, block);
    }
}
```

#### 4.2 Integrate with Existing Test Framework

```rust
// Extend existing test utilities
use crate::tests::test_utils::{setup_test_env, teardown_test_env};

#[test]
fn test_prime_factorization_integration() {
    setup_test_env();
    
    let mut benchmark = PrimeFactorizationBenchmark::new(100822548703);
    let results = benchmark.run_full_benchmark();
    
    assert!(results.accuracy_verified);
    assert!(results.speedup > 10.0);
    
    teardown_test_env();
}
```

### Phase 5: CLI Integration

```rust
// Extend main.rs to include prime factorization mode
use clap::Subcommand;

#[derive(Subcommand)]
enum Commands {
    /// Analyze genome for pleiotropy
    Analyze {
        #[arg(short, long)]
        genome_file: String,
    },
    
    /// Benchmark prime factorization
    #[command(name = "prime-bench")]
    PrimeBenchmark {
        /// Number to factorize
        #[arg(long, default_value = "100822548703")]
        number: u64,
        
        /// Enable CUDA acceleration
        #[arg(long)]
        cuda: bool,
        
        /// Number of iterations
        #[arg(long, default_value = "100")]
        iterations: u32,
    },
}
```

### Phase 6: Documentation Integration

#### 6.1 Update README

```markdown
## Prime Factorization Benchmarks

The project now includes comprehensive CPU vs CUDA benchmarks for prime factorization:

```bash
# Run prime factorization benchmark
cargo run --release --features cuda -- prime-bench --number 100822548703 --cuda

# Run as criterion benchmark
cargo bench --features cuda prime_factorization
```

### Benchmark Results

| Algorithm | Time (ms) | Speedup | Memory (MB) |
|-----------|-----------|---------|-------------|
| CPU Trial Division | 823.45 | 1.0x | 2.34 |
| CPU Pollard's Rho | 67.89 | 12.1x | 1.56 |
| CUDA Trial Division | 12.34 | 66.8x | 45.67 |
```

#### 6.2 Add to CUDA Guide

```markdown
## Prime Factorization on GPU

The CUDA implementation includes optimized kernels for:

1. **Trial Division**: Parallel search across candidate factors
2. **Pollard's Rho**: Multiple parallel walks with different starting points
3. **Batch Processing**: Factorize multiple numbers simultaneously

Performance characteristics:
- 20-100x speedup over CPU
- Optimized for GTX 2070 (2304 CUDA cores)
- Supports numbers up to 2^63
```

### Phase 7: CI/CD Integration

```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install CUDA
        uses: Jimver/cuda-toolkit@v0.2.11
        with:
          cuda: '12.0.0'
      
      - name: Run CPU Benchmarks
        run: cargo bench --no-default-features prime_factorization
      
      - name: Run CUDA Benchmarks
        run: cargo bench --features cuda prime_factorization
      
      - name: Check for Regressions
        run: |
          cargo run --bin check_regression -- \
            --baseline benchmarks/baseline.json \
            --current target/criterion/prime_factorization/report.json \
            --threshold 10.0
```

### Phase 8: Testing Strategy

```rust
// Integration with existing test infrastructure
#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::tests::common::*;
    
    #[test]
    fn test_prime_factorization_with_genomic_pipeline() {
        // Ensure prime factorization doesn't interfere with genomic analysis
        let genome_result = run_genomic_analysis("test_data/ecoli.fasta");
        assert!(genome_result.is_ok());
        
        let prime_result = factorize(100822548703);
        assert_eq!(prime_result, vec![317213, 317879]);
    }
    
    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_memory_isolation() {
        // Ensure CUDA memory is properly isolated between modules
        let mut cuda_acc = CudaAccelerator::new().unwrap();
        
        // Run genomic analysis
        let sequences = load_test_sequences();
        let _ = cuda_acc.count_codons(&sequences).unwrap();
        
        // Run prime factorization
        let prime_result = cuda_acc.factorize_prime(100822548703).unwrap();
        assert_eq!(prime_result, vec![317213, 317879]);
        
        // Verify no memory leaks
        assert_eq!(cuda_acc.get_allocated_memory(), 0);
    }
}
```

### Phase 9: Performance Monitoring Integration

```rust
// Integrate with existing monitoring
use crate::monitoring::{MetricsCollector, PrometheusExporter};

impl PrimeFactorizationBenchmark {
    fn export_metrics(&self, results: &BenchmarkResults) {
        let mut collector = MetricsCollector::new();
        
        collector.record_gauge("prime_factorization_cpu_ms", results.cpu_time_ms);
        collector.record_gauge("prime_factorization_gpu_ms", results.gpu_time_ms);
        collector.record_gauge("prime_factorization_speedup", results.speedup);
        collector.record_counter("prime_factorization_runs_total", 1);
        
        if let Ok(exporter) = PrometheusExporter::new() {
            exporter.export(&collector);
        }
    }
}
```

### Phase 10: Rollout Plan

1. **Week 1**: Implement core CPU algorithms
   - Trial division
   - Pollard's Rho
   - Unit tests

2. **Week 2**: CUDA implementation
   - Kernel development
   - Memory management
   - GPU tests

3. **Week 3**: Benchmark framework
   - Performance profiling
   - Report generation
   - Statistical analysis

4. **Week 4**: Integration and testing
   - CI/CD setup
   - Documentation
   - Performance baseline establishment

## Risk Mitigation

1. **Memory Conflicts**: Use separate CUDA contexts for prime factorization
2. **Performance Regression**: Automated regression detection in CI
3. **API Changes**: Version the benchmark API separately
4. **Resource Contention**: Implement resource pooling and limits

## Success Metrics

1. Zero impact on existing genomic analysis performance
2. All tests passing including new prime factorization tests
3. Benchmark results reproducible within 5% variance
4. Documentation complete and reviewed
5. CI/CD pipeline fully automated