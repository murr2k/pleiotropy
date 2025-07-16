# Prime Factorization CUDA Implementation Plan

## Overview
This document outlines the implementation plan for adding prime factorization benchmarking to the existing CUDA-accelerated genomic cryptanalysis system.

## Architecture Summary

The pleiotropy project has a well-structured CUDA implementation with:
- **cudarc v0.10** as the CUDA library
- **ComputeBackend** for transparent CPU/GPU switching
- **CudaAccelerator** managing all GPU operations
- **4 existing kernels**: codon_counter, frequency_calculator, pattern_matcher, matrix_processor
- **GTX 2070** with 2304 CUDA cores and 8GB memory

## Integration Points

### 1. ComputeBackend (rust_impl/src/compute_backend.rs)
Add new method:
```rust
pub fn factorize_numbers(&mut self, numbers: &[u64]) -> Result<Vec<Vec<(u64, u32)>>>
```

### 2. CudaAccelerator (rust_impl/src/cuda/mod.rs)
Add new method:
```rust
pub fn prime_factorize(&mut self, numbers: &[u64]) -> CudaResult<Vec<Vec<(u64, u32)>>>
```

### 3. New Kernel Module (rust_impl/src/cuda/kernels/prime_factorizer.rs)
Create new module with:
- Trial division kernel
- Pollard's rho kernel
- Sieve of Eratosthenes kernel
- Miller-Rabin primality test kernel

### 4. Performance Integration
- Add timing to existing PerformanceMetrics struct
- Track factorizations per second
- Compare CPU vs GPU speedup

## Recommended Kernel Implementations

### Trial Division Kernel
```cuda
__global__ void trial_division_kernel(
    const uint64_t* numbers,
    uint64_t* factors,
    uint32_t* factor_counts,
    const uint32_t num_count
)
```
- Use shared memory for small prime cache
- Warp-level parallel division testing
- Early exit on complete factorization

### Pollard's Rho Kernel
```cuda
__global__ void pollard_rho_kernel(
    const uint64_t* numbers,
    uint64_t* factors,
    const uint32_t num_count,
    const uint32_t max_iterations
)
```
- Multiple parallel chains per number
- Different random seeds per thread
- Brent's improvement for cycle detection

### Sieve Kernel
```cuda
__global__ void segmented_sieve_kernel(
    uint8_t* is_prime,
    const uint64_t segment_start,
    const uint64_t segment_size
)
```
- Segmented sieve for large ranges
- Shared memory for marking composites
- Bank conflict-free access patterns

## Build and Test Strategy

1. **Feature Flag**: Use existing `cuda` feature
2. **Build Command**: `cargo build --release --features cuda`
3. **Unit Tests**: Create `test_prime_factorizer.rs` in cuda/tests/
4. **Benchmarks**: Add to `cuda_benchmarks.rs`
5. **Integration**: Test with compute_backend fallback

## Performance Targets

Based on GTX 2070 capabilities:
- **32-bit numbers**: 1 million factorizations/second
- **64-bit numbers**: 100K factorizations/second  
- **Sieve generation**: 1 billion primes/second
- **Memory bandwidth**: 300+ GB/s utilization

## Implementation Steps

1. Create `prime_factorizer.rs` kernel module
2. Implement basic trial division kernel
3. Add method to CudaAccelerator
4. Integrate with ComputeBackend
5. Add CPU fallback implementation
6. Create comprehensive tests
7. Benchmark and optimize
8. Add advanced algorithms (Pollard's rho, etc.)

## Files to Modify

1. `rust_impl/src/cuda/kernels/mod.rs` - Add module export
2. `rust_impl/src/cuda/mod.rs` - Add prime_factorize method
3. `rust_impl/src/compute_backend.rs` - Add factorize_numbers method
4. `rust_impl/src/cuda/kernels/prime_factorizer.rs` - Create new file
5. `rust_impl/src/cuda/tests/test_prime_factorizer.rs` - Create test file
6. `rust_impl/benches/cuda_benchmarks.rs` - Add benchmarks

## Success Criteria

- 20-50x speedup over CPU implementation
- Support for 32-bit, 64-bit, and 128-bit integers
- Correct factorization of all test cases
- Integration with existing performance monitoring
- Seamless CPU/GPU switching