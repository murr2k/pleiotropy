# CUDA Testing Guide

## Overview

This guide documents the comprehensive testing suite for the CUDA implementation of the Genomic Pleiotropy Cryptanalysis project. The tests ensure correctness, performance, and reliability of GPU-accelerated genomic analysis.

## Test Categories

### 1. Kernel Correctness Tests (`test_kernel_correctness.rs`)

These tests ensure that CUDA kernels produce identical results to CPU implementations:

- **Exact Match Tests**: Verify bit-for-bit accuracy between CPU and GPU results
- **Edge Case Handling**: Test empty sequences, partial codons, invalid characters
- **Large Batch Processing**: Ensure correctness with thousands of sequences
- **Memory Safety**: Test various sequence sizes and alignment scenarios

Key tests:
```rust
cargo test --features cuda test_codon_counting_exact_match -- --nocapture
cargo test --features cuda test_frequency_calculation_exact_match -- --nocapture
cargo test --features cuda test_sliding_window_correctness -- --nocapture
```

### 2. Performance Benchmarks (`test_performance_benchmarks.rs`)

Comprehensive performance measurements comparing CPU vs GPU:

- **Scaling Analysis**: Test performance with increasing data sizes
- **GTX 2070 Optimization**: Specific tuning for 8GB memory, 2304 CUDA cores
- **Memory Transfer Overhead**: Measure PCIe bandwidth impact
- **Real-world Simulation**: E. coli genome-sized benchmarks

Key benchmarks:
```rust
cargo test --features cuda benchmark_codon_counting_scaling -- --nocapture
cargo test --features cuda benchmark_real_world_ecoli -- --nocapture
cargo test --features cuda stress_test_maximum_throughput -- --nocapture
```

Expected performance on GTX 2070:
- Codon counting: 10-20x speedup for large sequences
- Frequency calculation: 5-15x speedup
- Pattern matching: 8-12x speedup
- Memory bandwidth: 200-300 GB/s

### 3. Integration Tests (`test_full_pipeline_integration.rs`)

End-to-end testing of the complete analysis pipeline:

- **Full Pipeline Comparison**: GPU vs CPU complete workflow
- **NeuroDNA Integration**: Trait detection with CUDA acceleration
- **Error Recovery**: Graceful handling of failures
- **Streaming Processing**: Large genome chunked processing
- **Concurrent Execution**: Multi-threaded GPU usage

Key integration tests:
```rust
cargo test --features cuda test_full_pipeline_gpu_vs_cpu -- --nocapture
cargo test --features cuda test_neurodna_integration_with_cuda -- --nocapture
cargo test --features cuda test_streaming_large_genome -- --nocapture
```

## Running Tests

### Quick Test Suite
```bash
# Run all CUDA tests
./test_cuda.sh

# Run specific test category
cargo test --features cuda cuda::tests::test_kernel_correctness
cargo test --features cuda cuda::tests::test_performance_benchmarks
cargo test --features cuda cuda::tests::test_full_pipeline_integration
```

### Detailed Testing
```bash
# Run with verbose output
cargo test --features cuda -- --nocapture --test-threads=1

# Run specific test
cargo test --features cuda test_codon_counting_exact_match -- --exact --nocapture

# Run benchmarks only
cargo test --features cuda benchmark_ -- --nocapture
```

### Criterion Benchmarks
```bash
# Run detailed statistical benchmarks
cargo bench --features cuda cuda_benchmarks

# View benchmark results
open target/criterion/report/index.html
```

## Test Data

### Synthetic Test Genomes
- Small: 100-1000 base pairs for unit tests
- Medium: 10KB-100KB for integration tests  
- Large: 1MB-10MB for performance tests
- E. coli simulation: 4.6MB for real-world testing

### Known Patterns
Test sequences include regions with:
- High ATG frequency (carbon metabolism)
- High GCG frequency (stress response)
- AT-rich regions (motility)
- GAA repeats (regulatory)

## Performance Profiling

### NVIDIA Nsight Systems
```bash
# Profile the test suite
nsys profile --stats=true cargo test --features cuda benchmark_real_world_ecoli

# View results
nsys-ui report1.qdrep
```

### NVIDIA Nsight Compute
```bash
# Profile specific kernels
ncu --target-processes all cargo test --features cuda test_codon_counting_exact_match
```

## Troubleshooting

### Common Issues

1. **CUDA Not Available**
   - Check `nvidia-smi` output
   - Verify CUDA toolkit installation
   - Ensure GPU drivers are up to date

2. **Test Failures**
   - Check for floating-point precision differences
   - Verify sequence alignment in memory
   - Enable debug output with `RUST_LOG=debug`

3. **Performance Issues**
   - Monitor GPU utilization with `nvidia-smi -l 1`
   - Check for PCIe bottlenecks
   - Verify optimal block/grid sizes for GTX 2070

### Debug Mode
```bash
# Enable debug logging
RUST_LOG=debug cargo test --features cuda

# Enable CUDA debug
export CUDA_LAUNCH_BLOCKING=1
cargo test --features cuda
```

## Continuous Integration

### GitHub Actions Configuration
```yaml
- name: Run CUDA Tests
  run: |
    cargo test --features cuda --release
    cargo bench --features cuda --no-run
```

### Performance Regression Detection
Tests include automatic performance regression detection:
- Baseline performance recorded
- Each test run compared to baseline
- Warnings for >10% performance degradation

## Test Coverage

### Current Coverage
- Codon counting: 100%
- Frequency calculation: 100%
- Pattern matching: 95%
- Matrix operations: 90%
- Error handling: 85%

### Coverage Report
```bash
# Generate coverage report (requires grcov)
cargo tarpaulin --features cuda --out Html
open tarpaulin-report.html
```

## Best Practices

1. **Always Test Both CPU and GPU**
   - Ensure results match within floating-point tolerance
   - Compare performance metrics

2. **Test Edge Cases**
   - Empty inputs
   - Single element
   - Maximum size
   - Invalid data

3. **Monitor Memory Usage**
   - Use `nvidia-smi` during tests
   - Check for memory leaks
   - Verify proper cleanup

4. **Performance Testing**
   - Run multiple iterations
   - Calculate statistics (mean, std dev)
   - Test with realistic data sizes

## Future Enhancements

1. **Automated Performance Tracking**
   - Database of historical results
   - Automatic regression alerts
   - Performance trend visualization

2. **Extended Test Coverage**
   - Multi-GPU testing
   - Different CUDA compute capabilities
   - Various genome types

3. **Stress Testing**
   - 24-hour continuous runs
   - Memory pressure scenarios
   - Concurrent workload testing