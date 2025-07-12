# CUDA Integration for Genomic Pleiotropy Cryptanalysis

## Overview

This document describes the CUDA integration for accelerating genomic cryptanalysis computations. The integration provides seamless CPU/GPU switching with automatic fallback mechanisms.

## Architecture

### Unified Compute Backend

The `ComputeBackend` module provides a unified interface that automatically switches between CPU and CUDA implementations based on:
- Hardware availability
- User preferences
- Operation complexity
- Data size thresholds

### Key Components

1. **ComputeBackend (`src/compute_backend.rs`)**
   - Unified API for CPU/GPU operations
   - Automatic fallback on CUDA failures
   - Performance statistics tracking
   - Runtime GPU selection

2. **CUDA Kernels (`src/cuda/kernels/`)**
   - `codon_counter.rs`: Parallel codon counting
   - `frequency_calculator.rs`: GPU-accelerated frequency analysis
   - `pattern_matcher.rs`: High-speed pattern matching
   - `matrix_processor.rs`: Matrix operations and eigenanalysis

3. **GPU-Enhanced Trait Extraction (`src/trait_extractor_gpu.rs`)**
   - Batch processing for GPU efficiency
   - Automatic CPU/GPU selection based on data size
   - Parallel trait pattern detection

## Building with CUDA Support

### Prerequisites

1. NVIDIA GPU with compute capability 5.2 or higher
2. CUDA Toolkit 11.0 or newer
3. Rust 1.70 or newer

### Build Commands

```bash
# Build with CUDA support
cargo build --release --features cuda

# Run tests including CUDA integration tests
cargo test --features cuda

# Use the build script for complete setup
./build_cuda.sh
```

## Usage

### Basic Usage

```rust
use genomic_cryptanalysis::GenomicCryptanalysis;

// CUDA is automatically detected and used if available
let mut analyzer = GenomicCryptanalysis::new();

// Check if CUDA is enabled
if analyzer.is_cuda_enabled() {
    println!("Using GPU acceleration");
}

// Force CPU usage for testing
analyzer.set_force_cpu(true);

// Get performance statistics
let stats = analyzer.get_performance_stats();
println!("GPU calls: {}, CPU calls: {}", stats.cuda_calls, stats.cpu_calls);
```

### Environment Variables

- `PLEIOTROPY_FORCE_CPU=1`: Force CPU usage even if CUDA is available
- `CUDA_VISIBLE_DEVICES`: Control which GPUs are visible to the application

## Performance Optimization

### Automatic GPU Selection

The system automatically uses GPU for:
- Large sequence datasets (>10 sequences)
- Batch operations with sufficient parallelism
- Matrix operations larger than 100x100

### Memory Management

- Pinned memory for efficient CPU-GPU transfers
- Streaming for large datasets
- Automatic memory pooling

### Performance Monitoring

The compute backend tracks:
- Number of CPU vs GPU calls
- Average execution times
- Failure rates and fallback statistics
- Total sequences processed

## API Changes

### New Methods in GenomicCryptanalysis

```rust
// Check GPU status
pub fn is_cuda_enabled(&self) -> bool

// Get performance statistics
pub fn get_performance_stats(&self) -> &PerformanceStats

// Control GPU usage
pub fn set_force_cpu(&mut self, force: bool)
```

### Performance Statistics Structure

```rust
pub struct PerformanceStats {
    pub cpu_calls: usize,
    pub cuda_calls: usize,
    pub cuda_failures: usize,
    pub total_sequences_processed: usize,
    pub avg_cpu_time_ms: f64,
    pub avg_cuda_time_ms: f64,
}
```

## Backward Compatibility

The CUDA integration maintains full backward compatibility:
- CPU-only builds work without any CUDA dependencies
- The same API works regardless of CUDA availability
- Automatic fallback ensures reliability

## Troubleshooting

### CUDA Not Detected

1. Check GPU availability: `nvidia-smi`
2. Verify CUDA installation: `nvcc --version`
3. Ensure the `cuda` feature is enabled in build
4. Check `CUDA_VISIBLE_DEVICES` environment variable

### Performance Issues

1. Monitor GPU utilization: `nvidia-smi dmon`
2. Check batch sizes in performance stats
3. Verify data is large enough to benefit from GPU
4. Consider memory transfer overhead for small datasets

### Build Errors

1. Ensure CUDA toolkit matches GPU driver version
2. Check Rust toolchain is up to date
3. Verify all CUDA dependencies are installed
4. Try the alternative CUDA backend: `--features cuda-alt`

## Benchmarking

Run benchmarks to compare CPU vs GPU performance:

```bash
# Run with GPU
cargo bench --features cuda

# Force CPU for comparison
PLEIOTROPY_FORCE_CPU=1 cargo bench --features cuda
```

Expected speedups:
- Codon counting: 10-20x
- Pattern matching: 15-30x
- Matrix operations: 20-50x
- End-to-end pipeline: 5-15x

## Future Enhancements

1. Multi-GPU support for very large genomes
2. Mixed precision computation for newer GPUs
3. CUDA graphs for reduced kernel launch overhead
4. Tensor Core utilization for matrix operations
5. Direct NeuroDNA neural network integration

## Integration with NeuroDNA

The CUDA backend is designed to work seamlessly with NeuroDNA:
- GPU-accelerated trait pattern matching
- Parallel neural network inference (planned)
- Shared memory pool for efficient data exchange

## Contributing

When adding new CUDA kernels:
1. Implement CPU fallback in `compute_backend.rs`
2. Add kernel to `src/cuda/kernels/`
3. Update integration tests
4. Document performance characteristics
5. Ensure error handling with automatic fallback