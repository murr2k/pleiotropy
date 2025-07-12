#!/bin/bash
# Build script for CUDA-enabled genomic cryptanalysis

set -e

echo "=== Building Genomic Cryptanalysis with CUDA Support ==="
echo

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo
else
    echo "WARNING: nvidia-smi not found. CUDA may not be available."
    echo
fi

# Check for CUDA toolkit
if command -v nvcc &> /dev/null; then
    echo "CUDA toolkit found:"
    nvcc --version | head -n 4
    echo
else
    echo "WARNING: nvcc not found. CUDA toolkit may not be installed."
    echo "Install CUDA toolkit from: https://developer.nvidia.com/cuda-downloads"
    echo
fi

# Clean previous builds
echo "Cleaning previous builds..."
cargo clean

# Build with CUDA feature
echo "Building with CUDA support..."
cargo build --release --features cuda

# Run tests
echo
echo "Running tests..."
cargo test --features cuda -- --nocapture

# Run CUDA-specific integration tests
echo
echo "Running CUDA integration tests..."
cargo test --features cuda cuda_integration_tests -- --nocapture

# Build documentation
echo
echo "Building documentation..."
cargo doc --features cuda --no-deps

echo
echo "=== Build Complete ==="
echo
echo "To run with CUDA support:"
echo "  cargo run --release --features cuda"
echo
echo "To disable CUDA at runtime:"
echo "  PLEIOTROPY_FORCE_CPU=1 cargo run --release --features cuda"
echo
echo "Performance statistics will be displayed showing GPU vs CPU usage."