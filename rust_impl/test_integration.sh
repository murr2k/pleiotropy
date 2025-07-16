#!/bin/bash

# Integration test script for CUDA benchmark system

echo "=== CUDA Benchmark Integration Test ==="
echo

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    echo "ERROR: Cargo not found. Please install Rust."
    echo "Visit: https://rustup.rs/"
    exit 1
fi

# Build with CUDA features
echo "Building with CUDA support..."
cargo build --release --features cuda

if [ $? -ne 0 ]; then
    echo "WARNING: CUDA build failed, trying CPU-only build..."
    cargo build --release
fi

# Run tests
echo
echo "Running unit tests..."
cargo test --features cuda -- --nocapture

# Run benchmark if build succeeded
if [ -f target/release/benchmark ]; then
    echo
    echo "Running benchmarks..."
    ./target/release/benchmark --output integration_test_results.txt
else
    echo "Benchmark binary not found, build may have failed"
fi

echo
echo "=== Integration Test Complete ==="