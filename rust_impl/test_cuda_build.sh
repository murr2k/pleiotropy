#!/bin/bash
# Test script for CUDA build configuration

set -e

echo "Testing CUDA build configuration for Genomic Cryptanalysis..."
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for CUDA installation
echo -n "Checking for CUDA installation... "
if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}Found${NC}"
    nvcc --version | head -n 4
else
    echo -e "${RED}Not found${NC}"
    echo "Please install CUDA toolkit or add it to PATH"
fi

echo ""

# Check for NVIDIA GPU
echo -n "Checking for NVIDIA GPU... "
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}Found${NC}"
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv | head -n 2
else
    echo -e "${YELLOW}nvidia-smi not found${NC}"
fi

echo ""

# Check CUDA environment variables
echo "Checking CUDA environment variables:"
for var in CUDA_PATH CUDA_ROOT CUDA_TOOLKIT_ROOT_DIR; do
    value="${!var}"
    if [ -n "$value" ]; then
        echo -e "  $var = ${GREEN}$value${NC}"
    else
        echo -e "  $var = ${YELLOW}(not set)${NC}"
    fi
done

echo ""

# Test CPU build
echo "Testing CPU-only build..."
if cargo build --release 2>&1 | grep -q "error"; then
    echo -e "${RED}CPU build failed${NC}"
    exit 1
else
    echo -e "${GREEN}CPU build successful${NC}"
fi

echo ""

# Test CUDA build
echo "Testing CUDA build..."
if cargo build --release --features cuda 2>&1 | tee /tmp/cuda_build.log | grep -q "error"; then
    echo -e "${RED}CUDA build failed${NC}"
    echo "Check /tmp/cuda_build.log for details"
    exit 1
else
    if grep -q "Found CUDA" /tmp/cuda_build.log; then
        echo -e "${GREEN}CUDA build successful - CUDA detected${NC}"
    else
        echo -e "${YELLOW}CUDA build successful - CUDA not detected (CPU fallback)${NC}"
    fi
fi

echo ""

# Run CUDA feature detection
echo "Running CUDA feature detection test..."
cargo test --features cuda cuda::features::test_cuda_detection -- --nocapture

echo ""

# Check binary sizes
echo "Comparing binary sizes:"
CPU_SIZE=$(du -h target/release/genomic_cryptanalysis 2>/dev/null | cut -f1)
cargo build --release --features cuda &>/dev/null
CUDA_SIZE=$(du -h target/release/genomic_cryptanalysis 2>/dev/null | cut -f1)

echo "  CPU-only binary: ${CPU_SIZE:-N/A}"
echo "  CUDA binary: ${CUDA_SIZE:-N/A}"

echo ""
echo "=============================================="
echo -e "${GREEN}CUDA build test complete!${NC}"
echo ""
echo "Next steps:"
echo "1. If CUDA was not detected, install CUDA toolkit and set CUDA_PATH"
echo "2. Run benchmarks to compare CPU vs CUDA performance"
echo "3. See BUILDING_WITH_CUDA.md for detailed instructions"