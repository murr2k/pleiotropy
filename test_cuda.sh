#!/bin/bash
# CUDA Testing Script for Genomic Pleiotropy Cryptanalysis
# Runs comprehensive tests and benchmarks for CUDA implementation

set -e

echo "======================================"
echo "CUDA Testing Suite"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}WARNING: nvidia-smi not found. CUDA may not be available.${NC}"
else
    echo -e "${GREEN}CUDA Device Information:${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
    echo ""
fi

cd rust_impl

# Build with CUDA feature
echo -e "${YELLOW}Building with CUDA support...${NC}"
cargo build --release --features cuda
echo ""

# Run unit tests
echo -e "${YELLOW}Running CUDA unit tests...${NC}"
cargo test --features cuda -- --test-threads=1 --nocapture cuda::tests::test_kernel_correctness
echo ""

# Run integration tests
echo -e "${YELLOW}Running CUDA integration tests...${NC}"
cargo test --features cuda -- --test-threads=1 --nocapture cuda::tests::test_full_pipeline_integration
echo ""

# Run performance benchmarks
echo -e "${YELLOW}Running CUDA performance benchmarks...${NC}"
cargo test --features cuda -- --test-threads=1 --nocapture cuda::tests::test_performance_benchmarks
echo ""

# Run existing CUDA tests
echo -e "${YELLOW}Running existing CUDA tests...${NC}"
cargo test --features cuda cuda:: -- --test-threads=1
echo ""

# Run criterion benchmarks (if available)
if [ -f "benches/cuda_benchmarks.rs" ]; then
    echo -e "${YELLOW}Running criterion benchmarks...${NC}"
    cargo bench --features cuda cuda_benchmarks
    echo ""
fi

# Generate test report
echo -e "${YELLOW}Generating test report...${NC}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="../reports/cuda_test_$TIMESTAMP"
mkdir -p "$REPORT_DIR"

# Run tests with JSON output for report
cargo test --features cuda -- --format json > "$REPORT_DIR/test_results.json" 2>&1 || true

# Create summary report
cat > "$REPORT_DIR/summary.md" << EOF
# CUDA Test Report
Generated: $(date)

## System Information
$(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv || echo "CUDA not available")

## Test Summary
- Unit Tests: See test_results.json
- Performance Benchmarks: Completed
- Integration Tests: Completed

## Recommendations
1. Ensure all CUDA kernels pass correctness tests
2. Monitor performance benchmarks for regressions
3. Check memory usage during large genome processing
4. Verify CPU/GPU result consistency

## Next Steps
- Run with real E. coli genome data
- Profile with NVIDIA Nsight
- Optimize based on GTX 2070 specifications
EOF

echo -e "${GREEN}Test report saved to: $REPORT_DIR/summary.md${NC}"
echo ""

# Performance comparison summary
echo -e "${YELLOW}Performance Summary:${NC}"
echo "Run the following command to see detailed benchmark results:"
echo "cargo test --features cuda test_performance_benchmarks -- --nocapture"
echo ""

echo -e "${GREEN}CUDA testing completed successfully!${NC}"