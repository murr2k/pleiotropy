# Building the Genomic Cryptanalysis Project with CUDA Support

## Prerequisites

### System Requirements
- NVIDIA GPU with Compute Capability 5.2 or higher (GTX 2070 is CC 7.5)
- CUDA Toolkit 11.0 or newer
- Rust 1.70 or newer
- C++ compiler (gcc/clang on Linux, MSVC on Windows)

### CUDA Installation

#### Linux (Ubuntu/Debian)
```bash
# Install CUDA Toolkit (recommended: 11.8 or 12.0)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# Set environment variables
echo 'export CUDA_PATH=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_PATH/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
nvidia-smi
```

#### Windows
1. Download CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
2. Run the installer and follow the prompts
3. Add CUDA to PATH (usually done automatically)
4. Verify with `nvcc --version` in Command Prompt

#### macOS
Note: CUDA support on macOS is deprecated. Consider using Linux or Windows for CUDA development.

## Building the Project

### Build with CUDA Support
```bash
cd rust_impl

# Build with CUDA support
cargo build --release --features cuda

# Build with CUDA debug information
cargo build --features cuda-debug

# Force CUDA requirement (fails if CUDA not found)
CUDA_REQUIRED=1 cargo build --release --features cuda
```

### Build without CUDA (CPU only)
```bash
cargo build --release
```

### Verify CUDA Detection
```bash
# Check if CUDA was detected during build
cargo build --features cuda 2>&1 | grep "Found CUDA"

# Run with CUDA info
cargo run --features cuda -- --cuda-info
```

## Troubleshooting

### Common Issues

#### 1. CUDA Not Found
```
error: CUDA feature enabled but CUDA toolkit not found!
```
**Solution**: Set the CUDA_PATH environment variable:
```bash
export CUDA_PATH=/usr/local/cuda-11.8  # Adjust version as needed
```

#### 2. Library Linking Errors
```
error: linking with `cc` failed
```
**Solution**: Ensure CUDA libraries are in the linker path:
```bash
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

#### 3. Version Mismatch
```
warning: CUDA version X is older than recommended version 11.0
```
**Solution**: Update CUDA toolkit to version 11.0 or newer.

#### 4. Windows Specific Issues
- Ensure Visual Studio with C++ support is installed
- Use x64 Native Tools Command Prompt
- Check that CUDA and VS versions are compatible

### Checking GPU Compatibility
```bash
# Check your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Expected output for GTX 2070:
# compute_cap
# 7.5
```

## Performance Optimization

### Compile-time Optimizations
The build script automatically sets:
- Architecture: `-arch=sm_75` for GTX 2070
- Optimization: `-O3 --use_fast_math`
- Multiple compute capabilities for compatibility

### Runtime Configuration
Set these environment variables for optimal performance:
```bash
# Force GPU selection (0 = first GPU)
export CUDA_VISIBLE_DEVICES=0

# Enable persistent mode for lower latency
sudo nvidia-smi -pm 1

# Set GPU to maximum performance
sudo nvidia-smi -ac 4004,1710  # GTX 2070 specific
```

## Testing CUDA Functionality

### Unit Tests
```bash
# Run CUDA-specific tests
cargo test --features cuda cuda::

# Run all tests including CUDA
cargo test --all-features
```

### Benchmarks
```bash
# Run CUDA benchmarks
cargo bench --features cuda

# Compare CPU vs CUDA performance
cargo bench --features cuda -- --baseline cpu
```

### Integration Test
```bash
# Run full E. coli analysis with CUDA
cargo run --release --features cuda -- \
    --genome ../genome_research/genomes/e_coli_k12.fasta \
    --traits ../genome_research/trait_definitions/e_coli_traits.json \
    --use-cuda
```

## Build Configurations

### Development Build
Faster compilation, includes debug symbols:
```toml
[profile.dev]
opt-level = 0
debug = true
```

### Release Build
Maximum performance:
```toml
[profile.release]
lto = true          # Link-time optimization
opt-level = 3       # Maximum optimization
codegen-units = 1   # Better optimization
```

### CUDA-specific Profile
Create a custom profile for CUDA development:
```toml
[profile.cuda-dev]
inherits = "dev"
opt-level = 2  # Some optimization for kernel testing
```

Build with: `cargo build --profile cuda-dev --features cuda`

## Dependencies

The project uses these CUDA-related crates:
- `cudarc` (0.10): High-level CUDA bindings
- `cuda-runtime-sys` (0.3): Low-level runtime API bindings
- `cuda-driver-sys` (0.3): Low-level driver API bindings
- `bytemuck` (1.14): Safe data transmutation for GPU transfers
- `half` (2.3): FP16 support for newer GPUs

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `CUDA_PATH` | CUDA installation root | `/usr/local/cuda` |
| `CUDA_VISIBLE_DEVICES` | GPU selection | `0` or `0,1` |
| `CUDA_REQUIRED` | Fail if CUDA not found | `1` |
| `NVCC_FLAGS` | Additional compiler flags | `-arch=sm_75` |

## Monitoring GPU Usage

During execution, monitor GPU utilization:
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Detailed profiling
nsys profile cargo run --release --features cuda -- [args]
```

## Next Steps

1. Build the project with CUDA support
2. Run the CUDA tests to verify functionality
3. Benchmark your specific GPU performance
4. Optimize kernel parameters for your hardware

For kernel implementation details, see `src/cuda/architecture.md`.