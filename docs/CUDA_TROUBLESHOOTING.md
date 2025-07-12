# CUDA Troubleshooting Guide

Comprehensive guide for diagnosing and fixing CUDA-related issues in the Genomic Pleiotropy Cryptanalysis project.

## Table of Contents

1. [Common Issues](#common-issues)
2. [Installation Problems](#installation-problems)
3. [Runtime Errors](#runtime-errors)
4. [Performance Issues](#performance-issues)
5. [Memory Problems](#memory-problems)
6. [Compatibility Issues](#compatibility-issues)
7. [Debugging Techniques](#debugging-techniques)
8. [FAQ](#faq)

## Common Issues

### Issue: "CUDA not available" despite having NVIDIA GPU

**Symptoms:**
```
Error: CUDA not available on this system
```

**Diagnosis:**
```bash
# Check if NVIDIA driver is installed
nvidia-smi

# Check CUDA installation
nvcc --version

# Check if GPU is visible to CUDA
nvidia-smi -L

# Test CUDA functionality
cd rust_impl
cargo run --features cuda --example cuda_test
```

**Solutions:**

1. **Driver not installed:**
```bash
# Ubuntu
sudo apt update
sudo apt install nvidia-driver-535

# Reboot
sudo reboot
```

2. **CUDA toolkit missing:**
```bash
# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-8
```

3. **Path not set:**
```bash
# Add to ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

4. **GPU in compute mode:**
```bash
# Check compute mode
nvidia-smi -q | grep "Compute Mode"

# If "Exclusive Process", change to default:
sudo nvidia-smi -c 0
```

### Issue: Build fails with "cannot find -lcudart"

**Symptoms:**
```
error: linking with `cc` failed
  = note: /usr/bin/ld: cannot find -lcudart
```

**Solution:**
```bash
# Find CUDA libraries
find /usr -name "libcudart.so*" 2>/dev/null

# Add library path
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# For build.rs
export CUDA_PATH=/usr/local/cuda

# Rebuild
cargo clean
cargo build --features cuda
```

### Issue: "CudaError: out of memory"

**Symptoms:**
```
Error: GPU memory allocation failed: requested 2147483648 bytes, available 1073741824 bytes
```

**Quick fixes:**
```bash
# Check GPU memory usage
nvidia-smi

# Kill processes using GPU memory
sudo fuser -v /dev/nvidia*
sudo kill -9 <PID>

# Clear GPU memory cache
sudo nvidia-smi --gpu-reset
```

**Code solutions:**
```rust
// Reduce batch size
let batch_size = 1000; // Instead of 10000

// Enable memory pooling
let config = CudaConfig {
    memory_pool_size_mb: 1024,
    ..Default::default()
};

// Process in chunks
for chunk in sequences.chunks(batch_size) {
    let results = accelerator.process(chunk)?;
    // Handle results
}
```

## Installation Problems

### CUDA Version Mismatch

**Symptoms:**
```
CUDA driver version is insufficient for CUDA runtime version
```

**Check versions:**
```bash
# Driver version (top right)
nvidia-smi

# CUDA runtime version
nvcc --version

# Detailed info
nvidia-smi -q | grep "CUDA Version"
```

**Compatibility table:**
| Driver Version | CUDA Toolkit |
|----------------|--------------|
| 450.51+ | 11.0 |
| 460.32+ | 11.2 |
| 470.57+ | 11.4 |
| 510.47+ | 11.6 |
| 515.43+ | 11.7 |
| 525.60+ | 12.0 |
| 535.54+ | 12.2 |

**Fix version mismatch:**
```bash
# Option 1: Update driver
sudo apt update
sudo apt upgrade nvidia-driver-535

# Option 2: Install compatible CUDA version
# Remove old CUDA
sudo apt-get --purge remove "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "cuda*" "nsight*"

# Install specific version
sudo apt-get install cuda-11-8
```

### WSL2 Specific Issues

**GPU not detected in WSL2:**

1. **Check Windows version:**
```powershell
# In Windows PowerShell
winver
# Need Windows 11 or Windows 10 build 21H2+
```

2. **Update WSL2:**
```bash
# In Windows PowerShell
wsl --update
wsl --shutdown
```

3. **Install CUDA for WSL2:**
```bash
# Inside WSL2
# DO NOT install regular Linux NVIDIA drivers!
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-11-8
```

4. **Verify GPU access:**
```bash
# Should show GPU without installing Linux drivers
nvidia-smi
```

### Docker CUDA Issues

**GPU not available in container:**

1. **Install NVIDIA Container Toolkit:**
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

2. **Run with GPU support:**
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Run our container
docker run --gpus all -it genomic-cryptanalysis:cuda
```

## Runtime Errors

### Kernel Launch Failures

**Symptoms:**
```
CudaError: Kernel execution failed: invalid configuration argument
```

**Common causes and fixes:**

1. **Block size too large:**
```rust
// Check max threads per block
let props = device.properties();
let max_threads = props.max_threads_per_block; // Usually 1024

// Use safe block size
let block_size = 256; // Safe for all GPUs
```

2. **Grid size exceeds limits:**
```rust
// Check grid limits
let max_grid_x = props.max_grid_dims.0; // Usually 2^31-1

// Calculate safe grid size
let grid_size = (total_work + block_size - 1) / block_size;
let grid_size = grid_size.min(max_grid_x as usize);
```

3. **Shared memory exceeds limit:**
```rust
// Check shared memory limit
let max_shared = props.max_shared_memory_per_block; // Usually 48KB

// Reduce shared memory usage
let shared_size = std::mem::size_of::<f32>() * 256; // Example
assert!(shared_size <= max_shared);
```

### Memory Access Violations

**Symptoms:**
```
CUDA_ERROR_ILLEGAL_ADDRESS
Segmentation fault (core dumped)
```

**Debug with cuda-memcheck:**
```bash
# Install cuda-memcheck (if not present)
sudo apt-get install cuda-memcheck

# Run with memory checking
cuda-memcheck ./target/release/genomic-cryptanalysis analyze genome.fasta

# Detailed checking
cuda-memcheck --leak-check full --racecheck-report all ./your_program
```

**Common fixes:**

1. **Bounds checking:**
```rust
// Add bounds checking in kernels
if thread_id >= data_size { return; }

// Ensure array access is valid
let index = blockIdx.x * blockDim.x + threadIdx.x;
if index < array_length {
    array[index] = value;
}
```

2. **Alignment issues:**
```rust
// Ensure proper alignment
#[repr(align(16))]
struct AlignedData {
    values: [f32; 4],
}
```

### Synchronization Errors

**Symptoms:**
```
Race condition detected
Incorrect results intermittently
```

**Fix synchronization:**
```rust
// Add proper synchronization
cuda_device.synchronize()?;

// Use atomic operations
atomicAdd(&counter[index], 1);

// Add memory fences
__threadfence();
```

## Performance Issues

### Slow GPU Performance

**Diagnosis checklist:**

1. **Check GPU utilization:**
```bash
# Real-time monitoring
watch -n 0.1 nvidia-smi

# Detailed utilization
nvidia-smi dmon -i 0
```

2. **Profile the application:**
```bash
# Using nvprof (deprecated but still useful)
nvprof ./target/release/genomic-cryptanalysis analyze genome.fasta

# Using Nsight Systems (recommended)
nsys profile -o profile.nsys-rep ./target/release/genomic-cryptanalysis analyze genome.fasta
nsys-ui profile.nsys-rep
```

3. **Check for throttling:**
```bash
# Monitor clocks and temperature
nvidia-smi -q -d CLOCK,TEMPERATURE -l 1

# Check throttle reasons
nvidia-smi -q | grep -A 20 "Clocks Throttle Reasons"
```

**Common performance fixes:**

1. **Insufficient work:**
```rust
// Ensure enough parallel work
if sequences.len() < 1000 {
    // Use CPU instead
    return cpu_process(sequences);
}
```

2. **Memory transfer overhead:**
```rust
// Batch transfers
let mut all_sequences = Vec::new();
for batch in batches {
    all_sequences.extend(batch);
}
// Single transfer instead of multiple
gpu.process_all(&all_sequences)?;

// Use pinned memory
let pinned_buffer = PinnedBuffer::new(size)?;
```

3. **Kernel occupancy:**
```rust
// Check occupancy
let occupancy = calculate_occupancy(block_size, registers_per_thread, shared_mem_per_block);
println!("Kernel occupancy: {}%", occupancy * 100.0);

// Optimize block size
let optimal_block_size = find_optimal_block_size(kernel_func)?;
```

### Memory Bandwidth Bottlenecks

**Identify bandwidth issues:**
```bash
# Profile memory access
nvprof --metrics gld_throughput,gst_throughput ./your_program

# Check efficiency
nvprof --metrics gld_efficiency,gst_efficiency ./your_program
```

**Optimization strategies:**

1. **Coalesced access:**
```cuda
// Bad - strided access
data[threadIdx.x * stride]

// Good - coalesced access  
data[blockIdx.x * blockDim.x + threadIdx.x]
```

2. **Use shared memory:**
```cuda
__shared__ float tile[TILE_SIZE];
tile[threadIdx.x] = global_data[global_idx];
__syncthreads();
```

3. **Optimize data layout:**
```rust
// Structure of Arrays (SoA) instead of Array of Structures (AoS)
struct SoA {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
}
```

## Memory Problems

### Memory Leak Detection

**Check for leaks:**
```bash
# Using cuda-memcheck
cuda-memcheck --leak-check full ./target/release/genomic-cryptanalysis

# Monitor memory usage over time
while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    sleep 1
done | tee memory_log.txt
```

**Common leak sources:**

1. **Forgot to free CUDA memory:**
```rust
// Use RAII wrappers
let buffer = CudaBuffer::new(size)?; // Automatically freed on drop

// Or explicitly free
unsafe {
    cudaFree(ptr);
}
```

2. **Circular references:**
```rust
// Avoid circular references in GPU memory
// Use weak references or indices instead
```

### Out of Memory Strategies

**Progressive degradation:**
```rust
impl ComputeBackend {
    pub fn process_adaptive(&mut self, data: &[Data]) -> Result<Output> {
        // Try full GPU processing
        match self.gpu_process_full(data) {
            Ok(result) => return Ok(result),
            Err(CudaError::AllocationFailed { .. }) => {
                log::warn!("GPU OOM, trying smaller batch");
            }
            Err(e) => return Err(e.into()),
        }
        
        // Try smaller batches
        let batch_sizes = [1000, 500, 100, 50];
        for &batch_size in &batch_sizes {
            match self.gpu_process_batched(data, batch_size) {
                Ok(result) => return Ok(result),
                Err(CudaError::AllocationFailed { .. }) => continue,
                Err(e) => return Err(e.into()),
            }
        }
        
        // Fall back to CPU
        log::warn!("GPU memory exhausted, using CPU");
        self.cpu_process(data)
    }
}
```

## Compatibility Issues

### GPU Architecture Mismatches

**Check compute capability:**
```bash
# Your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Required compute capability
# Our minimum: 5.2 (Maxwell)
```

**Handle architecture differences:**
```rust
#[cfg(feature = "cuda")]
pub fn select_kernel_variant(device: &CudaDevice) -> KernelVariant {
    let (major, minor) = device.compute_capability();
    
    match (major, minor) {
        (8, 6) | (8, 9) => KernelVariant::Ampere,    // RTX 30xx, A100
        (7, 5) => KernelVariant::Turing,              // RTX 20xx
        (7, 0) => KernelVariant::Volta,               // V100
        (6, _) => KernelVariant::Pascal,              // GTX 10xx
        (5, _) => KernelVariant::Maxwell,             // GTX 9xx
        _ => KernelVariant::Generic,                  // Fallback
    }
}
```

### Multi-GPU Heterogeneous Systems

**Handle different GPU types:**
```rust
pub fn select_best_gpu() -> Result<i32> {
    let count = CudaDevice::count();
    let mut best_score = 0;
    let mut best_device = 0;
    
    for i in 0..count {
        let device = CudaDevice::new(i)?;
        let props = device.properties();
        
        // Score based on compute capability and memory
        let score = props.major * 1000 + 
                   props.minor * 100 + 
                   (props.total_memory / 1_000_000_000) as i32;
        
        if score > best_score {
            best_score = score;
            best_device = i;
        }
    }
    
    Ok(best_device)
}
```

## Debugging Techniques

### Enable Debug Mode

**Compile with debug info:**
```bash
# In Cargo.toml
[profile.release]
debug = true

# Build
cargo build --release --features cuda,cuda-debug
```

**Set debug environment:**
```bash
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_DEBUG=1
export RUST_LOG=genomic_cryptanalysis::cuda=debug
export RUST_BACKTRACE=full
```

### Using CUDA-GDB

**Debug CUDA kernels:**
```bash
# Install cuda-gdb
sudo apt-get install cuda-gdb

# Compile with debug info
nvcc -g -G kernel.cu -c -o kernel.o

# Debug
cuda-gdb ./target/release/genomic-cryptanalysis
(cuda-gdb) set cuda break_on_launch application
(cuda-gdb) run analyze genome.fasta
(cuda-gdb) info cuda kernels
(cuda-gdb) cuda kernel 0 block 0,0,0 thread 0,0,0
(cuda-gdb) print threadIdx
```

### Printf Debugging in Kernels

```cuda
__global__ void debug_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Print from first thread only
    if (idx == 0) {
        printf("Kernel launched with %d threads\n", gridDim.x * blockDim.x);
        printf("First value: %f\n", data[0]);
    }
    
    // Print specific thread
    if (idx == 100) {
        printf("Thread 100: data[%d] = %f\n", idx, data[idx]);
    }
    
    // Conditional printing
    if (data[idx] < 0) {
        printf("Warning: negative value at index %d: %f\n", idx, data[idx]);
    }
}
```

### Memory Debugging

**Track allocations:**
```rust
#[cfg(feature = "cuda-debug")]
pub struct DebugAllocator {
    allocations: Mutex<HashMap<*mut u8, (usize, String)>>,
}

impl DebugAllocator {
    pub fn alloc(&self, size: usize, location: &str) -> CudaResult<*mut u8> {
        let ptr = unsafe { cuda_malloc(size)? };
        self.allocations.lock().unwrap().insert(
            ptr,
            (size, location.to_string())
        );
        log::debug!("Allocated {} bytes at {} from {}", size, ptr as usize, location);
        Ok(ptr)
    }
    
    pub fn free(&self, ptr: *mut u8) -> CudaResult<()> {
        if let Some((size, location)) = self.allocations.lock().unwrap().remove(&ptr) {
            log::debug!("Freeing {} bytes at {} allocated from {}", size, ptr as usize, location);
            unsafe { cuda_free(ptr) }
        } else {
            log::warn!("Attempting to free untracked pointer: {}", ptr as usize);
            Err(CudaError::InvalidPointer)
        }
    }
    
    pub fn report_leaks(&self) {
        let allocations = self.allocations.lock().unwrap();
        if !allocations.is_empty() {
            log::error!("Memory leaks detected:");
            for (ptr, (size, location)) in allocations.iter() {
                log::error!("  {} bytes at {} from {}", size, *ptr as usize, location);
            }
        }
    }
}
```

## FAQ

### Q: Why is GPU slower than CPU for small inputs?

**A:** GPU has overhead for:
- Data transfer (PCIe)
- Kernel launch
- Memory allocation

**Rule of thumb:** Use GPU when:
- Genome > 1 Mbp
- Batch size > 1000 sequences
- Computation > 100ms on CPU

### Q: How do I know if I'm using GPU effectively?

**Check these metrics:**
```bash
# GPU utilization should be >80%
nvidia-smi dmon -i 0

# Memory bandwidth should be >70% of theoretical
nvprof --metrics gld_throughput,gst_throughput ./program

# No kernel launch gaps
nsys profile --stats=true ./program
```

### Q: Can I use multiple GPUs?

**Yes, but consider:**
- Data distribution overhead
- GPU-to-GPU communication cost
- Load balancing

**Example multi-GPU:**
```rust
let num_gpus = CudaDevice::count();
let chunk_size = data.len() / num_gpus;

let handles: Vec<_> = (0..num_gpus)
    .map(|gpu_id| {
        let chunk = data[gpu_id * chunk_size..(gpu_id + 1) * chunk_size].to_vec();
        std::thread::spawn(move || {
            let mut acc = CudaAccelerator::new_with_device(gpu_id)?;
            acc.process(&chunk)
        })
    })
    .collect();
```

### Q: How do I handle GPU errors gracefully?

**Use the automatic fallback system:**
```rust
let backend = ComputeBackend::new()?;
// Automatically uses GPU if available, CPU if not
let results = backend.analyze(genome)?;

// Check what was actually used
if backend.get_stats().cuda_calls > 0 {
    println!("Used GPU acceleration");
} else {
    println!("Used CPU processing");
}
```

### Q: What about AMD GPUs?

Currently not supported, but planned:
- ROCm port in development
- Similar API through HIP
- Expected Q3 2024

### Q: How do I report a bug?

1. **Collect system info:**
```bash
./scripts/collect_debug_info.sh > debug_info.txt
```

2. **Minimal reproducer:**
```rust
// Create minimal test case
#[test]
fn reproduce_issue() {
    let data = vec![...]; // Minimal data
    let result = process(data);
    assert!(result.is_err());
}
```

3. **File issue:** https://github.com/genomic-pleiotropy/cryptanalysis/issues

Include:
- System info
- Error messages
- Minimal reproducer
- Expected vs actual behavior

---

*Last updated: January 2024*  
*For urgent support: cuda-support@genomic-cryptanalysis.org*