/// CUDA implementation for semiprime factorization
/// Provides GPU-accelerated factorization of numbers with exactly two prime factors

use crate::semiprime_factorization::{SemiprimeResult, is_prime};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;
use std::time::Instant;
use anyhow::{Result, Context};

/// CUDA kernel source for semiprime factorization
const SEMIPRIME_KERNEL: &str = r#"
extern "C" __global__ void factorize_semiprimes_kernel(
    const unsigned long long* numbers,
    unsigned long long* factors1,
    unsigned long long* factors2,
    unsigned char* success,
    const int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    unsigned long long n = numbers[idx];
    success[idx] = 0;
    
    // Quick check for even numbers
    if (n % 2 == 0) {
        unsigned long long other = n / 2;
        // Simple primality check (would be optimized in real implementation)
        if (is_prime_device(other)) {
            factors1[idx] = 2;
            factors2[idx] = other;
            success[idx] = 1;
            return;
        }
    }
    
    // Trial division with thread cooperation
    unsigned long long sqrt_n = (unsigned long long)sqrtf((float)n) + 1;
    
    // Each thread checks different candidates
    for (unsigned long long i = 3 + 2 * idx; i <= sqrt_n; i += 2 * blockDim.x * gridDim.x) {
        if (n % i == 0) {
            unsigned long long other = n / i;
            if (is_prime_device(i) && is_prime_device(other)) {
                factors1[idx] = i;
                factors2[idx] = other;
                success[idx] = 1;
                return;
            }
        }
    }
}

// Device function for primality testing
__device__ bool is_prime_device(unsigned long long n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    
    unsigned long long sqrt_n = (unsigned long long)sqrtf((float)n) + 1;
    for (unsigned long long i = 3; i <= sqrt_n; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}
"#;

/// CUDA accelerator for semiprime factorization
pub struct SemiprimeCuda {
    device: Arc<CudaDevice>,
    module: cudarc::driver::CudaModule,
}

impl SemiprimeCuda {
    /// Create a new CUDA semiprime factorizer
    pub fn new() -> Result<Self> {
        let device = CudaDevice::new(0)
            .context("Failed to initialize CUDA device")?;
        
        let ptx = cudarc::nvrtc::compile_ptx(SEMIPRIME_KERNEL)
            .context("Failed to compile CUDA kernel")?;
        
        let module = device.load_ptx(ptx, "semiprime", &["factorize_semiprimes_kernel"])
            .context("Failed to load PTX module")?;
        
        Ok(Self {
            device: Arc::new(device),
            module,
        })
    }
    
    /// Factorize a batch of semiprimes on GPU
    pub fn factorize_batch(&self, numbers: &[u64]) -> Result<Vec<SemiprimeResult>> {
        let start = Instant::now();
        let count = numbers.len();
        
        // Allocate device memory
        let d_numbers = self.device.htod_copy(numbers)
            .context("Failed to copy numbers to device")?;
        
        let mut factors1 = vec![0u64; count];
        let mut factors2 = vec![0u64; count];
        let mut success = vec![0u8; count];
        
        let d_factors1 = self.device.htod_copy(&factors1)
            .context("Failed to allocate factors1 on device")?;
        let d_factors2 = self.device.htod_copy(&factors2)
            .context("Failed to allocate factors2 on device")?;
        let d_success = self.device.htod_copy(&success)
            .context("Failed to allocate success flags on device")?;
        
        // Configure kernel launch
        let block_size = 256;
        let grid_size = (count + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        // Launch kernel
        let kernel = self.module.get_func("factorize_semiprimes_kernel")
            .context("Failed to get kernel function")?;
        
        unsafe {
            kernel.launch(
                config,
                (&d_numbers, &d_factors1, &d_factors2, &d_success, count as i32),
            ).context("Failed to launch kernel")?;
        }
        
        // Copy results back
        self.device.dtoh_sync_copy_into(&d_factors1, &mut factors1)
            .context("Failed to copy factors1 from device")?;
        self.device.dtoh_sync_copy_into(&d_factors2, &mut factors2)
            .context("Failed to copy factors2 from device")?;
        self.device.dtoh_sync_copy_into(&d_success, &mut success)
            .context("Failed to copy success flags from device")?;
        
        let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        let time_per_number = total_time_ms / count as f64;
        
        // Build results
        let mut results = Vec::with_capacity(count);
        for i in 0..count {
            if success[i] == 1 {
                results.push(SemiprimeResult::new(
                    numbers[i],
                    factors1[i],
                    factors2[i],
                    time_per_number,
                    "cuda_batch".to_string(),
                ));
            } else {
                // For failed factorizations, still return a result indicating failure
                results.push(SemiprimeResult {
                    number: numbers[i],
                    factor1: 0,
                    factor2: 0,
                    time_ms: time_per_number,
                    algorithm: "cuda_batch_failed".to_string(),
                    verified: false,
                });
            }
        }
        
        Ok(results)
    }
}

/// Public API for CUDA semiprime factorization
pub fn factorize_semiprime_cuda(n: u64) -> Result<SemiprimeResult> {
    let cuda = SemiprimeCuda::new()?;
    let results = cuda.factorize_batch(&[n])?;
    
    results.into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No result returned from CUDA"))
}

/// Batch factorization on GPU
pub fn factorize_semiprime_cuda_batch(numbers: &[u64]) -> Result<Vec<Result<SemiprimeResult, String>>> {
    let cuda = SemiprimeCuda::new()?;
    let results = cuda.factorize_batch(numbers)?;
    
    Ok(results.into_iter()
        .map(|r| {
            if r.verified {
                Ok(r)
            } else {
                Err(format!("{} could not be factored as semiprime", r.number))
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_semiprime_single() {
        // Test with a known semiprime
        let result = factorize_semiprime_cuda(15).expect("CUDA factorization should work");
        assert!(result.verified);
        assert_eq!(result.factor1 * result.factor2, 15);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_semiprime_batch() {
        let numbers = vec![15, 77, 221, 10403];
        let results = factorize_semiprime_cuda_batch(&numbers)
            .expect("Batch factorization should work");
        
        assert_eq!(results.len(), 4);
        for (i, result) in results.iter().enumerate() {
            assert!(result.is_ok(), "Number {} should factor successfully", numbers[i]);
        }
    }
}