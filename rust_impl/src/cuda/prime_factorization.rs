/// CUDA implementation of prime factorization
/// Uses parallel trial division for GPU acceleration

use super::{CudaDevice, CudaBuffer, CudaResult, CudaError};
use cudarc::driver::{CudaFunction, LaunchAsync, LaunchConfig};

const PRIME_KERNEL: &str = r#"
extern "C" __global__ void prime_factorize_kernel(
    unsigned long long n,
    unsigned long long* factors,
    int* factor_count,
    unsigned long long start,
    unsigned long long end
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long candidate = start + idx * 2; // Only check odd numbers
    
    if (candidate > end || candidate > n) return;
    
    // Check if candidate divides n
    if (n % candidate == 0) {
        // Check if candidate is prime
        bool is_prime = true;
        for (unsigned long long i = 3; i * i <= candidate; i += 2) {
            if (candidate % i == 0) {
                is_prime = false;
                break;
            }
        }
        
        if (is_prime) {
            // Atomically add to factors array
            int pos = atomicAdd(factor_count, 1);
            if (pos < 100) { // Max 100 factors
                factors[pos] = candidate;
            }
        }
    }
}

extern "C" __global__ void check_small_factors_kernel(
    unsigned long long n,
    unsigned long long* factors,
    int* factor_count
) {
    // Check factor of 2
    unsigned long long num = n;
    while (num % 2 == 0) {
        int pos = atomicAdd(factor_count, 1);
        if (pos < 100) factors[pos] = 2;
        num /= 2;
    }
}
"#;

/// CUDA prime factorization implementation
pub struct CudaPrimeFactorizer {
    device: CudaDevice,
    factorize_fn: CudaFunction,
    check_small_fn: CudaFunction,
}

impl CudaPrimeFactorizer {
    /// Create new CUDA prime factorizer
    pub fn new(device: &CudaDevice) -> CudaResult<Self> {
        // Compile kernels
        let module = device.compile_ptx(PRIME_KERNEL)?;
        
        let factorize_fn = module
            .get_function("prime_factorize_kernel")
            .map_err(|e| CudaError::KernelError(format!("Failed to get factorize kernel: {}", e)))?;
        
        let check_small_fn = module
            .get_function("check_small_factors_kernel")
            .map_err(|e| CudaError::KernelError(format!("Failed to get check_small kernel: {}", e)))?;
        
        Ok(Self {
            device: device.clone(),
            factorize_fn,
            check_small_fn,
        })
    }
    
    /// Factorize a number using CUDA
    pub fn factorize(&self, n: u64) -> CudaResult<Vec<u64>> {
        if n <= 1 {
            return Ok(vec![]);
        }
        
        // Allocate device memory
        let mut d_factors = CudaBuffer::<u64>::zeros(&self.device, 100)?;
        let mut d_factor_count = CudaBuffer::<i32>::zeros(&self.device, 1)?;
        
        // First check small factors (2)
        unsafe {
            self.check_small_fn.launch(
                LaunchConfig::for_num_elems(1),
                (&n, &d_factors.as_device_ptr(), &d_factor_count.as_device_ptr()),
            ).map_err(|e| CudaError::KernelError(format!("Check small factors failed: {}", e)))?;
        }
        
        // Now check odd factors in parallel
        let sqrt_n = (n as f64).sqrt() as u64;
        let block_size = 256;
        let num_candidates = (sqrt_n - 3) / 2 + 1; // Number of odd candidates from 3 to sqrt(n)
        let grid_size = (num_candidates + block_size as u64 - 1) / block_size as u64;
        
        unsafe {
            self.factorize_fn.launch(
                LaunchConfig {
                    grid_dim: (grid_size as u32, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &n,
                    &d_factors.as_device_ptr(),
                    &d_factor_count.as_device_ptr(),
                    &3u64, // start at 3
                    &sqrt_n,
                ),
            ).map_err(|e| CudaError::KernelError(format!("Factorization kernel failed: {}", e)))?;
        }
        
        // Copy results back
        self.device.synchronize()?;
        let factor_count = d_factor_count.copy_to_host()?[0] as usize;
        let factors_host = d_factors.copy_to_host()?;
        
        // Get valid factors
        let mut factors: Vec<u64> = factors_host[..factor_count.min(100)].to_vec();
        
        // If we found factors, check if we need to add the complementary factor
        if !factors.is_empty() {
            let product: u64 = factors.iter().product();
            if product < n {
                let remaining = n / product;
                if remaining > 1 {
                    factors.push(remaining);
                }
            }
        } else if n > 1 {
            // n is prime
            factors.push(n);
        }
        
        // Sort factors
        factors.sort();
        
        Ok(factors)
    }
}

/// Public API function for CUDA prime factorization
pub fn factorize_cuda(n: u64) -> CudaResult<Vec<u64>> {
    let device = CudaDevice::new(0)?;
    let factorizer = CudaPrimeFactorizer::new(&device)?;
    factorizer.factorize(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cuda_prime_factorization() {
        if !crate::cuda::cuda_available() {
            println!("Skipping CUDA test - no GPU available");
            return;
        }
        
        // Test small prime
        let factors = factorize_cuda(17).unwrap();
        assert_eq!(factors, vec![17]);
        
        // Test composite
        let factors = factorize_cuda(24).unwrap();
        assert_eq!(factors, vec![2, 2, 2, 3]);
        
        // Test the main benchmark number
        let factors = factorize_cuda(100822548703).unwrap();
        assert_eq!(factors.len(), 2);
        let product: u64 = factors.iter().product();
        assert_eq!(product, 100822548703);
    }
}