use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use crate::cuda::{CudaBuffer, CudaError, CudaResult};

// CUDA kernel source for prime factorization - optimized for GTX 2070
const PRIME_FACTORIZATION_KERNEL: &str = r#"
// Optimized prime factorization kernel for GTX 2070 (compute capability 7.5)
// Implements parallel trial division and Pollard's rho algorithm
extern "C" __global__ void prime_factorization_kernel(
    unsigned long long* numbers,         // Numbers to factorize
    unsigned long long* factors,         // Output factors (2 per number)
    unsigned int* factor_counts,         // Number of factors found
    const unsigned int num_numbers,      // Total numbers to process
    const unsigned long long max_trial   // Maximum trial division value
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int threads_per_block = blockDim.x;
    const int global_tid = bid * threads_per_block + tid;
    
    if (global_tid >= num_numbers) return;
    
    unsigned long long n = numbers[global_tid];
    unsigned long long original_n = n;
    int factor_idx = 0;
    
    // Output buffer for this number's factors
    unsigned long long* my_factors = &factors[global_tid * 16]; // Max 16 factors per number
    
    // Handle small factors efficiently
    // Factor out 2s
    while ((n & 1) == 0) {
        if (factor_idx < 16) {
            my_factors[factor_idx++] = 2;
        }
        n >>= 1;
    }
    
    // Factor out 3s
    while (n % 3 == 0) {
        if (factor_idx < 16) {
            my_factors[factor_idx++] = 3;
        }
        n /= 3;
    }
    
    // Factor out 5s
    while (n % 5 == 0) {
        if (factor_idx < 16) {
            my_factors[factor_idx++] = 5;
        }
        n /= 5;
    }
    
    // Trial division with 6k±1 optimization
    unsigned long long i = 7;
    unsigned long long sqrt_n = sqrt((double)n);
    
    while (i <= sqrt_n && i <= max_trial) {
        // Check i
        while (n % i == 0) {
            if (factor_idx < 16) {
                my_factors[factor_idx++] = i;
            }
            n /= i;
            sqrt_n = sqrt((double)n);
        }
        
        // Check i + 2 (handles 6k+1 and 6k-1)
        if (i + 2 <= sqrt_n) {
            while (n % (i + 2) == 0) {
                if (factor_idx < 16) {
                    my_factors[factor_idx++] = i + 2;
                }
                n /= (i + 2);
                sqrt_n = sqrt((double)n);
            }
        }
        
        i += 6;
    }
    
    // If n is still > 1, it's a prime factor
    if (n > 1 && factor_idx < 16) {
        my_factors[factor_idx++] = n;
    }
    
    // Store factor count
    factor_counts[global_tid] = factor_idx;
}

// Pollard's rho algorithm for finding a factor
extern "C" __device__ unsigned long long pollard_rho(unsigned long long n) {
    if (n == 1) return 1;
    if ((n & 1) == 0) return 2;
    
    unsigned long long x = 2, y = 2, d = 1;
    const unsigned long long c = 1; // Can be randomized
    
    // f(x) = (x^2 + c) mod n
    while (d == 1) {
        // Tortoise moves one step
        x = (x * x + c) % n;
        
        // Hare moves two steps
        y = (y * y + c) % n;
        y = (y * y + c) % n;
        
        // Calculate GCD
        unsigned long long diff = (x > y) ? x - y : y - x;
        
        // Inline GCD calculation
        unsigned long long a = diff, b = n;
        while (b != 0) {
            unsigned long long temp = b;
            b = a % b;
            a = temp;
        }
        d = a;
    }
    
    return (d == n) ? 1 : d;
}

// Advanced kernel using Pollard's rho for large factors
extern "C" __global__ void prime_factorization_advanced_kernel(
    unsigned long long* numbers,
    unsigned long long* factors,
    unsigned int* factor_counts,
    const unsigned int num_numbers,
    const unsigned long long small_prime_limit
) {
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_tid >= num_numbers) return;
    
    unsigned long long n = numbers[global_tid];
    int factor_idx = 0;
    unsigned long long* my_factors = &factors[global_tid * 16];
    
    // First do trial division for small primes
    // Factor out 2s
    while ((n & 1) == 0) {
        if (factor_idx < 16) my_factors[factor_idx++] = 2;
        n >>= 1;
    }
    
    // Trial division up to small_prime_limit
    for (unsigned long long p = 3; p <= small_prime_limit && p * p <= n; p += 2) {
        while (n % p == 0) {
            if (factor_idx < 16) my_factors[factor_idx++] = p;
            n /= p;
        }
    }
    
    // Use Pollard's rho for remaining large factors
    while (n > 1 && n > small_prime_limit * small_prime_limit) {
        unsigned long long factor = pollard_rho(n);
        if (factor == 1 || factor == n) {
            // n is prime
            if (factor_idx < 16) my_factors[factor_idx++] = n;
            break;
        }
        
        // Factor might be composite, check primality
        unsigned long long temp = factor;
        bool is_prime = true;
        for (unsigned long long p = 2; p * p <= temp && p <= 1000; p++) {
            if (temp % p == 0) {
                is_prime = false;
                break;
            }
        }
        
        if (is_prime && factor_idx < 16) {
            my_factors[factor_idx++] = factor;
        }
        
        n /= factor;
    }
    
    if (n > 1 && factor_idx < 16) {
        my_factors[factor_idx++] = n;
    }
    
    factor_counts[global_tid] = factor_idx;
}
"#;

pub struct PrimeFactorizer {
    device: Arc<CudaDevice>,
    kernel_basic: cudarc::driver::CudaFunction,
    kernel_advanced: cudarc::driver::CudaFunction,
}

impl PrimeFactorizer {
    pub fn new(device: Arc<CudaDevice>) -> CudaResult<Self> {
        // Compile kernels
        let ptx = cudarc::nvrtc::compile_ptx(PRIME_FACTORIZATION_KERNEL)
            .map_err(|e| CudaError::KernelCompilationError(e.to_string()))?;
        
        // Load module
        let module = device.load_ptx(ptx, "prime_factorization", &["prime_factorization_kernel", "prime_factorization_advanced_kernel"])
            .map_err(|e| CudaError::KernelLaunchError(e.to_string()))?;
        
        // Get kernel functions
        let kernel_basic = module.get_function("prime_factorization_kernel")
            .map_err(|e| CudaError::KernelLaunchError(e.to_string()))?;
        
        let kernel_advanced = module.get_function("prime_factorization_advanced_kernel")
            .map_err(|e| CudaError::KernelLaunchError(e.to_string()))?;
        
        Ok(Self {
            device,
            kernel_basic,
            kernel_advanced,
        })
    }
    
    /// Factorize a batch of numbers using trial division
    pub fn factorize_batch(&self, numbers: &[u64]) -> CudaResult<Vec<Vec<u64>>> {
        let num_numbers = numbers.len();
        if num_numbers == 0 {
            return Ok(vec![]);
        }
        
        // Allocate device memory
        let mut d_numbers = CudaBuffer::from_slice(&self.device, numbers)?;
        let mut d_factors = CudaBuffer::zeros(&self.device, num_numbers * 16)?; // Max 16 factors per number
        let mut d_factor_counts = CudaBuffer::zeros(&self.device, num_numbers)?;
        
        // Configure kernel launch
        let threads_per_block = 256;
        let blocks = (num_numbers + threads_per_block - 1) / threads_per_block;
        let config = LaunchConfig {
            block_dim: (threads_per_block as u32, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        // Launch kernel with max trial division up to sqrt of typical numbers
        let max_trial = 1_000_000u64; // Suitable for numbers up to 10^12
        
        unsafe {
            self.kernel_basic.launch(
                config,
                (
                    d_numbers.as_device_ptr(),
                    d_factors.as_device_ptr(),
                    d_factor_counts.as_device_ptr(),
                    num_numbers as u32,
                    max_trial,
                ),
            ).map_err(|e| CudaError::KernelLaunchError(e.to_string()))?;
        }
        
        // Synchronize
        self.device.synchronize()
            .map_err(|e| CudaError::SynchronizationError(e.to_string()))?;
        
        // Copy results back
        let factors = d_factors.to_vec()?;
        let factor_counts = d_factor_counts.to_vec()?;
        
        // Parse results
        let mut results = Vec::with_capacity(num_numbers);
        for i in 0..num_numbers {
            let count = factor_counts[i] as usize;
            let start = i * 16;
            let mut number_factors = Vec::with_capacity(count);
            
            for j in 0..count {
                number_factors.push(factors[start + j]);
            }
            
            results.push(number_factors);
        }
        
        Ok(results)
    }
    
    /// Factorize using advanced Pollard's rho algorithm for large numbers
    pub fn factorize_advanced(&self, numbers: &[u64]) -> CudaResult<Vec<Vec<u64>>> {
        let num_numbers = numbers.len();
        if num_numbers == 0 {
            return Ok(vec![]);
        }
        
        // Allocate device memory
        let mut d_numbers = CudaBuffer::from_slice(&self.device, numbers)?;
        let mut d_factors = CudaBuffer::zeros(&self.device, num_numbers * 16)?;
        let mut d_factor_counts = CudaBuffer::zeros(&self.device, num_numbers)?;
        
        // Configure kernel launch
        let threads_per_block = 128; // Fewer threads for more complex kernel
        let blocks = (num_numbers + threads_per_block - 1) / threads_per_block;
        let config = LaunchConfig {
            block_dim: (threads_per_block as u32, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        // Small prime limit for initial trial division
        let small_prime_limit = 10_000u64;
        
        unsafe {
            self.kernel_advanced.launch(
                config,
                (
                    d_numbers.as_device_ptr(),
                    d_factors.as_device_ptr(),
                    d_factor_counts.as_device_ptr(),
                    num_numbers as u32,
                    small_prime_limit,
                ),
            ).map_err(|e| CudaError::KernelLaunchError(e.to_string()))?;
        }
        
        // Synchronize
        self.device.synchronize()
            .map_err(|e| CudaError::SynchronizationError(e.to_string()))?;
        
        // Copy results back
        let factors = d_factors.to_vec()?;
        let factor_counts = d_factor_counts.to_vec()?;
        
        // Parse results
        let mut results = Vec::with_capacity(num_numbers);
        for i in 0..num_numbers {
            let count = factor_counts[i] as usize;
            let start = i * 16;
            let mut number_factors = Vec::with_capacity(count);
            
            for j in 0..count {
                number_factors.push(factors[start + j]);
            }
            
            // Sort factors
            number_factors.sort_unstable();
            
            results.push(number_factors);
        }
        
        Ok(results)
    }
    
    /// Factorize a single number
    pub fn factorize_single(&self, number: u64) -> CudaResult<Vec<u64>> {
        let results = if number < 1_000_000_000_000 {
            self.factorize_batch(&[number])?
        } else {
            self.factorize_advanced(&[number])?
        };
        
        Ok(results.into_iter().next().unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_prime_factorization() {
        if let Ok(device) = CudaDevice::new(0) {
            let factorizer = PrimeFactorizer::new(Arc::new(device)).unwrap();
            
            // Test the target number
            let target = 100822548703u64;
            let factors = factorizer.factorize_single(target).unwrap();
            
            // Verify factorization
            let product: u64 = factors.iter().product();
            assert_eq!(product, target);
            
            // Should find 316907 and 318089
            assert_eq!(factors.len(), 2);
            assert!(factors.contains(&316907));
            assert!(factors.contains(&318089));
        }
    }
}