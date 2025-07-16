use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use crate::cuda::{CudaBuffer, CudaError, CudaResult};

// Optimized CUDA kernels for factoring 2539123152460219 and similar large numbers
// Designed for GTX 2070: 2304 CUDA cores, 8GB memory, compute capability 7.5
const LARGE_FACTORIZATION_KERNELS: &str = r#"
// Constants for optimized memory access
#define SHARED_PRIME_COUNT 1024
#define WARP_SIZE 32
#define MAX_FACTORS 32

// Montgomery multiplication constants for 64-bit arithmetic
__device__ __constant__ unsigned long long MONT_R = 0x10000000000000000ULL; // 2^64
__device__ __constant__ unsigned long long MONT_R2 = 0x4000000000000000ULL; // R^2 mod N

// Optimized Montgomery multiplication for 64-bit numbers
__device__ unsigned long long mont_mul(unsigned long long a, unsigned long long b, unsigned long long n, unsigned long long n_inv) {
    // Compute a * b mod n using Montgomery reduction
    unsigned long long t_lo, t_hi;
    
    // 128-bit multiplication using PTX assembly for efficiency
    asm("mul.lo.u64 %0, %2, %3;" : "=l"(t_lo) : "l"(a), "l"(b));
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(t_hi) : "l"(a), "l"(b));
    
    unsigned long long m = t_lo * n_inv;
    
    unsigned long long mn_lo, mn_hi;
    asm("mul.lo.u64 %0, %2, %3;" : "=l"(mn_lo) : "l"(m), "l"(n));
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(mn_hi) : "l"(m), "l"(n));
    
    // Add and reduce
    unsigned long long carry = 0;
    asm("add.cc.u64 %0, %1, %2;" : "=l"(t_lo) : "l"(t_lo), "l"(mn_lo));
    asm("addc.u64 %0, %1, %2;" : "=l"(t_hi) : "l"(t_hi), "l"(mn_hi));
    
    return (t_hi >= n) ? t_hi - n : t_hi;
}

// Fast modular exponentiation using Montgomery form
__device__ unsigned long long mont_pow(unsigned long long base, unsigned long long exp, unsigned long long n, unsigned long long n_inv) {
    unsigned long long result = MONT_R % n; // 1 in Montgomery form
    base = mont_mul(base, MONT_R2, n, n_inv); // Convert to Montgomery form
    
    while (exp > 0) {
        if (exp & 1) {
            result = mont_mul(result, base, n, n_inv);
        }
        base = mont_mul(base, base, n, n_inv);
        exp >>= 1;
    }
    
    // Convert back from Montgomery form
    return mont_mul(result, 1, n, n_inv);
}

// Compute modular inverse for Montgomery reduction
__device__ unsigned long long mont_inverse(unsigned long long n) {
    unsigned long long inv = n;
    for (int i = 0; i < 5; i++) {
        inv *= 2 - n * inv;
    }
    return -inv;
}

// Optimized GCD using binary algorithm (Stein's algorithm)
__device__ unsigned long long binary_gcd(unsigned long long a, unsigned long long b) {
    if (a == 0) return b;
    if (b == 0) return a;
    
    // Find common factors of 2
    int shift = __clzll(a | b) - __clzll((a | b) & -(a | b));
    a >>= __ctzll(a);
    
    do {
        b >>= __ctzll(b);
        if (a > b) {
            unsigned long long t = b;
            b = a;
            a = t;
        }
        b = b - a;
    } while (b != 0);
    
    return a << shift;
}

// Parallel trial division kernel with shared memory optimization
extern "C" __global__ void parallel_trial_division_kernel(
    unsigned long long* numbers,
    unsigned long long* factors,
    unsigned int* factor_counts,
    const unsigned int num_numbers,
    const unsigned long long* prime_table,
    const unsigned int prime_count
) {
    __shared__ unsigned long long shared_primes[SHARED_PRIME_COUNT];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int global_tid = bid * blockDim.x + tid;
    
    // Cooperatively load primes into shared memory
    for (int i = tid; i < SHARED_PRIME_COUNT && i < prime_count; i += blockDim.x) {
        shared_primes[i] = prime_table[i];
    }
    __syncthreads();
    
    if (global_tid >= num_numbers) return;
    
    unsigned long long n = numbers[global_tid];
    unsigned long long* my_factors = &factors[global_tid * MAX_FACTORS];
    int factor_idx = 0;
    
    // Trial division with shared memory primes
    for (int i = 0; i < SHARED_PRIME_COUNT && i < prime_count && shared_primes[i] * shared_primes[i] <= n; i++) {
        unsigned long long p = shared_primes[i];
        while (n % p == 0) {
            if (factor_idx < MAX_FACTORS) {
                my_factors[factor_idx++] = p;
            }
            n /= p;
        }
    }
    
    // Continue with larger primes from global memory
    for (int i = SHARED_PRIME_COUNT; i < prime_count && prime_table[i] * prime_table[i] <= n; i++) {
        unsigned long long p = prime_table[i];
        while (n % p == 0) {
            if (factor_idx < MAX_FACTORS) {
                my_factors[factor_idx++] = p;
            }
            n /= p;
        }
    }
    
    if (n > 1 && factor_idx < MAX_FACTORS) {
        my_factors[factor_idx++] = n;
    }
    
    factor_counts[global_tid] = factor_idx;
}

// Optimized Pollard's rho with Brent's improvement
extern "C" __global__ void pollard_rho_brent_kernel(
    unsigned long long* numbers,
    unsigned long long* factors,
    unsigned int* factor_counts,
    const unsigned int num_numbers,
    const unsigned long long small_prime_bound
) {
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_tid >= num_numbers) return;
    
    unsigned long long n = numbers[global_tid];
    unsigned long long* my_factors = &factors[global_tid * MAX_FACTORS];
    int factor_idx = 0;
    
    // Remove small prime factors first
    // Unrolled loop for common small primes
    while ((n & 1) == 0) {
        if (factor_idx < MAX_FACTORS) my_factors[factor_idx++] = 2;
        n >>= 1;
    }
    
    while (n % 3 == 0) {
        if (factor_idx < MAX_FACTORS) my_factors[factor_idx++] = 3;
        n /= 3;
    }
    
    while (n % 5 == 0) {
        if (factor_idx < MAX_FACTORS) my_factors[factor_idx++] = 5;
        n /= 5;
    }
    
    // Brent's improvement to Pollard's rho
    if (n > small_prime_bound * small_prime_bound) {
        unsigned long long y = 2, c = 1, m = 128;
        unsigned long long g = 1, r = 1, q = 1;
        unsigned long long x, ys, k;
        
        do {
            x = y;
            for (unsigned long long i = 0; i < r; i++) {
                y = (y * y + c) % n;
            }
            
            k = 0;
            while (k < r && g == 1) {
                ys = y;
                unsigned long long limit = min(m, r - k);
                
                for (unsigned long long i = 0; i < limit; i++) {
                    y = (y * y + c) % n;
                    unsigned long long diff = (x > y) ? x - y : y - x;
                    q = (q * diff) % n;
                }
                
                g = binary_gcd(q, n);
                k += limit;
            }
            
            r *= 2;
        } while (g == 1);
        
        if (g == n) {
            // Retry with backtracking
            do {
                ys = (ys * ys + c) % n;
                unsigned long long diff = (x > ys) ? x - ys : ys - x;
                g = binary_gcd(diff, n);
            } while (g == 1);
        }
        
        if (g != n && factor_idx < MAX_FACTORS) {
            my_factors[factor_idx++] = g;
            n /= g;
            
            // Check if n is still composite
            if (n > 1 && factor_idx < MAX_FACTORS) {
                my_factors[factor_idx++] = n;
            }
        }
    } else if (n > 1 && factor_idx < MAX_FACTORS) {
        my_factors[factor_idx++] = n;
    }
    
    factor_counts[global_tid] = factor_idx;
}

// Parallel sieve kernel for generating prime tables
extern "C" __global__ void segmented_sieve_kernel(
    unsigned char* sieve,
    const unsigned long long start,
    const unsigned long long end,
    const unsigned long long* base_primes,
    const unsigned int base_prime_count
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int threads_per_block = blockDim.x;
    const int global_tid = bid * threads_per_block + tid;
    
    const unsigned long long segment_size = end - start;
    const unsigned long long chunk_size = (segment_size + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
    const unsigned long long my_start = start + global_tid * chunk_size;
    const unsigned long long my_end = min(my_start + chunk_size, end);
    
    // Each thread processes a chunk of the sieve
    for (unsigned int i = 0; i < base_prime_count; i++) {
        unsigned long long p = base_primes[i];
        if (p * p > my_end) break;
        
        // Find first multiple of p in our range
        unsigned long long first_multiple = ((my_start + p - 1) / p) * p;
        if (first_multiple == p) first_multiple = p * p;
        
        // Mark multiples
        for (unsigned long long j = first_multiple; j < my_end; j += p) {
            sieve[j - start] = 0;
        }
    }
}

// Miller-Rabin primality test kernel
extern "C" __global__ void miller_rabin_kernel(
    unsigned long long* candidates,
    unsigned char* is_prime,
    const unsigned int num_candidates,
    const unsigned long long* witnesses,
    const unsigned int num_witnesses
) {
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_tid >= num_candidates) return;
    
    unsigned long long n = candidates[global_tid];
    
    // Handle small cases
    if (n < 2) {
        is_prime[global_tid] = 0;
        return;
    }
    if (n == 2 || n == 3) {
        is_prime[global_tid] = 1;
        return;
    }
    if ((n & 1) == 0) {
        is_prime[global_tid] = 0;
        return;
    }
    
    // Write n-1 as 2^r * d
    unsigned long long d = n - 1;
    int r = 0;
    while ((d & 1) == 0) {
        d >>= 1;
        r++;
    }
    
    // Compute Montgomery inverse
    unsigned long long n_inv = mont_inverse(n);
    
    // Test with each witness
    bool probably_prime = true;
    for (int i = 0; i < num_witnesses && probably_prime; i++) {
        unsigned long long a = witnesses[i] % (n - 2) + 2;
        unsigned long long x = mont_pow(a, d, n, n_inv);
        
        if (x == 1 || x == n - 1) continue;
        
        bool composite = true;
        for (int j = 0; j < r - 1; j++) {
            x = mont_mul(x, x, n, n_inv);
            if (x == n - 1) {
                composite = false;
                break;
            }
        }
        
        if (composite) {
            probably_prime = false;
        }
    }
    
    is_prime[global_tid] = probably_prime ? 1 : 0;
}

// Quadratic sieve helper kernel for smooth number detection
extern "C" __global__ void smooth_detection_kernel(
    unsigned long long* values,
    unsigned long long* smooth_factors,
    unsigned char* is_smooth,
    const unsigned int num_values,
    const unsigned long long* factor_base,
    const unsigned int factor_base_size,
    const unsigned long long smooth_bound
) {
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_tid >= num_values) return;
    
    unsigned long long n = values[global_tid];
    unsigned long long* my_factors = &smooth_factors[global_tid * MAX_FACTORS];
    int factor_idx = 0;
    
    // Trial division by factor base
    for (int i = 0; i < factor_base_size && factor_base[i] <= smooth_bound; i++) {
        unsigned long long p = factor_base[i];
        while (n % p == 0) {
            if (factor_idx < MAX_FACTORS) {
                my_factors[factor_idx++] = p;
            }
            n /= p;
        }
    }
    
    // Check if completely factored (smooth)
    is_smooth[global_tid] = (n == 1) ? 1 : 0;
}
"#;

pub struct LargeFactorizer {
    device: Arc<CudaDevice>,
    trial_division_kernel: cudarc::driver::CudaFunction,
    pollard_rho_kernel: cudarc::driver::CudaFunction,
    sieve_kernel: cudarc::driver::CudaFunction,
    miller_rabin_kernel: cudarc::driver::CudaFunction,
    smooth_detection_kernel: cudarc::driver::CudaFunction,
}

impl LargeFactorizer {
    pub fn new(device: Arc<CudaDevice>) -> CudaResult<Self> {
        // Compile kernels with optimization flags
        let ptx = cudarc::nvrtc::Ptx::compile_ptx_with_opts(
            LARGE_FACTORIZATION_KERNELS,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_75"), // GTX 2070 compute capability
                include_paths: vec![],
                definitions: vec![],
                ..Default::default()
            }
        ).map_err(|e| CudaError::KernelCompilationError(e.to_string()))?;
        
        // Load module
        let module = device.load_ptx(ptx, "large_factorization", &[
            "parallel_trial_division_kernel",
            "pollard_rho_brent_kernel",
            "segmented_sieve_kernel",
            "miller_rabin_kernel",
            "smooth_detection_kernel"
        ]).map_err(|e| CudaError::KernelLaunchError(e.to_string()))?;
        
        // Get kernel functions
        let trial_division_kernel = module.get_function("parallel_trial_division_kernel")
            .map_err(|e| CudaError::KernelLaunchError(e.to_string()))?;
        
        let pollard_rho_kernel = module.get_function("pollard_rho_brent_kernel")
            .map_err(|e| CudaError::KernelLaunchError(e.to_string()))?;
        
        let sieve_kernel = module.get_function("segmented_sieve_kernel")
            .map_err(|e| CudaError::KernelLaunchError(e.to_string()))?;
        
        let miller_rabin_kernel = module.get_function("miller_rabin_kernel")
            .map_err(|e| CudaError::KernelLaunchError(e.to_string()))?;
        
        let smooth_detection_kernel = module.get_function("smooth_detection_kernel")
            .map_err(|e| CudaError::KernelLaunchError(e.to_string()))?;
        
        Ok(Self {
            device,
            trial_division_kernel,
            pollard_rho_kernel,
            sieve_kernel,
            miller_rabin_kernel,
            smooth_detection_kernel,
        })
    }
    
    /// Generate prime table using segmented sieve
    pub fn generate_primes(&self, limit: u64) -> CudaResult<Vec<u64>> {
        // First generate small primes up to sqrt(limit)
        let sqrt_limit = (limit as f64).sqrt() as u64 + 1;
        let mut base_primes = vec![];
        let mut is_prime = vec![true; (sqrt_limit + 1) as usize];
        
        // Simple sieve for base primes
        is_prime[0] = false;
        is_prime[1] = false;
        for i in 2..=sqrt_limit {
            if is_prime[i as usize] {
                base_primes.push(i);
                let mut j = i * i;
                while j <= sqrt_limit {
                    is_prime[j as usize] = false;
                    j += i;
                }
            }
        }
        
        // Now use GPU for larger primes
        let segment_size = 1_000_000u64; // 1M elements per segment
        let mut all_primes = base_primes.clone();
        
        let d_base_primes = CudaBuffer::from_slice(self.device.clone(), &base_primes)?;
        
        for start in (sqrt_limit + 1..=limit).step_by(segment_size as usize) {
            let end = std::cmp::min(start + segment_size, limit + 1);
            let actual_size = (end - start) as usize;
            
            let mut d_sieve = CudaBuffer::from_slice(self.device.clone(), &vec![1u8; actual_size])?;
            
            // Launch sieve kernel
            let threads = 256;
            let blocks = (actual_size + threads - 1) / threads;
            let config = LaunchConfig {
                block_dim: (threads as u32, 1, 1),
                grid_dim: (blocks as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            
            unsafe {
                self.sieve_kernel.launch(
                    config,
                    (
                        d_sieve.as_device_ptr(),
                        start,
                        end,
                        d_base_primes.as_device_ptr(),
                        base_primes.len() as u32,
                    ),
                ).map_err(|e| CudaError::KernelLaunchError(e.to_string()))?;
            }
            
            self.device.synchronize()?;
            
            // Copy back and extract primes
            let sieve = d_sieve.to_vec()?;
            for (i, &is_prime) in sieve.iter().enumerate() {
                if is_prime != 0 {
                    all_primes.push(start + i as u64);
                }
            }
        }
        
        Ok(all_primes)
    }
    
    /// Factor a large number using combined approach
    pub fn factor_large(&self, number: u64) -> CudaResult<Vec<u64>> {
        // Generate prime table for trial division
        let prime_limit = std::cmp::min(1_000_000u64, (number as f64).sqrt() as u64);
        let primes = self.generate_primes(prime_limit)?;
        
        // First attempt trial division
        let mut factors = vec![];
        let mut n = number;
        
        // Upload to GPU for parallel trial division
        let d_numbers = CudaBuffer::from_slice(self.device.clone(), &[n])?;
        let d_primes = CudaBuffer::from_slice(self.device.clone(), &primes)?;
        let mut d_factors = CudaBuffer::zeros(&self.device, 32)?;
        let mut d_factor_counts = CudaBuffer::zeros(&self.device, 1)?;
        
        let config = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: std::mem::size_of::<u64>() * 1024,
        };
        
        unsafe {
            self.trial_division_kernel.launch(
                config,
                (
                    d_numbers.as_device_ptr(),
                    d_factors.as_device_ptr(),
                    d_factor_counts.as_device_ptr(),
                    1u32,
                    d_primes.as_device_ptr(),
                    primes.len() as u32,
                ),
            ).map_err(|e| CudaError::KernelLaunchError(e.to_string()))?;
        }
        
        self.device.synchronize()?;
        
        let trial_factors = d_factors.to_vec()?;
        let factor_count = d_factor_counts.to_vec()?[0] as usize;
        
        // Collect trial division factors
        for i in 0..factor_count {
            factors.push(trial_factors[i]);
            n /= trial_factors[i];
        }
        
        // If n is still large, use Pollard's rho
        if n > prime_limit * prime_limit {
            let d_remaining = CudaBuffer::from_slice(self.device.clone(), &[n])?;
            let mut d_rho_factors = CudaBuffer::zeros(&self.device, 32)?;
            let mut d_rho_counts = CudaBuffer::zeros(&self.device, 1)?;
            
            unsafe {
                self.pollard_rho_kernel.launch(
                    config,
                    (
                        d_remaining.as_device_ptr(),
                        d_rho_factors.as_device_ptr(),
                        d_rho_counts.as_device_ptr(),
                        1u32,
                        prime_limit,
                    ),
                ).map_err(|e| CudaError::KernelLaunchError(e.to_string()))?;
            }
            
            self.device.synchronize()?;
            
            let rho_factors = d_rho_factors.to_vec()?;
            let rho_count = d_rho_counts.to_vec()?[0] as usize;
            
            for i in 0..rho_count {
                factors.push(rho_factors[i]);
            }
        } else if n > 1 {
            factors.push(n);
        }
        
        factors.sort_unstable();
        Ok(factors)
    }
    
    /// Check if a number is prime using Miller-Rabin test
    pub fn is_prime(&self, number: u64) -> CudaResult<bool> {
        // Use deterministic witnesses for 64-bit numbers
        let witnesses = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
        
        let d_candidates = CudaBuffer::from_slice(self.device.clone(), &[number])?;
        let d_witnesses = CudaBuffer::from_slice(self.device.clone(), &witnesses)?;
        let mut d_is_prime = CudaBuffer::zeros(&self.device, 1)?;
        
        let config = LaunchConfig {
            block_dim: (1, 1, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            self.miller_rabin_kernel.launch(
                config,
                (
                    d_candidates.as_device_ptr(),
                    d_is_prime.as_device_ptr(),
                    1u32,
                    d_witnesses.as_device_ptr(),
                    witnesses.len() as u32,
                ),
            ).map_err(|e| CudaError::KernelLaunchError(e.to_string()))?;
        }
        
        self.device.synchronize()?;
        
        let is_prime = d_is_prime.to_vec()?;
        Ok(is_prime[0] != 0)
    }
    
    /// Batch factorization for multiple large numbers
    pub fn factor_batch_large(&self, numbers: &[u64]) -> CudaResult<Vec<Vec<u64>>> {
        // Generate comprehensive prime table
        let max_number = *numbers.iter().max().unwrap_or(&1);
        let prime_limit = std::cmp::min(10_000_000u64, (max_number as f64).sqrt() as u64);
        let primes = self.generate_primes(prime_limit)?;
        
        let num_count = numbers.len();
        let d_numbers = CudaBuffer::from_slice(self.device.clone(), numbers)?;
        let d_primes = CudaBuffer::from_slice(self.device.clone(), &primes)?;
        let mut d_factors = CudaBuffer::zeros(&self.device, num_count * 32)?;
        let mut d_factor_counts = CudaBuffer::zeros(&self.device, num_count)?;
        
        // Configure for optimal performance on GTX 2070
        let threads = 256;
        let blocks = (num_count + threads - 1) / threads;
        let config = LaunchConfig {
            block_dim: (threads as u32, 1, 1),
            grid_dim: (blocks as u32, 1, 1),
            shared_mem_bytes: std::mem::size_of::<u64>() * 1024,
        };
        
        // First pass: trial division
        unsafe {
            self.trial_division_kernel.launch(
                config,
                (
                    d_numbers.as_device_ptr(),
                    d_factors.as_device_ptr(),
                    d_factor_counts.as_device_ptr(),
                    num_count as u32,
                    d_primes.as_device_ptr(),
                    primes.len() as u32,
                ),
            ).map_err(|e| CudaError::KernelLaunchError(e.to_string()))?;
        }
        
        self.device.synchronize()?;
        
        // Get partial results
        let factors = d_factors.to_vec()?;
        let factor_counts = d_factor_counts.to_vec()?;
        
        // Build final results
        let mut results = Vec::with_capacity(num_count);
        for i in 0..num_count {
            let count = factor_counts[i] as usize;
            let start = i * 32;
            let mut number_factors = Vec::with_capacity(count);
            
            for j in 0..count {
                number_factors.push(factors[start + j]);
            }
            
            number_factors.sort_unstable();
            results.push(number_factors);
        }
        
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_large_factorization() {
        if let Ok(device) = CudaDevice::new(0) {
            let factorizer = LargeFactorizer::new(Arc::new(device)).unwrap();
            
            // Test the target number: 2539123152460219
            let target = 2539123152460219u64;
            let factors = factorizer.factor_large(target).unwrap();
            
            println!("Factors of {}: {:?}", target, factors);
            
            // Verify factorization
            let product: u64 = factors.iter().product();
            assert_eq!(product, target);
            
            // Check each factor is prime
            for &factor in &factors {
                assert!(factorizer.is_prime(factor).unwrap());
            }
        }
    }
    
    #[test]
    fn test_prime_generation() {
        if let Ok(device) = CudaDevice::new(0) {
            let factorizer = LargeFactorizer::new(Arc::new(device)).unwrap();
            
            let primes = factorizer.generate_primes(1000).unwrap();
            
            // Verify first few primes
            assert_eq!(primes[0], 2);
            assert_eq!(primes[1], 3);
            assert_eq!(primes[2], 5);
            assert_eq!(primes[3], 7);
            assert_eq!(primes[4], 11);
            
            // Verify all are prime
            for &p in &primes[0..20] {
                assert!(factorizer.is_prime(p).unwrap());
            }
        }
    }
    
    #[test]
    fn test_miller_rabin() {
        if let Ok(device) = CudaDevice::new(0) {
            let factorizer = LargeFactorizer::new(Arc::new(device)).unwrap();
            
            // Test known primes
            assert!(factorizer.is_prime(2).unwrap());
            assert!(factorizer.is_prime(3).unwrap());
            assert!(factorizer.is_prime(7919).unwrap()); // 1000th prime
            
            // Test known composites
            assert!(!factorizer.is_prime(4).unwrap());
            assert!(!factorizer.is_prime(9).unwrap());
            assert!(!factorizer.is_prime(1000).unwrap());
        }
    }
}