/// CUDA Composite Number Factorizer
/// Specialized GPU-accelerated factorization for composite numbers
/// Implements multiple algorithms optimized for different composite number types

use super::{CudaDevice, CudaBuffer, CudaResult, CudaError};
use cudarc::driver::{CudaFunction, LaunchAsync, LaunchConfig};
use std::collections::HashMap;

/// CUDA kernels for composite number factorization
const COMPOSITE_KERNELS: &str = r#"
#include <cuda_runtime.h>

// Fermat's factorization method for numbers close to perfect squares
extern "C" __global__ void fermat_factorization_kernel(
    unsigned long long n,
    unsigned long long* factors,
    int* found,
    unsigned long long start_a,
    unsigned long long end_a
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long a = start_a + idx;
    
    if (a > end_a || *found) return;
    
    unsigned long long a_squared = a * a;
    if (a_squared <= n) return;
    
    unsigned long long b_squared = a_squared - n;
    
    // Check if b_squared is a perfect square
    unsigned long long b = sqrtf(b_squared);
    if (b * b == b_squared) {
        // Found factorization: n = (a-b)(a+b)
        if (atomicCAS(found, 0, 1) == 0) {
            factors[0] = a - b;
            factors[1] = a + b;
        }
    }
}

// Pollard's rho algorithm for general composite numbers
extern "C" __global__ void pollard_rho_kernel(
    unsigned long long n,
    unsigned long long* factors,
    int* found,
    unsigned long long seed
) {
    if (*found) return;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread uses a different starting value
    unsigned long long x = seed + idx;
    unsigned long long y = x;
    unsigned long long d = 1;
    
    // Pollard's rho with different polynomial
    for (int i = 0; i < 10000 && d == 1; i++) {
        x = (x * x + 1) % n;
        y = (y * y + 1) % n;
        y = (y * y + 1) % n;
        
        unsigned long long diff = (x > y) ? x - y : y - x;
        d = gcd(diff, n);
    }
    
    if (d != 1 && d != n) {
        if (atomicCAS(found, 0, 1) == 0) {
            factors[0] = d;
            factors[1] = n / d;
        }
    }
}

// Quadratic sieve preprocessing kernel
extern "C" __global__ void quadratic_sieve_smooth_kernel(
    unsigned long long n,
    unsigned long long* smooth_numbers,
    int* smooth_count,
    unsigned long long* factor_base,
    int factor_base_size,
    unsigned long long start,
    unsigned long long range
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long x = start + idx;
    
    if (x >= start + range) return;
    
    // Compute Q(x) = x^2 - n
    unsigned long long qx = x * x - n;
    unsigned long long original_qx = qx;
    
    // Trial divide by factor base
    bool is_smooth = true;
    for (int i = 0; i < factor_base_size && qx > 1; i++) {
        while (qx % factor_base[i] == 0) {
            qx /= factor_base[i];
        }
    }
    
    if (qx == 1) {
        // Found a smooth number
        int pos = atomicAdd(smooth_count, 1);
        if (pos < 1000) {
            smooth_numbers[pos * 2] = x;
            smooth_numbers[pos * 2 + 1] = original_qx;
        }
    }
}

// Helper function: GCD
__device__ unsigned long long gcd(unsigned long long a, unsigned long long b) {
    while (b != 0) {
        unsigned long long temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Trial division for small factors
extern "C" __global__ void trial_division_kernel(
    unsigned long long n,
    unsigned long long* factors,
    int* factor_count,
    unsigned long long* primes,
    int prime_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= prime_count) return;
    
    unsigned long long prime = primes[idx];
    unsigned long long temp_n = n;
    
    while (temp_n % prime == 0) {
        int pos = atomicAdd(factor_count, 1);
        if (pos < 100) {
            factors[pos] = prime;
        }
        temp_n /= prime;
    }
}

// ECM (Elliptic Curve Method) point addition kernel
extern "C" __global__ void ecm_point_addition_kernel(
    unsigned long long n,
    unsigned long long* x_coords,
    unsigned long long* y_coords,
    unsigned long long* z_coords,
    int num_curves,
    unsigned long long scalar
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_curves) return;
    
    // Montgomery curve point multiplication
    unsigned long long x = x_coords[idx];
    unsigned long long z = z_coords[idx];
    unsigned long long a24 = 1; // Simplified for demo
    
    // Point doubling and addition operations
    for (int bit = 63; bit >= 0; bit--) {
        if ((scalar >> bit) & 1) {
            // Point addition logic
            unsigned long long new_x = (x * x - z * z) % n;
            unsigned long long new_z = (2 * x * z) % n;
            x = new_x;
            z = new_z;
        }
    }
    
    x_coords[idx] = x;
    z_coords[idx] = z;
}
"#;

/// Composite number type classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompositeType {
    Semiprime,           // Product of two primes
    PowerOfPrime,        // p^k for prime p
    HighlyComposite,     // Many small factors
    RSALike,            // Two large primes of similar size
    GeneralComposite,    // No special structure
}

/// CUDA Composite Number Factorizer
pub struct CudaCompositeFactorizer {
    device: CudaDevice,
    fermat_fn: CudaFunction,
    pollard_fn: CudaFunction,
    quadratic_fn: CudaFunction,
    trial_fn: CudaFunction,
    ecm_fn: CudaFunction,
    prime_cache: Vec<u64>,
}

impl CudaCompositeFactorizer {
    /// Create new CUDA composite factorizer
    pub fn new(device: &CudaDevice) -> CudaResult<Self> {
        let module = device.compile_ptx(COMPOSITE_KERNELS)?;
        
        let fermat_fn = module.get_function("fermat_factorization_kernel")
            .map_err(|e| CudaError::KernelError(format!("Fermat kernel: {}", e)))?;
        
        let pollard_fn = module.get_function("pollard_rho_kernel")
            .map_err(|e| CudaError::KernelError(format!("Pollard kernel: {}", e)))?;
        
        let quadratic_fn = module.get_function("quadratic_sieve_smooth_kernel")
            .map_err(|e| CudaError::KernelError(format!("Quadratic kernel: {}", e)))?;
        
        let trial_fn = module.get_function("trial_division_kernel")
            .map_err(|e| CudaError::KernelError(format!("Trial kernel: {}", e)))?;
        
        let ecm_fn = module.get_function("ecm_point_addition_kernel")
            .map_err(|e| CudaError::KernelError(format!("ECM kernel: {}", e)))?;
        
        // Generate prime cache for trial division
        let prime_cache = generate_primes(10000);
        
        Ok(Self {
            device: device.clone(),
            fermat_fn,
            pollard_fn,
            quadratic_fn,
            trial_fn,
            ecm_fn,
            prime_cache,
        })
    }
    
    /// Classify composite number type
    pub fn classify_composite(&self, n: u64) -> CompositeType {
        // Quick primality test
        if is_probable_prime(n) {
            return CompositeType::GeneralComposite;
        }
        
        // Check if it's a power of a prime
        for exp in 2..64 {
            let root = (n as f64).powf(1.0 / exp as f64) as u64;
            if root.pow(exp) == n && is_probable_prime(root) {
                return CompositeType::PowerOfPrime;
            }
        }
        
        // Check number of small prime factors
        let mut temp = n;
        let mut small_factors = 0;
        for &p in &self.prime_cache[..100] {
            while temp % p == 0 {
                small_factors += 1;
                temp /= p;
            }
        }
        
        if small_factors > 5 {
            return CompositeType::HighlyComposite;
        }
        
        // Check if it might be RSA-like (two large primes)
        let sqrt_n = (n as f64).sqrt() as u64;
        if temp > sqrt_n / 10 {
            return CompositeType::RSALike;
        }
        
        // Check if it's likely a semiprime
        if small_factors == 0 {
            return CompositeType::Semiprime;
        }
        
        CompositeType::GeneralComposite
    }
    
    /// Factorize using Fermat's method (best for numbers close to perfect squares)
    pub fn fermat_factorize(&self, n: u64) -> CudaResult<Option<(u64, u64)>> {
        let start_a = ((n as f64).sqrt() as u64) + 1;
        let range = (n as f64).sqrt() as u64 / 10; // Search range
        
        let mut d_factors = CudaBuffer::<u64>::zeros(&self.device, 2)?;
        let mut d_found = CudaBuffer::<i32>::zeros(&self.device, 1)?;
        
        let block_size = 256;
        let grid_size = (range + block_size - 1) / block_size;
        
        unsafe {
            self.fermat_fn.launch(
                LaunchConfig {
                    grid_dim: (grid_size as u32, 1, 1),
                    block_dim: (block_size as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &n,
                    &d_factors.as_device_ptr(),
                    &d_found.as_device_ptr(),
                    &start_a,
                    &(start_a + range),
                ),
            )?;
        }
        
        self.device.synchronize()?;
        
        let found = d_found.copy_to_host()?[0];
        if found == 1 {
            let factors = d_factors.copy_to_host()?;
            Ok(Some((factors[0], factors[1])))
        } else {
            Ok(None)
        }
    }
    
    /// Factorize using Pollard's rho algorithm
    pub fn pollard_rho_factorize(&self, n: u64) -> CudaResult<Option<(u64, u64)>> {
        let mut d_factors = CudaBuffer::<u64>::zeros(&self.device, 2)?;
        let mut d_found = CudaBuffer::<i32>::zeros(&self.device, 1)?;
        
        let block_size = 256;
        let grid_size = 64; // Multiple starting points
        
        unsafe {
            self.pollard_fn.launch(
                LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &n,
                    &d_factors.as_device_ptr(),
                    &d_found.as_device_ptr(),
                    &2u64, // seed
                ),
            )?;
        }
        
        self.device.synchronize()?;
        
        let found = d_found.copy_to_host()?[0];
        if found == 1 {
            let factors = d_factors.copy_to_host()?;
            Ok(Some((factors[0], factors[1])))
        } else {
            Ok(None)
        }
    }
    
    /// Main factorization method that chooses the best algorithm
    pub fn factorize(&self, n: u64) -> CudaResult<Vec<u64>> {
        if n <= 1 {
            return Ok(vec![]);
        }
        
        // Classify the composite number
        let composite_type = self.classify_composite(n);
        
        match composite_type {
            CompositeType::RSALike | CompositeType::Semiprime => {
                // Try Fermat's method first for numbers close to perfect squares
                if let Some((f1, f2)) = self.fermat_factorize(n)? {
                    return Ok(vec![f1, f2]);
                }
                
                // Fall back to Pollard's rho
                if let Some((f1, f2)) = self.pollard_rho_factorize(n)? {
                    return Ok(vec![f1, f2]);
                }
            }
            
            CompositeType::HighlyComposite => {
                // Use trial division for many small factors
                return self.trial_division_factorize(n);
            }
            
            CompositeType::PowerOfPrime => {
                // Find the prime base and exponent
                for exp in 2..64 {
                    let root = (n as f64).powf(1.0 / exp as f64) as u64;
                    if root.pow(exp) == n && is_probable_prime(root) {
                        return Ok(vec![root; exp as usize]);
                    }
                }
            }
            
            CompositeType::GeneralComposite => {
                // Try multiple methods
                if let Some((f1, f2)) = self.pollard_rho_factorize(n)? {
                    let mut factors = vec![];
                    factors.extend(self.factorize(f1)?);
                    factors.extend(self.factorize(f2)?);
                    return Ok(factors);
                }
            }
        }
        
        // If all else fails, return the number itself
        Ok(vec![n])
    }
    
    /// Trial division using GPU
    fn trial_division_factorize(&self, n: u64) -> CudaResult<Vec<u64>> {
        let prime_count = self.prime_cache.len().min(1000);
        let d_primes = CudaBuffer::from_slice(&self.device, &self.prime_cache[..prime_count])?;
        let mut d_factors = CudaBuffer::<u64>::zeros(&self.device, 100)?;
        let mut d_factor_count = CudaBuffer::<i32>::zeros(&self.device, 1)?;
        
        let block_size = 256;
        let grid_size = (prime_count + block_size - 1) / block_size;
        
        unsafe {
            self.trial_fn.launch(
                LaunchConfig {
                    grid_dim: (grid_size as u32, 1, 1),
                    block_dim: (block_size as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &n,
                    &d_factors.as_device_ptr(),
                    &d_factor_count.as_device_ptr(),
                    &d_primes.as_device_ptr(),
                    &(prime_count as i32),
                ),
            )?;
        }
        
        self.device.synchronize()?;
        
        let factor_count = d_factor_count.copy_to_host()?[0] as usize;
        let factors = d_factors.copy_to_host()?;
        
        Ok(factors[..factor_count.min(100)].to_vec())
    }
}

/// Generate first n primes using sieve
fn generate_primes(limit: usize) -> Vec<u64> {
    let mut sieve = vec![true; limit];
    sieve[0] = false;
    sieve[1] = false;
    
    for i in 2..((limit as f64).sqrt() as usize + 1) {
        if sieve[i] {
            for j in (i * i..limit).step_by(i) {
                sieve[j] = false;
            }
        }
    }
    
    sieve.iter()
        .enumerate()
        .filter(|(_, &is_prime)| is_prime)
        .map(|(i, _)| i as u64)
        .collect()
}

/// Miller-Rabin primality test
fn is_probable_prime(n: u64) -> bool {
    if n < 2 { return false; }
    if n == 2 || n == 3 { return true; }
    if n % 2 == 0 { return false; }
    
    // Write n-1 as 2^r * d
    let mut d = n - 1;
    let mut r = 0;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }
    
    // Witnesses for deterministic test up to 2^64
    let witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    
    'witness: for &a in &witnesses {
        if a >= n { continue; }
        
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 { continue; }
        
        for _ in 0..r - 1 {
            x = mod_mul(x, x, n);
            if x == n - 1 { continue 'witness; }
        }
        
        return false;
    }
    
    true
}

/// Modular exponentiation
fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1;
    base %= modulus;
    
    while exp > 0 {
        if exp % 2 == 1 {
            result = mod_mul(result, base, modulus);
        }
        exp >>= 1;
        base = mod_mul(base, base, modulus);
    }
    
    result
}

/// Modular multiplication avoiding overflow
fn mod_mul(a: u64, b: u64, modulus: u64) -> u64 {
    ((a as u128 * b as u128) % modulus as u128) as u64
}

/// Public API for composite factorization
pub fn factorize_composite_cuda(n: u64) -> CudaResult<Vec<u64>> {
    let device = CudaDevice::new(0)?;
    let factorizer = CudaCompositeFactorizer::new(&device)?;
    factorizer.factorize(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_composite_classification() {
        if !crate::cuda::cuda_available() {
            println!("Skipping CUDA test - no GPU available");
            return;
        }
        
        let device = CudaDevice::new(0).unwrap();
        let factorizer = CudaCompositeFactorizer::new(&device).unwrap();
        
        // Test semiprime
        assert_eq!(factorizer.classify_composite(15), CompositeType::GeneralComposite);
        
        // Test power of prime
        assert_eq!(factorizer.classify_composite(32), CompositeType::PowerOfPrime); // 2^5
        
        // Test highly composite
        assert_eq!(factorizer.classify_composite(720), CompositeType::HighlyComposite); // 2^4 * 3^2 * 5
    }
    
    #[test]
    fn test_fermat_factorization() {
        if !crate::cuda::cuda_available() {
            println!("Skipping CUDA test - no GPU available");
            return;
        }
        
        let device = CudaDevice::new(0).unwrap();
        let factorizer = CudaCompositeFactorizer::new(&device).unwrap();
        
        // Test number close to perfect square: 403 = 13 * 31
        let result = factorizer.fermat_factorize(403).unwrap();
        assert!(result.is_some());
        let (f1, f2) = result.unwrap();
        assert_eq!(f1 * f2, 403);
    }
    
    #[test]
    fn test_pollard_rho() {
        if !crate::cuda::cuda_available() {
            println!("Skipping CUDA test - no GPU available");
            return;
        }
        
        let device = CudaDevice::new(0).unwrap();
        let factorizer = CudaCompositeFactorizer::new(&device).unwrap();
        
        // Test composite number
        let result = factorizer.pollard_rho_factorize(8051).unwrap();
        assert!(result.is_some());
        let (f1, f2) = result.unwrap();
        assert_eq!(f1 * f2, 8051);
    }
    
    #[test]
    fn test_full_factorization() {
        if !crate::cuda::cuda_available() {
            println!("Skipping CUDA test - no GPU available");
            return;
        }
        
        // Test various composite numbers
        let test_cases = vec![
            (24, vec![2, 2, 2, 3]),
            (100, vec![2, 2, 5, 5]),
            (1001, vec![7, 11, 13]),
            (9797, vec![97, 101]), // semiprime
        ];
        
        for (n, expected) in test_cases {
            let mut factors = factorize_composite_cuda(n).unwrap();
            factors.sort();
            assert_eq!(factors, expected, "Failed for n={}", n);
        }
    }
}