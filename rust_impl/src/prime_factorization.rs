/// CPU-based prime factorization algorithms
/// Implements efficient trial division and Pollard's rho algorithm

use rayon::prelude::*;
use std::collections::HashMap;

/// Result of prime factorization
#[derive(Debug, Clone, PartialEq)]
pub struct Factorization {
    pub number: u64,
    pub factors: Vec<u64>,
    pub prime_factors: HashMap<u64, u32>, // prime -> exponent
}

impl Factorization {
    /// Create from a list of prime factors
    pub fn from_factors(number: u64, mut factors: Vec<u64>) -> Self {
        factors.sort_unstable();
        
        let mut prime_factors = HashMap::new();
        for &factor in &factors {
            *prime_factors.entry(factor).or_insert(0) += 1;
        }
        
        Self {
            number,
            factors,
            prime_factors,
        }
    }
    
    /// Verify the factorization is correct
    pub fn verify(&self) -> bool {
        let product: u64 = self.factors.iter().product();
        product == self.number
    }
}

/// CPU-based prime factorization engine
pub struct PrimeFactorizer {
    /// Precomputed small primes for trial division
    small_primes: Vec<u64>,
    /// Maximum value for trial division before switching algorithms
    trial_division_limit: u64,
}

impl Default for PrimeFactorizer {
    fn default() -> Self {
        Self::new()
    }
}

impl PrimeFactorizer {
    /// Create a new prime factorizer
    pub fn new() -> Self {
        // Generate small primes up to 10,000 using sieve
        let small_primes = Self::sieve_of_eratosthenes(10_000);
        
        Self {
            small_primes,
            trial_division_limit: 1_000_000,
        }
    }
    
    /// Generate primes up to n using Sieve of Eratosthenes
    fn sieve_of_eratosthenes(n: u64) -> Vec<u64> {
        let n_usize = n as usize;
        let mut is_prime = vec![true; n_usize + 1];
        is_prime[0] = false;
        is_prime[1] = false;
        
        for i in 2..=((n as f64).sqrt() as usize) {
            if is_prime[i] {
                for j in ((i * i)..=n_usize).step_by(i) {
                    is_prime[j] = false;
                }
            }
        }
        
        is_prime.iter()
            .enumerate()
            .filter_map(|(i, &prime)| if prime { Some(i as u64) } else { None })
            .collect()
    }
    
    /// Trial division algorithm
    fn trial_division(&self, mut n: u64) -> Vec<u64> {
        let mut factors = Vec::new();
        
        // Check small primes first
        for &prime in &self.small_primes {
            if prime * prime > n {
                break;
            }
            
            while n % prime == 0 {
                factors.push(prime);
                n /= prime;
            }
            
            if n == 1 {
                return factors;
            }
        }
        
        // Continue with 6k±1 optimization for larger primes
        let mut i = self.small_primes.last().copied().unwrap_or(5) + 2;
        
        // Ensure we start at 6k±1
        while i % 6 != 1 && i % 6 != 5 {
            i += 2;
        }
        
        while i * i <= n && i <= self.trial_division_limit {
            while n % i == 0 {
                factors.push(i);
                n /= i;
            }
            
            // Next candidate
            i += if i % 6 == 1 { 4 } else { 2 };
        }
        
        if n > 1 {
            factors.push(n);
        }
        
        factors
    }
    
    /// Pollard's rho algorithm for finding a factor
    fn pollard_rho(&self, n: u64) -> u64 {
        if n <= 1 {
            return n;
        }
        if n % 2 == 0 {
            return 2;
        }
        
        // Check if n is prime using Miller-Rabin
        if self.is_prime_miller_rabin(n) {
            return n;
        }
        
        let mut x = 2u64;
        let mut y = 2u64;
        let mut d = 1u64;
        let c = 1u64; // Can be randomized for better performance
        
        // f(x) = (x^2 + c) mod n
        let f = |x: u64| -> u64 {
            ((x as u128 * x as u128 + c as u128) % n as u128) as u64
        };
        
        while d == 1 {
            x = f(x);
            y = f(f(y));
            
            let diff = if x > y { x - y } else { y - x };
            d = Self::gcd(diff, n);
        }
        
        if d == n {
            // Failed, try with different starting values
            1
        } else {
            d
        }
    }
    
    /// Miller-Rabin primality test
    fn is_prime_miller_rabin(&self, n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 || n == 3 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }
        
        // Write n-1 as 2^r * d
        let mut r = 0;
        let mut d = n - 1;
        while d % 2 == 0 {
            r += 1;
            d /= 2;
        }
        
        // Witnesses for deterministic test up to 2^64
        let witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
        
        'witness: for &a in &witnesses {
            if a >= n {
                continue;
            }
            
            let mut x = Self::mod_pow(a, d, n);
            if x == 1 || x == n - 1 {
                continue;
            }
            
            for _ in 0..r - 1 {
                x = ((x as u128 * x as u128) % n as u128) as u64;
                if x == n - 1 {
                    continue 'witness;
                }
            }
            
            return false;
        }
        
        true
    }
    
    /// Modular exponentiation: base^exp mod modulus
    fn mod_pow(base: u64, mut exp: u64, modulus: u64) -> u64 {
        if modulus == 1 {
            return 0;
        }
        
        let mut result = 1u64;
        let mut base = base % modulus;
        
        while exp > 0 {
            if exp % 2 == 1 {
                result = ((result as u128 * base as u128) % modulus as u128) as u64;
            }
            exp >>= 1;
            base = ((base as u128 * base as u128) % modulus as u128) as u64;
        }
        
        result
    }
    
    /// Greatest common divisor using Euclidean algorithm
    fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }
    
    /// Factorize a single number
    pub fn factorize(&self, n: u64) -> Factorization {
        if n <= 1 {
            return Factorization::from_factors(n, vec![]);
        }
        
        let mut factors = Vec::new();
        let mut remaining = n;
        
        // First try trial division for small factors
        if remaining <= self.trial_division_limit * self.trial_division_limit {
            factors = self.trial_division(remaining);
        } else {
            // Factor out small primes first
            for &prime in &self.small_primes {
                while remaining % prime == 0 {
                    factors.push(prime);
                    remaining /= prime;
                }
                
                if remaining == 1 {
                    break;
                }
            }
            
            // Use Pollard's rho for larger factors
            while remaining > 1 && !self.is_prime_miller_rabin(remaining) {
                let factor = self.pollard_rho(remaining);
                if factor == 1 || factor == remaining {
                    // Failed to find factor, assume prime
                    factors.push(remaining);
                    break;
                }
                
                // Factor might be composite, recursively factorize
                let sub_factorization = self.factorize(factor);
                factors.extend(sub_factorization.factors);
                
                remaining /= factor;
            }
            
            if remaining > 1 {
                factors.push(remaining);
            }
        }
        
        Factorization::from_factors(n, factors)
    }
    
    /// Factorize multiple numbers in parallel
    pub fn factorize_batch(&self, numbers: &[u64]) -> Vec<Factorization> {
        numbers.par_iter()
            .map(|&n| self.factorize(n))
            .collect()
    }
    
    /// Factorize with progress callback
    pub fn factorize_with_progress<F>(&self, numbers: &[u64], mut progress: F) -> Vec<Factorization>
    where
        F: FnMut(usize, usize) + Send,
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        
        let completed = Arc::new(AtomicUsize::new(0));
        let total = numbers.len();
        
        let results: Vec<_> = numbers.par_iter()
            .map(|&n| {
                let result = self.factorize(n);
                let count = completed.fetch_add(1, Ordering::Relaxed) + 1;
                progress(count, total);
                result
            })
            .collect();
        
        results
    }
}

/// Optimized factorization for the specific test case
pub fn factorize_test_number() -> Factorization {
    let factorizer = PrimeFactorizer::new();
    let target = 100822548703u64;
    
    let start = std::time::Instant::now();
    let result = factorizer.factorize(target);
    let elapsed = start.elapsed();
    
    println!("CPU Factorization of {} completed in {:?}", target, elapsed);
    println!("Factors: {:?}", result.factors);
    println!("Prime factors: {:?}", result.prime_factors);
    println!("Verification: {}", result.verify());
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_small_primes() {
        let factorizer = PrimeFactorizer::new();
        
        // Test small primes
        assert_eq!(factorizer.factorize(2).factors, vec![2]);
        assert_eq!(factorizer.factorize(3).factors, vec![3]);
        assert_eq!(factorizer.factorize(5).factors, vec![5]);
        assert_eq!(factorizer.factorize(7).factors, vec![7]);
    }
    
    #[test]
    fn test_composite_numbers() {
        let factorizer = PrimeFactorizer::new();
        
        // Test composite numbers
        assert_eq!(factorizer.factorize(12).factors, vec![2, 2, 3]);
        assert_eq!(factorizer.factorize(100).factors, vec![2, 2, 5, 5]);
        assert_eq!(factorizer.factorize(1001).factors, vec![7, 11, 13]);
    }
    
    #[test]
    fn test_target_number() {
        let factorizer = PrimeFactorizer::new();
        
        // Test the specific target
        let target = 100822548703u64;
        let result = factorizer.factorize(target);
        
        assert!(result.verify());
        assert_eq!(result.factors.len(), 2);
        assert!(result.factors.contains(&316907));
        assert!(result.factors.contains(&318089));
    }
    
    #[test]
    fn test_large_primes() {
        let factorizer = PrimeFactorizer::new();
        
        // Test large prime
        let large_prime = 1000000007u64;
        let result = factorizer.factorize(large_prime);
        assert_eq!(result.factors, vec![large_prime]);
    }
    
    #[test]
    fn test_batch_factorization() {
        let factorizer = PrimeFactorizer::new();
        
        let numbers = vec![12, 35, 77, 100822548703];
        let results = factorizer.factorize_batch(&numbers);
        
        assert_eq!(results.len(), 4);
        assert_eq!(results[0].factors, vec![2, 2, 3]);
        assert_eq!(results[1].factors, vec![5, 7]);
        assert_eq!(results[2].factors, vec![7, 11]);
        assert!(results[3].verify());
    }
}