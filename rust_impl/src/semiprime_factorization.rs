/// Specialized factorization for semiprimes (numbers with exactly two prime factors)
/// This module provides optimized algorithms for factoring composite numbers
/// that are the product of exactly two prime numbers.

use std::time::Instant;
use rayon::prelude::*;

/// Result of semiprime factorization
#[derive(Debug, Clone, PartialEq)]
pub struct SemiprimeResult {
    pub number: u64,
    pub factor1: u64,
    pub factor2: u64,
    pub time_ms: f64,
    pub algorithm: String,
    pub verified: bool,
}

impl SemiprimeResult {
    /// Create a new semiprime result and verify it
    pub fn new(number: u64, factor1: u64, factor2: u64, time_ms: f64, algorithm: String) -> Self {
        let verified = factor1 * factor2 == number && is_prime(factor1) && is_prime(factor2);
        Self {
            number,
            factor1,
            factor2,
            time_ms,
            algorithm,
            verified,
        }
    }
}

/// Check if a number is prime using deterministic Miller-Rabin test
pub fn is_prime(n: u64) -> bool {
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
    let mut d = n - 1;
    let mut r = 0;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }

    // Witnesses for deterministic test up to 2^64
    let witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    
    'witness_loop: for &a in &witnesses {
        if a >= n {
            continue;
        }
        
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue 'witness_loop;
        }
        
        for _ in 0..r - 1 {
            x = mod_mul(x, x, n);
            if x == n - 1 {
                continue 'witness_loop;
            }
        }
        
        return false;
    }
    
    true
}

/// Modular exponentiation: (base^exp) % modulus
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

/// Modular multiplication: (a * b) % modulus, avoiding overflow
fn mod_mul(a: u64, b: u64, modulus: u64) -> u64 {
    ((a as u128 * b as u128) % modulus as u128) as u64
}

/// Trial division optimized for semiprimes
pub fn factorize_semiprime_trial(n: u64) -> Result<SemiprimeResult, String> {
    let start = Instant::now();
    
    // Quick check for even numbers
    if n % 2 == 0 {
        let other = n / 2;
        if is_prime(2) && is_prime(other) {
            let time_ms = start.elapsed().as_secs_f64() * 1000.0;
            return Ok(SemiprimeResult::new(n, 2, other, time_ms, "trial_even".to_string()));
        }
    }
    
    // Check small primes first
    let small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73];
    for &p in &small_primes {
        if n % p == 0 {
            let other = n / p;
            if other != p && is_prime(other) {
                let time_ms = start.elapsed().as_secs_f64() * 1000.0;
                return Ok(SemiprimeResult::new(n, p, other, time_ms, "trial_small".to_string()));
            }
        }
    }
    
    // Full trial division up to sqrt(n)
    let sqrt_n = (n as f64).sqrt() as u64 + 1;
    
    // Use 6k±1 optimization
    let mut i = 5;
    while i <= sqrt_n {
        if n % i == 0 {
            let other = n / i;
            if is_prime(i) && is_prime(other) {
                let time_ms = start.elapsed().as_secs_f64() * 1000.0;
                return Ok(SemiprimeResult::new(n, i, other, time_ms, "trial_division".to_string()));
            }
        }
        
        if i + 2 <= sqrt_n && n % (i + 2) == 0 {
            let other = n / (i + 2);
            if is_prime(i + 2) && is_prime(other) {
                let time_ms = start.elapsed().as_secs_f64() * 1000.0;
                return Ok(SemiprimeResult::new(n, i + 2, other, time_ms, "trial_division".to_string()));
            }
        }
        
        i += 6;
    }
    
    let time_ms = start.elapsed().as_secs_f64() * 1000.0;
    Err(format!("{} is not a semiprime (time: {:.2}ms)", n, time_ms))
}

/// Pollard's rho algorithm optimized for semiprimes
pub fn factorize_semiprime_pollard(n: u64) -> Result<SemiprimeResult, String> {
    let start = Instant::now();
    
    // Handle small cases
    if n < 4 {
        let time_ms = start.elapsed().as_secs_f64() * 1000.0;
        return Err(format!("{} is too small to be a semiprime (time: {:.2}ms)", n, time_ms));
    }
    
    // Quick even check
    if n % 2 == 0 {
        let other = n / 2;
        if is_prime(2) && is_prime(other) {
            let time_ms = start.elapsed().as_secs_f64() * 1000.0;
            return Ok(SemiprimeResult::new(n, 2, other, time_ms, "pollard_even".to_string()));
        }
    }
    
    // Pollard's rho with different parameters
    let c_values = [1, 2, 3, 5, 7, 11];
    
    for &c in &c_values {
        let mut x = 2;
        let mut y = 2;
        let mut d = 1;
        
        let f = |x: u64| -> u64 {
            mod_mul(x, x, n).wrapping_add(c) % n
        };
        
        let mut iterations = 0;
        const MAX_ITERATIONS: u32 = 100000;
        
        while d == 1 && iterations < MAX_ITERATIONS {
            x = f(x);
            y = f(f(y));
            d = gcd(x.abs_diff(y), n);
            iterations += 1;
        }
        
        if d != 1 && d != n {
            let other = n / d;
            if is_prime(d) && is_prime(other) {
                let time_ms = start.elapsed().as_secs_f64() * 1000.0;
                return Ok(SemiprimeResult::new(n, d.min(other), d.max(other), time_ms, "pollard_rho".to_string()));
            }
        }
    }
    
    let time_ms = start.elapsed().as_secs_f64() * 1000.0;
    Err(format!("{} could not be factored as a semiprime (time: {:.2}ms)", n, time_ms))
}

/// Greatest common divisor
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Combined factorization that tries multiple methods
pub fn factorize_semiprime(n: u64) -> Result<SemiprimeResult, String> {
    // Try trial division first for smaller factors
    if let Ok(result) = factorize_semiprime_trial(n) {
        return Ok(result);
    }
    
    // Fall back to Pollard's rho for larger factors
    factorize_semiprime_pollard(n)
}

/// Batch factorization of multiple semiprimes
pub fn factorize_semiprimes_batch(numbers: &[u64]) -> Vec<Result<SemiprimeResult, String>> {
    numbers.par_iter()
        .map(|&n| factorize_semiprime(n))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_prime() {
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(is_prime(100003));
        assert!(is_prime(999979));
        assert!(!is_prime(4));
        assert!(!is_prime(100000));
    }

    #[test]
    fn test_valid_semiprimes() {
        // These are actual semiprimes (product of two primes)
        let test_cases = vec![
            (15, 3, 5),              // Small semiprime
            (77, 7, 11),             // Medium semiprime
            (100000899937, 100003, 999979), // Large semiprime (6-digit primes)
            (100015099259, 100019, 999961), // Another large semiprime
            (100038898237, 100043, 999959), // Another large semiprime
        ];

        for (n, expected_f1, expected_f2) in test_cases {
            let result = factorize_semiprime(n).expect(&format!("Failed to factor {}", n));
            assert!(result.verified, "Factorization not verified for {}", n);
            assert_eq!(result.number, n);
            
            // Check factors match (order doesn't matter)
            let (f1, f2) = (result.factor1.min(result.factor2), result.factor1.max(result.factor2));
            let (e1, e2) = (expected_f1.min(expected_f2), expected_f1.max(expected_f2));
            assert_eq!(f1, e1, "First factor mismatch for {}", n);
            assert_eq!(f2, e2, "Second factor mismatch for {}", n);
            
            println!("Successfully factored {}: {} × {} in {:.2}ms using {}", 
                     n, f1, f2, result.time_ms, result.algorithm);
        }
    }

    #[test]
    fn test_not_semiprimes() {
        // These numbers are NOT semiprimes
        let non_semiprimes = vec![
            100822548703, // Has 4 prime factors: 17 × 139 × 4159 × 10259
            12,           // 2² × 3 (not exactly two primes)
            30,           // 2 × 3 × 5 (three prime factors)
            17,           // Prime number (not composite)
        ];

        for n in non_semiprimes {
            let result = factorize_semiprime(n);
            assert!(result.is_err(), "Number {} should not factor as semiprime", n);
        }
    }

    #[test]
    fn test_batch_factorization() {
        let numbers = vec![15, 77, 100000899937, 100015099259];
        let results = factorize_semiprimes_batch(&numbers);
        
        let successes = results.iter().filter(|r| r.is_ok()).count();
        assert_eq!(successes, 4, "All test numbers should factor successfully");
    }
}