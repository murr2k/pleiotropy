/// Prime factorization benchmark
/// Tests factorization of 100822548703 = 316907 × 318089

use super::Benchmarkable;
use anyhow::Result;

/// Prime factorization benchmark implementation
pub struct PrimeFactorizationBenchmark {
    pub number: u64,
    pub expected_factors: (u64, u64),
}

impl Default for PrimeFactorizationBenchmark {
    fn default() -> Self {
        Self {
            number: 100822548703,
            expected_factors: (316907, 318089),
        }
    }
}

impl PrimeFactorizationBenchmark {
    /// Create a new prime factorization benchmark
    pub fn new(number: u64, expected_factors: (u64, u64)) -> Self {
        Self { number, expected_factors }
    }
    
    /// CPU implementation using trial division
    fn factorize_cpu(&self, n: u64) -> Vec<u64> {
        if n <= 1 {
            return vec![];
        }
        
        let mut factors = Vec::new();
        let mut num = n;
        
        // Check for factor of 2
        while num % 2 == 0 {
            factors.push(2);
            num /= 2;
        }
        
        // Check odd factors up to sqrt(n)
        let mut i = 3;
        while i * i <= num {
            while num % i == 0 {
                factors.push(i);
                num /= i;
            }
            i += 2;
        }
        
        // If num > 1, then it's a prime factor
        if num > 1 {
            factors.push(num);
        }
        
        factors
    }
    
    /// Verify the factors multiply to the original number
    fn verify_factors(&self, factors: &[u64]) -> bool {
        let product = factors.iter().product::<u64>();
        product == self.number
    }
}

impl Benchmarkable for PrimeFactorizationBenchmark {
    fn run_cpu(&self) -> Result<Vec<u64>> {
        let factors = self.factorize_cpu(self.number);
        
        // Verify the result
        if !self.verify_factors(&factors) {
            return Err(anyhow::anyhow!("CPU factorization failed verification"));
        }
        
        Ok(factors)
    }
    
    fn run_cuda(&self) -> Result<Vec<u64>> {
        #[cfg(feature = "cuda")]
        {
            // Get CUDA implementation
            match crate::cuda::prime_factorization::factorize_cuda(self.number) {
                Ok(factors) => {
                    if !self.verify_factors(&factors) {
                        return Err(anyhow::anyhow!("CUDA factorization failed verification"));
                    }
                    Ok(factors)
                }
                Err(e) => Err(anyhow::anyhow!("CUDA factorization failed: {}", e))
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(anyhow::anyhow!("CUDA support not compiled"))
        }
    }
    
    fn name(&self) -> &str {
        "Prime Factorization"
    }
    
    fn iterations(&self) -> usize {
        // Fewer iterations for prime factorization as it's expensive
        10
    }
}

/// Specialized benchmarks for different sized numbers
pub struct SmallPrimeBenchmark;

impl Benchmarkable for SmallPrimeBenchmark {
    fn run_cpu(&self) -> Result<Vec<u64>> {
        let benchmark = PrimeFactorizationBenchmark::new(1000003, (1000003, 1));
        benchmark.run_cpu()
    }
    
    fn run_cuda(&self) -> Result<Vec<u64>> {
        let benchmark = PrimeFactorizationBenchmark::new(1000003, (1000003, 1));
        benchmark.run_cuda()
    }
    
    fn name(&self) -> &str {
        "Small Prime (1000003)"
    }
    
    fn iterations(&self) -> usize {
        100
    }
}

pub struct MediumCompositeBenchmark;

impl Benchmarkable for MediumCompositeBenchmark {
    fn run_cpu(&self) -> Result<Vec<u64>> {
        let benchmark = PrimeFactorizationBenchmark::new(123456789, (3, 41152263));
        benchmark.run_cpu()
    }
    
    fn run_cuda(&self) -> Result<Vec<u64>> {
        let benchmark = PrimeFactorizationBenchmark::new(123456789, (3, 41152263));
        benchmark.run_cuda()
    }
    
    fn name(&self) -> &str {
        "Medium Composite (123456789)"
    }
    
    fn iterations(&self) -> usize {
        50
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_prime_factorization_cpu() {
        let benchmark = PrimeFactorizationBenchmark::default();
        let factors = benchmark.run_cpu().unwrap();
        
        // Check we got exactly 2 prime factors
        assert_eq!(factors.len(), 2);
        
        // Verify they multiply to the original
        let product: u64 = factors.iter().product();
        assert_eq!(product, 100822548703);
        
        // Check they match expected factors (order may vary)
        assert!(
            (factors[0] == 316907 && factors[1] == 318089) ||
            (factors[0] == 318089 && factors[1] == 316907)
        );
    }
    
    #[test]
    fn test_small_prime() {
        let benchmark = SmallPrimeBenchmark;
        let factors = benchmark.run_cpu().unwrap();
        assert_eq!(factors, vec![1000003]); // It's prime
    }
    
    #[test]
    fn test_verification() {
        let benchmark = PrimeFactorizationBenchmark::default();
        assert!(benchmark.verify_factors(&[316907, 318089]));
        assert!(!benchmark.verify_factors(&[2, 3, 5])); // Wrong factors
    }
}