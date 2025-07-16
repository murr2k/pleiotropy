/// Semiprime factorization benchmark
/// Tests factorization of numbers that are products of exactly two prime numbers

use super::{Benchmarkable, BenchmarkResult};
use crate::semiprime_factorization::{factorize_semiprime, factorize_semiprime_trial, factorize_semiprime_pollard};
use anyhow::Result;
use std::time::Instant;

/// Semiprime factorization benchmark implementation
pub struct SemiprimeBenchmark {
    pub test_cases: Vec<(u64, u64, u64)>, // (number, factor1, factor2)
}

impl Default for SemiprimeBenchmark {
    fn default() -> Self {
        Self {
            test_cases: vec![
                // Small semiprimes for quick tests
                (15, 3, 5),
                (77, 7, 11),
                (221, 13, 17),
                
                // Medium semiprimes
                (10403, 101, 103),
                (25117, 151, 167),
                
                // Large semiprimes (product of 6-digit primes)
                (100000899937, 100003, 999979),
                (100015099259, 100019, 999961),
                (100038898237, 100043, 999959),
                
                // Additional large test cases
                (10000960009, 100003, 100003), // Square of a prime
                (99998200081, 99991, 1000091),
            ],
        }
    }
}

impl SemiprimeBenchmark {
    /// Create a custom benchmark with specific test cases
    pub fn new(test_cases: Vec<(u64, u64, u64)>) -> Self {
        Self { test_cases }
    }
    
    /// Verify that the factors are correct
    fn verify_factors(&self, number: u64, factor1: u64, factor2: u64) -> bool {
        factor1 * factor2 == number
    }
}

impl Benchmarkable for SemiprimeBenchmark {
    fn name(&self) -> &str {
        "Semiprime Factorization"
    }
    
    fn description(&self) -> &str {
        "Factors composite numbers that are the product of exactly two prime numbers"
    }
    
    fn run_cpu(&self) -> Result<BenchmarkResult> {
        let start = Instant::now();
        let mut operations = 0;
        let mut errors = 0;
        
        for &(number, expected_f1, expected_f2) in &self.test_cases {
            match factorize_semiprime_trial(number) {
                Ok(result) => {
                    operations += 1;
                    
                    // Verify the result
                    let (f1, f2) = (result.factor1.min(result.factor2), result.factor1.max(result.factor2));
                    let (e1, e2) = (expected_f1.min(expected_f2), expected_f1.max(expected_f2));
                    
                    if f1 != e1 || f2 != e2 {
                        errors += 1;
                        eprintln!("CPU factorization error for {}: expected {}×{}, got {}×{}", 
                                 number, e1, e2, f1, f2);
                    }
                }
                Err(e) => {
                    errors += 1;
                    eprintln!("CPU factorization failed for {}: {}", number, e);
                }
            }
        }
        
        let duration = start.elapsed();
        
        Ok(BenchmarkResult {
            implementation: "CPU".to_string(),
            duration,
            operations,
            throughput: operations as f64 / duration.as_secs_f64(),
            speedup: 1.0,
            errors,
            details: format!("Trial division algorithm, {} test cases", self.test_cases.len()),
        })
    }
    
    fn run_gpu(&self) -> Result<BenchmarkResult> {
        #[cfg(feature = "cuda")]
        {
            use crate::cuda::semiprime_cuda::factorize_semiprime_cuda_batch;
            
            let start = Instant::now();
            let numbers: Vec<u64> = self.test_cases.iter().map(|(n, _, _)| *n).collect();
            
            match factorize_semiprime_cuda_batch(&numbers) {
                Ok(results) => {
                    let mut errors = 0;
                    
                    for (i, result) in results.iter().enumerate() {
                        let (expected_f1, expected_f2) = (self.test_cases[i].1, self.test_cases[i].2);
                        let (e1, e2) = (expected_f1.min(expected_f2), expected_f1.max(expected_f2));
                        
                        match result {
                            Ok(gpu_result) => {
                                let (f1, f2) = (gpu_result.factor1.min(gpu_result.factor2), 
                                               gpu_result.factor1.max(gpu_result.factor2));
                                
                                if f1 != e1 || f2 != e2 {
                                    errors += 1;
                                    eprintln!("GPU factorization error for {}: expected {}×{}, got {}×{}", 
                                             numbers[i], e1, e2, f1, f2);
                                }
                            }
                            Err(e) => {
                                errors += 1;
                                eprintln!("GPU factorization failed for {}: {}", numbers[i], e);
                            }
                        }
                    }
                    
                    let duration = start.elapsed();
                    let operations = self.test_cases.len();
                    
                    Ok(BenchmarkResult {
                        implementation: "CUDA".to_string(),
                        duration,
                        operations,
                        throughput: operations as f64 / duration.as_secs_f64(),
                        speedup: 1.0, // Will be calculated by runner
                        errors,
                        details: format!("CUDA parallel factorization, {} test cases", operations),
                    })
                }
                Err(e) => Err(anyhow::anyhow!("CUDA factorization failed: {}", e))
            }
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            Err(anyhow::anyhow!("CUDA support not enabled"))
        }
    }
    
    fn validate(&self) -> Result<()> {
        // Verify all test cases are valid semiprimes
        for &(number, factor1, factor2) in &self.test_cases {
            if factor1 * factor2 != number {
                return Err(anyhow::anyhow!(
                    "Invalid test case: {} != {} × {}", 
                    number, factor1, factor2
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_validation() {
        let benchmark = SemiprimeBenchmark::default();
        assert!(benchmark.validate().is_ok());
    }

    #[test]
    fn test_cpu_benchmark() {
        let benchmark = SemiprimeBenchmark::new(vec![
            (15, 3, 5),
            (77, 7, 11),
        ]);
        
        let result = benchmark.run_cpu().expect("CPU benchmark should succeed");
        assert_eq!(result.operations, 2);
        assert_eq!(result.errors, 0);
    }
}