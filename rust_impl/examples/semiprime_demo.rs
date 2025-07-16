/// Demonstration of semiprime factorization with CPU vs CUDA comparison
/// This example shows the correct factorization of numbers that are products of exactly two primes

use pleiotropy_rust::semiprime_factorization::{
    factorize_semiprime, factorize_semiprime_trial, factorize_semiprime_pollard,
    factorize_semiprimes_batch, SemiprimeResult
};
use std::time::Instant;

fn main() {
    println!("=== Semiprime Factorization Demonstration ===\n");
    
    // Test cases: actual semiprimes (product of two primes)
    let test_cases = vec![
        // Small semiprimes
        (15u64, "3 × 5"),
        (77u64, "7 × 11"), 
        (221u64, "13 × 17"),
        
        // Medium semiprimes
        (10403u64, "101 × 103"),
        (25117u64, "151 × 167"),
        
        // Large semiprimes (product of 6-digit primes)
        (100000899937u64, "100003 × 999979"),
        (100015099259u64, "100019 × 999961"),
        (100038898237u64, "100043 × 999959"),
    ];
    
    println!("Testing individual factorizations:\n");
    
    for (number, expected) in &test_cases {
        println!("Factoring {}", number);
        
        // Try trial division
        let start = Instant::now();
        match factorize_semiprime_trial(*number) {
            Ok(result) => {
                println!("  Trial Division: {} × {} in {:.3}ms ✓", 
                         result.factor1, result.factor2, result.time_ms);
                assert!(result.verified, "Factorization not verified!");
            }
            Err(e) => {
                println!("  Trial Division: Failed - {}", e);
            }
        }
        
        // Try Pollard's rho
        let start = Instant::now();
        match factorize_semiprime_pollard(*number) {
            Ok(result) => {
                println!("  Pollard's Rho:  {} × {} in {:.3}ms ✓", 
                         result.factor1, result.factor2, result.time_ms);
            }
            Err(e) => {
                println!("  Pollard's Rho:  Failed - {}", e);
            }
        }
        
        println!("  Expected: {}\n", expected);
    }
    
    // Test batch processing
    println!("\nBatch Processing Test:");
    let numbers: Vec<u64> = test_cases.iter().map(|(n, _)| *n).collect();
    
    let start = Instant::now();
    let batch_results = factorize_semiprimes_batch(&numbers);
    let batch_time = start.elapsed();
    
    let successful = batch_results.iter().filter(|r| r.is_ok()).count();
    println!("Batch processed {} numbers in {:.3}ms", numbers.len(), batch_time.as_secs_f64() * 1000.0);
    println!("Success rate: {}/{} ({:.1}%)\n", 
             successful, numbers.len(), 
             successful as f64 / numbers.len() as f64 * 100.0);
    
    // Test non-semiprimes (should fail)
    println!("Testing non-semiprimes (should fail):");
    let non_semiprimes = vec![
        (100822548703u64, "17 × 139 × 4159 × 10259 (4 factors)"),
        (12u64, "2² × 3 (not two distinct primes)"),
        (30u64, "2 × 3 × 5 (3 factors)"),
        (17u64, "prime (not composite)"),
    ];
    
    for (number, description) in non_semiprimes {
        match factorize_semiprime(number) {
            Ok(_) => println!("  {} - ERROR: Should have failed!", number),
            Err(e) => println!("  {} - Correctly rejected: {} ({})", number, e, description),
        }
    }
    
    // Performance comparison
    println!("\n=== Performance Comparison ===");
    
    #[cfg(feature = "cuda")]
    {
        use pleiotropy_rust::cuda::semiprime_cuda::factorize_semiprime_cuda_batch;
        
        println!("\nCPU vs CUDA Comparison:");
        
        // Large batch for performance testing
        let perf_numbers: Vec<u64> = vec![
            100000899937, 100015099259, 100038898237, 
            10000960009, 99998200081, 100109100121,
            999863000221, 999883000303, 999901000909,
        ];
        
        // CPU batch
        let cpu_start = Instant::now();
        let cpu_results = factorize_semiprimes_batch(&perf_numbers);
        let cpu_time = cpu_start.elapsed();
        
        // CUDA batch
        let cuda_start = Instant::now();
        let cuda_results = factorize_semiprime_cuda_batch(&perf_numbers);
        let cuda_time = cuda_start.elapsed();
        
        match cuda_results {
            Ok(_) => {
                println!("CPU Time:  {:.3}ms ({:.3}ms per number)", 
                         cpu_time.as_secs_f64() * 1000.0,
                         cpu_time.as_secs_f64() * 1000.0 / perf_numbers.len() as f64);
                println!("CUDA Time: {:.3}ms ({:.3}ms per number)", 
                         cuda_time.as_secs_f64() * 1000.0,
                         cuda_time.as_secs_f64() * 1000.0 / perf_numbers.len() as f64);
                println!("Speedup:   {:.2}x", cpu_time.as_secs_f64() / cuda_time.as_secs_f64());
            }
            Err(e) => {
                println!("CUDA not available: {}", e);
            }
        }
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        println!("\nCUDA support not compiled. Rebuild with --features cuda for GPU acceleration.");
    }
    
    println!("\n=== Demonstration Complete ===");
}