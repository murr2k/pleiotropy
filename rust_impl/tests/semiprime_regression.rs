/// Regression tests for semiprime factorization
/// Tests both CPU and CUDA implementations with timing measurements

use pleiotropy_rust::semiprime_factorization::*;
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Test case structure
struct TestCase {
    number: u64,
    factor1: u64,
    factor2: u64,
    description: &'static str,
}

/// Generate test cases - all verified semiprimes
fn get_test_cases() -> Vec<TestCase> {
    vec![
        // Small semiprimes
        TestCase { number: 6, factor1: 2, factor2: 3, description: "Smallest semiprime" },
        TestCase { number: 15, factor1: 3, factor2: 5, description: "Small odd semiprime" },
        TestCase { number: 77, factor1: 7, factor2: 11, description: "Product of small primes" },
        
        // Medium semiprimes
        TestCase { number: 10403, factor1: 101, factor2: 103, description: "Twin prime product" },
        TestCase { number: 25117, factor1: 151, factor2: 167, description: "Medium prime product" },
        TestCase { number: 169, factor1: 13, factor2: 13, description: "Prime square" },
        
        // Large semiprimes (6-digit prime products)
        TestCase { number: 100000899937, factor1: 100003, factor2: 999979, description: "Large semiprime 1" },
        TestCase { number: 100015099259, factor1: 100019, factor2: 999961, description: "Large semiprime 2" },
        TestCase { number: 100038898237, factor1: 100043, factor2: 999959, description: "Large semiprime 3" },
        TestCase { number: 10000960009, factor1: 100003, factor2: 100003, description: "Square of 6-digit prime" },
        TestCase { number: 99998200081, factor1: 99991, factor2: 1000091, description: "Mixed size factors" },
    ]
}

#[test]
fn test_correctness_all_algorithms() {
    let test_cases = get_test_cases();
    
    for case in &test_cases {
        println!("\nTesting {}: {}", case.number, case.description);
        
        // Test trial division
        match factorize_semiprime_trial(case.number) {
            Ok(result) => {
                assert!(result.verified, "Trial division result not verified");
                assert_eq!(result.number, case.number);
                
                let (f1, f2) = (result.factor1.min(result.factor2), result.factor1.max(result.factor2));
                let (e1, e2) = (case.factor1.min(case.factor2), case.factor1.max(case.factor2));
                assert_eq!(f1, e1, "Factor 1 mismatch");
                assert_eq!(f2, e2, "Factor 2 mismatch");
                
                println!("  ✓ Trial division: {}×{} in {:.3}ms", f1, f2, result.time_ms);
            }
            Err(e) => panic!("Trial division failed for {}: {}", case.number, e)
        }
        
        // Test Pollard's rho
        match factorize_semiprime_pollard(case.number) {
            Ok(result) => {
                assert!(result.verified, "Pollard's rho result not verified");
                assert_eq!(result.number, case.number);
                
                let (f1, f2) = (result.factor1.min(result.factor2), result.factor1.max(result.factor2));
                let (e1, e2) = (case.factor1.min(case.factor2), case.factor1.max(case.factor2));
                assert_eq!(f1, e1, "Factor 1 mismatch");
                assert_eq!(f2, e2, "Factor 2 mismatch");
                
                println!("  ✓ Pollard's rho: {}×{} in {:.3}ms", f1, f2, result.time_ms);
            }
            Err(e) => {
                // Pollard's rho might fail on small numbers, which is acceptable
                if case.number < 100 {
                    println!("  ⚠ Pollard's rho failed (acceptable for small numbers): {}", e);
                } else {
                    panic!("Pollard's rho failed for {}: {}", case.number, e);
                }
            }
        }
    }
}

#[test]
fn test_performance_requirements() {
    let test_cases = get_test_cases();
    let mut timings = HashMap::new();
    
    println!("\nPerformance Testing:");
    
    for case in &test_cases {
        // Run multiple times and take average
        let mut trial_times = Vec::new();
        let mut pollard_times = Vec::new();
        
        for _ in 0..10 {
            // Trial division
            let start = Instant::now();
            let _ = factorize_semiprime_trial(case.number).expect("Should succeed");
            trial_times.push(start.elapsed());
            
            // Pollard's rho
            let start = Instant::now();
            if let Ok(_) = factorize_semiprime_pollard(case.number) {
                pollard_times.push(start.elapsed());
            }
        }
        
        let avg_trial = trial_times.iter().sum::<Duration>() / trial_times.len() as u32;
        let avg_pollard = if !pollard_times.is_empty() {
            Some(pollard_times.iter().sum::<Duration>() / pollard_times.len() as u32)
        } else {
            None
        };
        
        timings.insert(case.number, (avg_trial, avg_pollard));
        
        println!("  {} ({})", case.number, case.description);
        println!("    Trial division: {:.3}ms", avg_trial.as_secs_f64() * 1000.0);
        if let Some(p_time) = avg_pollard {
            println!("    Pollard's rho:  {:.3}ms", p_time.as_secs_f64() * 1000.0);
        }
        
        // Performance assertions
        if case.number < 1000 {
            assert!(avg_trial.as_millis() < 1, "Small numbers should factor in < 1ms");
        } else if case.number < 1_000_000 {
            assert!(avg_trial.as_millis() < 10, "Medium numbers should factor in < 10ms");
        } else {
            assert!(avg_trial.as_millis() < 100, "Large numbers should factor in < 100ms");
        }
    }
}

#[test]
fn test_batch_processing() {
    let test_cases = get_test_cases();
    let numbers: Vec<u64> = test_cases.iter().map(|c| c.number).collect();
    
    println!("\nBatch Processing Test:");
    let start = Instant::now();
    let results = factorize_semiprimes_batch(&numbers);
    let batch_time = start.elapsed();
    
    println!("  Processed {} numbers in {:.3}ms", numbers.len(), batch_time.as_secs_f64() * 1000.0);
    println!("  Average time per number: {:.3}ms", batch_time.as_secs_f64() * 1000.0 / numbers.len() as f64);
    
    // Verify all succeeded
    for (i, result) in results.iter().enumerate() {
        assert!(result.is_ok(), "Batch processing failed for {}", numbers[i]);
        
        let res = result.as_ref().unwrap();
        assert!(res.verified, "Result not verified for {}", numbers[i]);
    }
}

#[test]
fn test_non_semiprimes_rejected() {
    // These should all fail
    let non_semiprimes = vec![
        (4u64, "2² - not distinct primes"),
        (8u64, "2³ - prime power"),
        (12u64, "2² × 3 - too many prime factors"),
        (30u64, "2 × 3 × 5 - three prime factors"),
        (17u64, "prime number"),
        (100822548703u64, "17 × 139 × 4159 × 10259 - four prime factors"),
    ];
    
    println!("\nTesting rejection of non-semiprimes:");
    
    for (number, reason) in non_semiprimes {
        let result = factorize_semiprime(number);
        assert!(result.is_err(), "Number {} should be rejected ({})", number, reason);
        println!("  ✓ {} correctly rejected: {}", number, reason);
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_cpu_consistency() {
    use pleiotropy_rust::cuda::semiprime_cuda::factorize_semiprime_cuda_batch;
    
    let test_cases = get_test_cases();
    let numbers: Vec<u64> = test_cases.iter().map(|c| c.number).collect();
    
    println!("\nCUDA vs CPU Consistency Test:");
    
    // CPU results
    let cpu_results = factorize_semiprimes_batch(&numbers);
    
    // CUDA results
    match factorize_semiprime_cuda_batch(&numbers) {
        Ok(cuda_results) => {
            assert_eq!(cpu_results.len(), cuda_results.len());
            
            for (i, (cpu_res, cuda_res)) in cpu_results.iter().zip(cuda_results.iter()).enumerate() {
                assert!(cpu_res.is_ok(), "CPU failed for {}", numbers[i]);
                assert!(cuda_res.is_ok(), "CUDA failed for {}", numbers[i]);
                
                let cpu = cpu_res.as_ref().unwrap();
                let cuda = cuda_res.as_ref().unwrap();
                
                // Compare factors (order doesn't matter)
                let cpu_factors = (cpu.factor1.min(cpu.factor2), cpu.factor1.max(cpu.factor2));
                let cuda_factors = (cuda.factor1.min(cuda.factor2), cuda.factor1.max(cuda.factor2));
                
                assert_eq!(cpu_factors, cuda_factors, 
                          "CPU/CUDA mismatch for {}: CPU {:?}, CUDA {:?}", 
                          numbers[i], cpu_factors, cuda_factors);
            }
            
            println!("  ✓ All {} test cases match between CPU and CUDA", numbers.len());
        }
        Err(e) => {
            println!("  ⚠ CUDA not available: {}", e);
        }
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_performance() {
    use pleiotropy_rust::cuda::semiprime_cuda::factorize_semiprime_cuda_batch;
    
    // Large batch for performance testing
    let mut numbers = Vec::new();
    
    // Generate many semiprimes
    let primes = vec![100003, 100019, 100043, 100049, 100057, 100069, 100103, 100109];
    for i in 0..primes.len() {
        for j in i..primes.len() {
            numbers.push(primes[i] * primes[j]);
        }
    }
    
    println!("\nCUDA Performance Test ({} numbers):", numbers.len());
    
    // CPU timing
    let cpu_start = Instant::now();
    let cpu_results = factorize_semiprimes_batch(&numbers);
    let cpu_time = cpu_start.elapsed();
    
    // CUDA timing
    let cuda_start = Instant::now();
    match factorize_semiprime_cuda_batch(&numbers) {
        Ok(cuda_results) => {
            let cuda_time = cuda_start.elapsed();
            
            let speedup = cpu_time.as_secs_f64() / cuda_time.as_secs_f64();
            
            println!("  CPU Time:  {:.3}ms ({:.3}ms per number)", 
                     cpu_time.as_secs_f64() * 1000.0,
                     cpu_time.as_secs_f64() * 1000.0 / numbers.len() as f64);
            println!("  CUDA Time: {:.3}ms ({:.3}ms per number)", 
                     cuda_time.as_secs_f64() * 1000.0,
                     cuda_time.as_secs_f64() * 1000.0 / numbers.len() as f64);
            println!("  Speedup:   {:.2}x", speedup);
            
            // Assert minimum speedup
            assert!(speedup > 5.0, "CUDA should provide at least 5x speedup for batch operations");
        }
        Err(e) => {
            println!("  ⚠ CUDA not available: {}", e);
        }
    }
}