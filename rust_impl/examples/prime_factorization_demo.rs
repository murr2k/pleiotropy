/// Demonstration of prime factorization with CPU and CUDA implementations
/// 
/// This example shows how to factorize the target number 100822548703
/// which equals 316907 × 318089

use anyhow::Result;
use pleiotropy_rust_impl::{
    prime_factorization::{PrimeFactorizer, factorize_test_number},
    prime_compute_backend::{PrimeComputeBackend, demo_prime_factorization},
};
use std::time::Instant;

fn main() -> Result<()> {
    env_logger::init();
    
    println!("===== Prime Factorization Demo =====");
    println!("Target number: 100822548703 (= 316907 × 318089)");
    println!();
    
    // Test CPU implementation directly
    println!("1. Testing CPU Implementation:");
    println!("-" * 40);
    let _ = factorize_test_number();
    println!();
    
    // Test unified compute backend
    println!("2. Testing Unified Compute Backend:");
    println!("-" * 40);
    demo_prime_factorization()?;
    println!();
    
    // Benchmark different number sizes
    println!("3. Benchmarking Different Number Sizes:");
    println!("-" * 40);
    benchmark_factorization()?;
    
    Ok(())
}

fn benchmark_factorization() -> Result<()> {
    let mut backend = PrimeComputeBackend::new()?;
    
    // Test cases of increasing difficulty
    let test_cases = vec![
        ("Small composite", vec![12, 35, 77, 143, 221]),
        ("Medium primes", vec![10007, 100003, 1000003, 10000019]),
        ("Large composites", vec![
            1000000007 * 13,  // Product of large prime and small
            999999937 * 999999929,  // Product of two large primes
            100822548703,  // Our target
        ]),
        ("Very large", vec![
            123456789012345,
            987654321098765,
            111111111111111,
        ]),
    ];
    
    for (category, numbers) in test_cases {
        println!("\n{} numbers:", category);
        
        let start = Instant::now();
        let results = backend.factorize_batch(&numbers)?;
        let elapsed = start.elapsed();
        
        for (number, result) in numbers.iter().zip(results.iter()) {
            println!("  {} = {}", 
                number,
                result.factors.iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(" × ")
            );
        }
        
        println!("  Time: {:?}", elapsed);
    }
    
    println!("\n4. Large Batch Performance Test:");
    println!("-" * 40);
    
    // Generate a large batch
    let mut large_batch = Vec::new();
    for i in 0..10000 {
        large_batch.push(1000000 + i * 31);
    }
    
    // Test with CPU
    backend.set_force_cpu(true);
    let cpu_start = Instant::now();
    let _ = backend.factorize_batch(&large_batch[..100])?; // Test subset
    let cpu_time = cpu_start.elapsed();
    
    // Test with GPU if available
    if backend.is_cuda_available() {
        backend.set_force_cpu(false);
        let gpu_start = Instant::now();
        let _ = backend.factorize_batch(&large_batch[..100])?; // Test subset
        let gpu_time = gpu_start.elapsed();
        
        println!("CPU time (100 numbers): {:?}", cpu_time);
        println!("GPU time (100 numbers): {:?}", gpu_time);
        println!("Speedup: {:.2}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
        
        // Test full batch with GPU
        println!("\nFull batch (10,000 numbers) with GPU:");
        let full_start = Instant::now();
        let _ = backend.factorize_batch(&large_batch)?;
        let full_time = full_start.elapsed();
        println!("Time: {:?}", full_time);
        println!("Numbers per second: {:.0}", 10000.0 / full_time.as_secs_f64());
    } else {
        println!("CUDA not available, showing CPU performance only");
        println!("CPU time (100 numbers): {:?}", cpu_time);
    }
    
    // Print final statistics
    println!();
    backend.print_performance_summary();
    
    Ok(())
}