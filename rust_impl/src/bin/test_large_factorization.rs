use std::sync::Arc;
use pleiotropy_rust::cuda::{CudaDevice, kernels::LargeFactorizer};

fn main() {
    println!("Testing CUDA Large Number Factorization");
    println!("========================================");
    
    // Initialize CUDA
    let device = match CudaDevice::new(0) {
        Ok(dev) => {
            println!("CUDA Device Initialized:");
            println!("{}", dev.info());
            Arc::new(dev)
        }
        Err(e) => {
            eprintln!("Failed to initialize CUDA: {}", e);
            eprintln!("Make sure you have built with --features cuda");
            return;
        }
    };
    
    // Create large factorizer
    let factorizer = match LargeFactorizer::new(device.clone()) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to create factorizer: {}", e);
            return;
        }
    };
    
    // Target number to factor
    let target = 2539123152460219u64;
    println!("\nFactoring: {}", target);
    println!("This is a ~51-bit number");
    
    // Measure performance
    let start = std::time::Instant::now();
    
    match factorizer.factor_large(target) {
        Ok(factors) => {
            let elapsed = start.elapsed();
            
            println!("\nFactorization complete in {:.3} seconds", elapsed.as_secs_f64());
            println!("Factors found: {:?}", factors);
            
            // Verify the factorization
            let product: u64 = factors.iter().product();
            println!("\nVerification: {} = {}", product, target);
            
            if product == target {
                println!("✓ Factorization is correct!");
                
                // Check if factors are prime
                println!("\nChecking primality of factors:");
                for &factor in &factors {
                    match factorizer.is_prime(factor) {
                        Ok(is_prime) => {
                            println!("  {} is prime: {}", factor, is_prime);
                        }
                        Err(e) => {
                            eprintln!("  Failed to check primality of {}: {}", factor, e);
                        }
                    }
                }
            } else {
                println!("✗ Factorization is incorrect!");
            }
        }
        Err(e) => {
            eprintln!("Factorization failed: {}", e);
        }
    }
    
    // Test batch factorization with similar sized numbers
    println!("\n\nTesting batch factorization:");
    let test_numbers = vec![
        100822548703u64,      // Known: 316907 × 318089
        2539123152460219u64,  // Our target
        1234567890123u64,     // Random large number
        9876543210987u64,     // Another large number
    ];
    
    let start = std::time::Instant::now();
    match factorizer.factor_batch_large(&test_numbers) {
        Ok(results) => {
            let elapsed = start.elapsed();
            println!("Batch factorization of {} numbers completed in {:.3} seconds", 
                test_numbers.len(), elapsed.as_secs_f64());
            
            for (i, (number, factors)) in test_numbers.iter().zip(results.iter()).enumerate() {
                println!("\n{}: {} = {:?}", i + 1, number, factors);
                let product: u64 = factors.iter().product();
                if product == *number {
                    println!("   ✓ Correct");
                } else {
                    println!("   ✗ Incorrect (product = {})", product);
                }
            }
        }
        Err(e) => {
            eprintln!("Batch factorization failed: {}", e);
        }
    }
    
    // Performance comparison
    println!("\n\nPerformance Analysis:");
    println!("- Parallel trial division with shared memory optimization");
    println!("- Brent's improved Pollard's rho for large factors");
    println!("- Montgomery multiplication for fast modular arithmetic");
    println!("- Optimized for GTX 2070: 2304 CUDA cores, 8GB memory");
    println!("- Uses compute capability 7.5 specific features");
}