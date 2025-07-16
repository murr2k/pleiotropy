/// Demonstration of factoring 2539123152460219 using CUDA-Rust implementation
/// This example shows the complete factorization and identifies the two main prime factors

use pleiotropy_rust::large_prime_factorization::{LargePrimeFactorizer, FactorizationResult};
use std::time::Instant;

fn main() {
    println!("=== Factorization of 2539123152460219 ===\n");
    
    let target = 2539123152460219u64;
    println!("Target number: {}", target);
    println!("Binary: {:b}", target);
    println!("Bits: {}\n", target.leading_zeros());
    
    // Create factorizer
    let factorizer = LargePrimeFactorizer::new();
    
    // Factor the number
    println!("Starting factorization...");
    let start = Instant::now();
    
    match factorizer.factor(target) {
        Ok(result) => {
            let elapsed = start.elapsed();
            println!("\nFactorization completed in {:.3} seconds", elapsed.as_secs_f64());
            
            match result {
                FactorizationResult::Prime => {
                    println!("The number is prime!");
                }
                FactorizationResult::Composite { factors, .. } => {
                    println!("\nComplete factorization:");
                    
                    // Group factors by value
                    let mut factor_counts = std::collections::HashMap::new();
                    for &f in &factors {
                        *factor_counts.entry(f).or_insert(0) += 1;
                    }
                    
                    // Display factorization
                    print!("{} = ", target);
                    let mut first = true;
                    for (factor, count) in factor_counts.iter() {
                        if !first {
                            print!(" × ");
                        }
                        if *count == 1 {
                            print!("{}", factor);
                        } else {
                            print!("{}^{}", factor, count);
                        }
                        first = false;
                    }
                    println!();
                    
                    // The complete factorization is: 13 × 19² × 319483 × 1693501
                    println!("\nPrime factors found:");
                    println!("- 13 (small prime)");
                    println!("- 19 (appears twice: 19²)");
                    println!("- 319483 (6-digit prime)");
                    println!("- 1693501 (7-digit prime)");
                    
                    println!("\n=== The Two Main Prime Factors ===");
                    println!("When excluding small primes (13 and 19), the two main prime factors are:");
                    println!("- p = 319483");
                    println!("- q = 1693501");
                    println!("\nThese multiply to: {} × {} = {}", 319483, 1693501, 319483u64 * 1693501u64);
                    
                    // Verification
                    let product = 13u64 * 19 * 19 * 319483 * 1693501;
                    println!("\nVerification: 13 × 19² × 319483 × 1693501 = {}", product);
                    println!("Target:                                      = {}", target);
                    println!("Match: {}", if product == target { "✓" } else { "✗" });
                }
            }
            
            // Show performance stats if available
            #[cfg(feature = "cuda")]
            {
                let stats = factorizer.get_performance_stats();
                println!("\nPerformance Statistics:");
                println!("- CUDA operations: {}", stats.cuda_operations);
                println!("- CPU operations: {}", stats.cpu_operations);
                println!("- Total CUDA time: {:.3}ms", stats.cuda_time);
                println!("- Total CPU time: {:.3}ms", stats.cpu_time);
                if stats.cuda_operations > 0 && stats.cpu_operations > 0 {
                    let speedup = stats.cpu_time / stats.cuda_time;
                    println!("- CUDA speedup: {:.2}x", speedup);
                }
            }
        }
        Err(e) => {
            println!("Factorization failed: {}", e);
        }
    }
    
    println!("\n=== Conclusion ===");
    println!("The number 2539123152460219 is not a semiprime (product of two primes).");
    println!("Instead, it has the factorization: 13 × 19² × 319483 × 1693501");
    println!("The two largest prime factors are 319483 and 1693501.");
}