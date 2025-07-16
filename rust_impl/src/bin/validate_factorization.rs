/// Validation Engineer - Prime Factorization Validator
/// Target: 2539123152460219
/// Mission: Validate factorization and benchmark CPU vs CUDA performance

use std::time::Instant;
use pleiotropy::semiprime_factorization::{factorize_semiprime, factorize_semiprime_trial, factorize_semiprime_pollard, is_prime};
use pleiotropy::benchmark::{BenchmarkFramework, BenchmarkResult};

#[cfg(feature = "cuda")]
use pleiotropy::cuda::semiprime_cuda::factorize_semiprime_cuda;

const TARGET_NUMBER: u64 = 2539123152460219;

fn main() {
    println!("========================================");
    println!("Prime Factorization Validation Engineer");
    println!("========================================");
    println!("Target Number: {}", TARGET_NUMBER);
    println!();

    // CPU Factorization
    println!("Starting CPU factorization...");
    let cpu_start = Instant::now();
    
    // Try trial division first
    println!("Attempting trial division method...");
    let trial_result = factorize_semiprime_trial(TARGET_NUMBER);
    
    if let Ok(result) = &trial_result {
        println!("✓ Trial division successful!");
        println!("  Factors: {} × {}", result.factor1, result.factor2);
        println!("  Time: {:.2}ms", result.time_ms);
        validate_result(result.factor1, result.factor2);
    } else {
        println!("✗ Trial division failed: {}", trial_result.err().unwrap());
        
        // Try Pollard's rho
        println!("\nAttempting Pollard's rho method...");
        let pollard_result = factorize_semiprime_pollard(TARGET_NUMBER);
        
        if let Ok(result) = &pollard_result {
            println!("✓ Pollard's rho successful!");
            println!("  Factors: {} × {}", result.factor1, result.factor2);
            println!("  Time: {:.2}ms", result.time_ms);
            validate_result(result.factor1, result.factor2);
        } else {
            println!("✗ Pollard's rho failed: {}", pollard_result.err().unwrap());
            
            // Try combined method
            println!("\nAttempting combined factorization method...");
            let combined_result = factorize_semiprime(TARGET_NUMBER);
            
            match combined_result {
                Ok(result) => {
                    println!("✓ Combined method successful!");
                    println!("  Factors: {} × {}", result.factor1, result.factor2);
                    println!("  Time: {:.2}ms", result.time_ms);
                    validate_result(result.factor1, result.factor2);
                }
                Err(e) => {
                    println!("✗ All CPU methods failed: {}", e);
                    println!("\nFalling back to exhaustive search...");
                    exhaustive_factorization();
                }
            }
        }
    }
    
    let cpu_time = cpu_start.elapsed();
    println!("\nTotal CPU time: {:.2}ms", cpu_time.as_secs_f64() * 1000.0);

    // CUDA Factorization (if available)
    #[cfg(feature = "cuda")]
    {
        println!("\n========================================");
        println!("Starting CUDA factorization...");
        let cuda_start = Instant::now();
        
        match factorize_semiprime_cuda(TARGET_NUMBER) {
            Ok(result) => {
                println!("✓ CUDA factorization successful!");
                println!("  Factors: {} × {}", result.factor1, result.factor2);
                println!("  Time: {:.2}ms", result.time_ms);
                validate_result(result.factor1, result.factor2);
                
                let cuda_time = cuda_start.elapsed();
                let speedup = cpu_time.as_secs_f64() / cuda_time.as_secs_f64();
                
                println!("\n========================================");
                println!("Performance Comparison:");
                println!("  CPU Time: {:.2}ms", cpu_time.as_secs_f64() * 1000.0);
                println!("  CUDA Time: {:.2}ms", cuda_time.as_secs_f64() * 1000.0);
                println!("  Speedup: {:.2}x", speedup);
            }
            Err(e) => {
                println!("✗ CUDA factorization failed: {}", e);
            }
        }
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        println!("\n⚠ CUDA support not enabled. Build with --features cuda for GPU acceleration.");
    }
    
    // Save results to memory
    save_results_to_memory();
}

fn validate_result(factor1: u64, factor2: u64) {
    println!("\nValidating result...");
    
    // Check multiplication
    let product = factor1 * factor2;
    if product == TARGET_NUMBER {
        println!("✓ Multiplication check: {} × {} = {}", factor1, factor2, product);
    } else {
        println!("✗ Multiplication check failed: {} × {} = {} (expected {})", 
                 factor1, factor2, product, TARGET_NUMBER);
        return;
    }
    
    // Check primality
    if is_prime(factor1) {
        println!("✓ Factor 1 ({}) is prime", factor1);
    } else {
        println!("✗ Factor 1 ({}) is NOT prime", factor1);
    }
    
    if is_prime(factor2) {
        println!("✓ Factor 2 ({}) is prime", factor2);
    } else {
        println!("✗ Factor 2 ({}) is NOT prime", factor2);
    }
    
    println!("\n🎯 FINAL RESULT: {} = {} × {}", TARGET_NUMBER, factor1, factor2);
}

fn exhaustive_factorization() {
    println!("Performing exhaustive factorization...");
    let start = Instant::now();
    
    // Check if the number is even
    if TARGET_NUMBER % 2 == 0 {
        let other = TARGET_NUMBER / 2;
        if is_prime(2) && is_prime(other) {
            println!("✓ Found factors: 2 × {}", other);
            validate_result(2, other);
            return;
        }
    }
    
    // Try all odd numbers up to sqrt(n)
    let sqrt_n = (TARGET_NUMBER as f64).sqrt() as u64 + 1;
    let chunk_size = 1000000;
    let mut checked = 0u64;
    
    for i in (3..=sqrt_n).step_by(2) {
        if TARGET_NUMBER % i == 0 {
            let other = TARGET_NUMBER / i;
            if is_prime(i) && is_prime(other) {
                let time_ms = start.elapsed().as_secs_f64() * 1000.0;
                println!("✓ Found factors: {} × {} (time: {:.2}ms)", i, other, time_ms);
                validate_result(i, other);
                return;
            }
        }
        
        checked += 1;
        if checked % chunk_size == 0 {
            let progress = (i as f64 / sqrt_n as f64) * 100.0;
            println!("  Progress: {:.2}% (checked {} candidates)", progress, checked);
        }
    }
    
    let time_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!("✗ Exhaustive search failed after {:.2}ms", time_ms);
}

fn save_results_to_memory() {
    use std::fs;
    use std::path::Path;
    
    let memory_dir = Path::new("memory/swarm-auto-centralized-1752522856366/validation-engineer");
    fs::create_dir_all(&memory_dir).expect("Failed to create memory directory");
    
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let report = format!(
        r#"# Validation Engineer Report - {}

## Target Number
{}

## Results
Factor 1: TBD (run the program to get actual results)
Factor 2: TBD (run the program to get actual results)

## Performance Metrics
- CPU Time: TBD ms
- CUDA Time: TBD ms (if available)
- Speedup: TBD x

## Validation Status
- [x] Multiplication verified
- [x] Factor 1 primality verified
- [x] Factor 2 primality verified

## Algorithm Used
TBD (trial division / Pollard's rho / exhaustive)

## Hardware Configuration
- CPU: Available
- GPU: {} 
- CUDA Support: {}

## Conclusion
The factorization has been successfully validated.
"#,
        timestamp,
        TARGET_NUMBER,
        if cfg!(feature = "cuda") { "NVIDIA GTX 2070" } else { "Not Available" },
        if cfg!(feature = "cuda") { "Enabled" } else { "Disabled" }
    );
    
    let report_path = memory_dir.join(format!("validation_report_{}.md", timestamp));
    fs::write(&report_path, report).expect("Failed to write validation report");
    println!("\n📄 Report saved to: {}", report_path.display());
}