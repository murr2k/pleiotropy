/// CUDA Composite Number Factorizer Demo
/// Demonstrates various factorization algorithms for different composite number types

use pleiotropy::cuda::{
    composite_factorizer::{CudaCompositeFactorizer, factorize_composite_cuda, CompositeType},
    CudaDevice,
};
use std::time::Instant;
use colored::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=== CUDA Composite Number Factorizer Demo ===".bold().green());
    println!();
    
    // Check CUDA availability
    if !pleiotropy::cuda::cuda_available() {
        eprintln!("{}", "Error: CUDA is not available on this system".red());
        return Ok(());
    }
    
    // Get device info
    if let Some(info) = pleiotropy::cuda::cuda_info() {
        println!("{}", "CUDA Device Information:".blue());
        println!("{}", info);
        println!();
    }
    
    // Create factorizer
    let device = CudaDevice::new(0)?;
    let factorizer = CudaCompositeFactorizer::new(&device)?;
    
    // Demo 1: Classify different composite types
    println!("{}", "1. Composite Number Classification".bold().cyan());
    println!("{}", "─".repeat(50));
    
    let classification_examples = vec![
        (32, "2^5 - Power of prime"),
        (720, "2^4 × 3^2 × 5 - Highly composite"),
        (143, "11 × 13 - Semiprime"),
        (403, "13 × 31 - Close to perfect square"),
        (1024, "2^10 - Power of 2"),
        (100822548703u64, "Large semiprime"),
    ];
    
    for (n, description) in classification_examples {
        let composite_type = factorizer.classify_composite(n);
        println!("{}: {} → {:?}", n, description, composite_type);
    }
    
    // Demo 2: Fermat's method for numbers close to perfect squares
    println!("\n{}", "2. Fermat's Method (for numbers close to perfect squares)".bold().cyan());
    println!("{}", "─".repeat(50));
    
    let fermat_examples = vec![403, 1517, 4189, 5767];
    
    for n in fermat_examples {
        let start = Instant::now();
        match factorizer.fermat_factorize(n)? {
            Some((f1, f2)) => {
                let elapsed = start.elapsed();
                println!("{} = {} × {} (found in {:?})", 
                    n.to_string().yellow(), 
                    f1.to_string().green(), 
                    f2.to_string().green(),
                    elapsed
                );
            }
            None => {
                println!("{} - Fermat's method failed", n.to_string().red());
            }
        }
    }
    
    // Demo 3: Pollard's rho for general composites
    println!("\n{}", "3. Pollard's Rho Algorithm".bold().cyan());
    println!("{}", "─".repeat(50));
    
    let pollard_examples = vec![8051, 455459, 1299709];
    
    for n in pollard_examples {
        let start = Instant::now();
        match factorizer.pollard_rho_factorize(n)? {
            Some((f1, f2)) => {
                let elapsed = start.elapsed();
                println!("{} = {} × {} (found in {:?})", 
                    n.to_string().yellow(), 
                    f1.to_string().green(), 
                    f2.to_string().green(),
                    elapsed
                );
            }
            None => {
                println!("{} - Pollard's rho failed", n.to_string().red());
            }
        }
    }
    
    // Demo 4: Complete factorization with automatic algorithm selection
    println!("\n{}", "4. Complete Factorization (automatic algorithm selection)".bold().cyan());
    println!("{}", "─".repeat(50));
    
    let complete_examples = vec![
        24,           // 2^3 × 3
        100,          // 2^2 × 5^2
        720,          // 2^4 × 3^2 × 5
        1001,         // 7 × 11 × 13
        123456,       // Mixed factors
        100822548703, // Large semiprime
    ];
    
    for n in complete_examples {
        let start = Instant::now();
        let factors = factorizer.factorize(n)?;
        let elapsed = start.elapsed();
        
        // Verify factorization
        let product: u64 = factors.iter().product();
        let verification = if product == n { "✓".green() } else { "✗".red() };
        
        println!("{} = {} {} (in {:?})", 
            n.to_string().yellow(),
            factors.iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join(" × "),
            verification,
            elapsed
        );
    }
    
    // Demo 5: Performance comparison
    println!("\n{}", "5. Performance Comparison".bold().cyan());
    println!("{}", "─".repeat(50));
    
    let benchmark_number = 100822548703u64;
    
    println!("Factoring {} using different methods:", benchmark_number);
    
    // Fermat's method
    let start = Instant::now();
    let fermat_result = factorizer.fermat_factorize(benchmark_number)?;
    let fermat_time = start.elapsed();
    
    // Pollard's rho
    let start = Instant::now();
    let pollard_result = factorizer.pollard_rho_factorize(benchmark_number)?;
    let pollard_time = start.elapsed();
    
    // Automatic selection
    let start = Instant::now();
    let auto_result = factorizer.factorize(benchmark_number)?;
    let auto_time = start.elapsed();
    
    println!("  Fermat's method: {:?} - {:?}", 
        fermat_result.map(|(a, b)| format!("{} × {}", a, b)).unwrap_or("Failed".to_string()),
        fermat_time
    );
    println!("  Pollard's rho: {:?} - {:?}", 
        pollard_result.map(|(a, b)| format!("{} × {}", a, b)).unwrap_or("Failed".to_string()),
        pollard_time
    );
    println!("  Auto selection: {:?} - {:?}", 
        auto_result.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(" × "),
        auto_time
    );
    
    // Demo 6: Batch factorization
    println!("\n{}", "6. Batch Factorization Performance".bold().cyan());
    println!("{}", "─".repeat(50));
    
    let batch_numbers: Vec<u64> = vec![
        1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008, 9009, 10010,
        10201, 10403, 10609, 10816, 11021, 11227, 11437, 11644, 11856, 12067,
    ];
    
    let start = Instant::now();
    let mut success_count = 0;
    
    for &n in &batch_numbers {
        let factors = factorize_composite_cuda(n)?;
        if !factors.is_empty() {
            success_count += 1;
        }
    }
    
    let total_time = start.elapsed();
    let avg_time = total_time.as_secs_f64() / batch_numbers.len() as f64 * 1000.0;
    
    println!("Factored {} numbers in {:?}", batch_numbers.len(), total_time);
    println!("Average time per number: {:.2} ms", avg_time);
    println!("Success rate: {}/{} ({:.1}%)", 
        success_count, 
        batch_numbers.len(),
        success_count as f64 / batch_numbers.len() as f64 * 100.0
    );
    
    // Summary
    println!("\n{}", "Summary".bold().green());
    println!("{}", "─".repeat(50));
    println!("The CUDA Composite Factorizer provides:");
    println!("  • {} - Identifies the structure of composite numbers", "Intelligent classification".bright_blue());
    println!("  • {} - Fermat, Pollard's rho, trial division, etc.", "Multiple algorithms".bright_blue());
    println!("  • {} - Automatically selects the best algorithm", "Adaptive selection".bright_blue());
    println!("  • {} - Leverages GPU parallelism for speed", "GPU acceleration".bright_blue());
    println!("  • {} - Handles various composite number types", "Comprehensive coverage".bright_blue());
    
    Ok(())
}