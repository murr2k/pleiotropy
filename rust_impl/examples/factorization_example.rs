use genomic_pleiotropy_cryptanalysis::{ComputeBackend, large_prime_factorization::FactorizationResult};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    
    // Create compute backend with automatic CUDA detection
    let mut backend = ComputeBackend::new()?;
    
    println!("Factorization Example - CUDA available: {}", backend.is_cuda_available());
    println!("=================================================");
    
    // Example 1: Factor the target number
    let target = 2539123152460219u64;
    println!("\nFactoring target number: {}", target);
    
    // Add progress callback
    backend.add_factorization_progress_callback(Box::new(|progress| {
        print!("\rProgress: {:.1}%", progress * 100.0);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
    }));
    
    match backend.factorize_u64(target) {
        Ok(result) => {
            println!("\n\nFactorization complete!");
            println!("Number: {}", result.number);
            println!("Factors: {:?}", result.factors);
            println!("Time: {:.2}ms", result.elapsed_ms);
            println!("Used CUDA: {}", result.used_cuda);
            
            // Verify the result
            let product: u128 = result.factors.iter().product();
            println!("Verification: {} = {}", product, target);
            assert_eq!(product, target as u128);
        }
        Err(e) => {
            println!("\nFactorization failed: {}", e);
        }
    }
    
    // Example 2: Batch factorization
    println!("\n\nBatch Factorization Example");
    println!("===========================");
    
    let numbers = vec![
        100822548703,  // Product of two primes
        123456789,     // Composite
        97,            // Prime
        1000000007,    // Large prime
    ];
    
    println!("Factoring {} numbers...", numbers.len());
    let results = backend.factorize_batch(&numbers)?;
    
    for (num, result) in numbers.iter().zip(results.iter()) {
        println!("\nNumber: {}", num);
        println!("Factors: {:?}", result.factors);
        println!("Time: {:.2}ms", result.elapsed_ms);
        
        // Verify
        let product: u128 = result.factors.iter().product();
        assert_eq!(product, *num as u128);
    }
    
    // Example 3: Async factorization
    println!("\n\nAsync Factorization Example");
    println!("===========================");
    
    let async_number = 9999999967u64;
    println!("Factoring {} asynchronously...", async_number);
    
    let result = backend.factorize_u64_async(async_number).await?;
    println!("Result: {:?}", result.factors);
    
    // Show performance statistics
    let stats = backend.get_stats();
    println!("\n\nPerformance Statistics");
    println!("======================");
    println!("CPU calls: {}", stats.cpu_calls);
    println!("CUDA calls: {}", stats.cuda_calls);
    println!("CUDA failures: {}", stats.cuda_failures);
    println!("Avg CPU time: {:.2}ms", stats.avg_cpu_time_ms);
    println!("Avg CUDA time: {:.2}ms", stats.avg_cuda_time_ms);
    
    Ok(())
}

#[cfg(target_arch = "wasm32")]
mod wasm_example {
    use wasm_bindgen::prelude::*;
    use genomic_pleiotropy_cryptanalysis::large_prime_factorization::wasm::WasmFactorizer;
    
    #[wasm_bindgen(start)]
    pub fn main() {
        // Set panic hook for better error messages in WASM
        console_error_panic_hook::set_once();
    }
    
    #[wasm_bindgen]
    pub async fn factorize_example(number: String) -> Result<String, JsValue> {
        let factorizer = WasmFactorizer::new()?;
        factorizer.factorize_async(&number).await
    }
}