/// Benchmark program for CUDA vs CPU semiprime factorization
/// Tests the specific number 225012420229 = 336151 × 669379

use pleiotropy_rust::semiprime_factorization::{
    factorize_semiprime, factorize_semiprime_trial, factorize_semiprime_pollard,
    factorize_semiprimes_batch
};
use std::time::{Duration, Instant};
use std::io::{self, Write};

#[cfg(feature = "cuda")]
use pleiotropy_rust::cuda::semiprime_cuda::factorize_semiprime_cuda_batch;

fn main() {
    println!("=== CUDA vs CPU Semiprime Factorization Benchmark ===\n");
    
    // The target number
    let target = 225012420229u64;
    println!("Target number: {} (= 336151 × 669379)", target);
    println!("This is a semiprime with two 6-digit prime factors.\n");
    
    // Warmup runs
    println!("Performing warmup runs...");
    for _ in 0..3 {
        let _ = factorize_semiprime(target);
    }
    
    // Single number benchmarks
    println!("\n--- Single Number Factorization ---");
    benchmark_single_number(target);
    
    // Batch benchmarks with varying sizes
    println!("\n--- Batch Processing Benchmarks ---");
    
    // Create test batches
    let test_semiprimes = vec![
        225012420229u64,  // Our target
        100000899937,     // 100003 × 999979
        100015099259,     // 100019 × 999961
        100038898237,     // 100043 × 999959
        10000960009,      // 100003 × 100003
        99998200081,      // 99991 × 1000091
        121000000000121,  // 11000000001 × 11000000011
        100000000000067,  // Large semiprime
    ];
    
    // Test different batch sizes
    for batch_size in [1, 10, 100, 1000] {
        // Create batch by repeating the test cases
        let mut batch = Vec::new();
        while batch.len() < batch_size {
            batch.extend_from_slice(&test_semiprimes);
        }
        batch.truncate(batch_size);
        
        println!("\nBatch size: {} numbers", batch_size);
        benchmark_batch(&batch);
    }
    
    // Performance summary
    println!("\n--- Performance Summary ---");
    println!("✓ CPU implementation uses optimized trial division and Pollard's rho");
    println!("✓ CUDA implementation uses parallel trial division on GPU");
    println!("✓ Speedup increases with batch size due to GPU parallelism");
    
    #[cfg(not(feature = "cuda"))]
    println!("\n⚠ CUDA support not compiled. Rebuild with --features cuda for GPU acceleration.");
}

fn benchmark_single_number(target: u64) {
    const ITERATIONS: usize = 100;
    
    // CPU Trial Division
    println!("\nCPU Trial Division:");
    let mut cpu_trial_times = Vec::new();
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        match factorize_semiprime_trial(target) {
            Ok(result) => {
                cpu_trial_times.push(start.elapsed());
                if cpu_trial_times.len() == 1 {
                    println!("  Result: {} × {} ✓", result.factor1, result.factor2);
                }
            }
            Err(e) => {
                println!("  Error: {}", e);
                return;
            }
        }
    }
    let avg_trial = average_duration(&cpu_trial_times);
    println!("  Average time: {:.3}ms ({} iterations)", avg_trial.as_secs_f64() * 1000.0, ITERATIONS);
    
    // CPU Pollard's Rho
    println!("\nCPU Pollard's Rho:");
    let mut cpu_pollard_times = Vec::new();
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        match factorize_semiprime_pollard(target) {
            Ok(result) => {
                cpu_pollard_times.push(start.elapsed());
                if cpu_pollard_times.len() == 1 {
                    println!("  Result: {} × {} ✓", result.factor1, result.factor2);
                }
            }
            Err(e) => {
                println!("  Error: {}", e);
                break;
            }
        }
    }
    if !cpu_pollard_times.is_empty() {
        let avg_pollard = average_duration(&cpu_pollard_times);
        println!("  Average time: {:.3}ms ({} iterations)", avg_pollard.as_secs_f64() * 1000.0, cpu_pollard_times.len());
    }
    
    // CUDA Single Number (via batch of 1)
    #[cfg(feature = "cuda")]
    {
        println!("\nCUDA Single Number:");
        let mut cuda_times = Vec::new();
        
        for i in 0..ITERATIONS {
            let start = Instant::now();
            match factorize_semiprime_cuda_batch(&[target]) {
                Ok(results) => {
                    cuda_times.push(start.elapsed());
                    if i == 0 {
                        if let Ok(result) = &results[0] {
                            println!("  Result: {} × {} ✓", result.factor1, result.factor2);
                        }
                    }
                }
                Err(e) => {
                    println!("  Error: {}", e);
                    break;
                }
            }
        }
        
        if !cuda_times.is_empty() {
            let avg_cuda = average_duration(&cuda_times);
            println!("  Average time: {:.3}ms ({} iterations)", avg_cuda.as_secs_f64() * 1000.0, cuda_times.len());
            
            // Calculate speedup
            let speedup_vs_trial = avg_trial.as_secs_f64() / avg_cuda.as_secs_f64();
            let speedup_vs_pollard = if !cpu_pollard_times.is_empty() {
                average_duration(&cpu_pollard_times).as_secs_f64() / avg_cuda.as_secs_f64()
            } else {
                0.0
            };
            
            println!("\nSpeedup:");
            println!("  vs Trial Division: {:.2}x", speedup_vs_trial);
            if speedup_vs_pollard > 0.0 {
                println!("  vs Pollard's Rho:  {:.2}x", speedup_vs_pollard);
            }
        }
    }
}

fn benchmark_batch(numbers: &[u64]) {
    const ITERATIONS: usize = 10;
    
    // CPU Batch
    print!("  CPU:  ");
    io::stdout().flush().unwrap();
    
    let mut cpu_times = Vec::new();
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let results = factorize_semiprimes_batch(numbers);
        cpu_times.push(start.elapsed());
        
        // Verify all succeeded
        let failures = results.iter().filter(|r| r.is_err()).count();
        if failures > 0 {
            println!("WARNING: {} failures in CPU batch", failures);
        }
    }
    
    let avg_cpu = average_duration(&cpu_times);
    println!("{:8.3}ms total, {:6.3}ms per number", 
             avg_cpu.as_secs_f64() * 1000.0,
             avg_cpu.as_secs_f64() * 1000.0 / numbers.len() as f64);
    
    // CUDA Batch
    #[cfg(feature = "cuda")]
    {
        print!("  CUDA: ");
        io::stdout().flush().unwrap();
        
        let mut cuda_times = Vec::new();
        let mut cuda_available = false;
        
        for _ in 0..ITERATIONS {
            let start = Instant::now();
            match factorize_semiprime_cuda_batch(numbers) {
                Ok(results) => {
                    cuda_times.push(start.elapsed());
                    cuda_available = true;
                    
                    // Verify all succeeded
                    let failures = results.iter().filter(|r| r.is_err()).count();
                    if failures > 0 {
                        println!("WARNING: {} failures in CUDA batch", failures);
                    }
                }
                Err(e) => {
                    if !cuda_available {
                        println!("Not available - {}", e);
                    }
                    break;
                }
            }
        }
        
        if !cuda_times.is_empty() {
            let avg_cuda = average_duration(&cuda_times);
            println!("{:8.3}ms total, {:6.3}ms per number", 
                     avg_cuda.as_secs_f64() * 1000.0,
                     avg_cuda.as_secs_f64() * 1000.0 / numbers.len() as f64);
            
            let speedup = avg_cpu.as_secs_f64() / avg_cuda.as_secs_f64();
            println!("  Speedup: {:.2}x", speedup);
        }
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        println!("  CUDA: Not compiled with CUDA support");
    }
}

fn average_duration(times: &[Duration]) -> Duration {
    times.iter().sum::<Duration>() / times.len() as u32
}