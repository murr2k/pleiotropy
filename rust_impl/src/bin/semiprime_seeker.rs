/// CUDA Semiprime Seeker
/// Finds the largest semiprime (product of exactly two primes) that takes approximately 10 minutes to solve
/// Uses a swarm of parallel searchers to explore the search space efficiently

use pleiotropy::cuda::composite_factorizer::{CudaCompositeFactorizer, factorize_composite_cuda};
use pleiotropy::cuda::CudaDevice;
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, AtomicU64, Ordering}};
use std::thread;
use rand::Rng;
use num_bigint::{BigUint, RandBigInt};
use num_prime::nt_funcs::is_prime;

/// Target solving time in seconds
const TARGET_TIME_SECS: u64 = 600; // 10 minutes
const TOLERANCE_SECS: u64 = 30; // ±30 seconds tolerance

/// Search parameters
const MIN_DIGITS: usize = 30; // Start with 30-digit numbers
const MAX_DIGITS: usize = 60; // Maximum digits to try
const SWARM_SIZE: usize = 8; // Number of parallel searchers

/// Result of a factorization attempt
#[derive(Debug, Clone)]
struct FactorizationResult {
    number: BigUint,
    factor1: BigUint,
    factor2: BigUint,
    time_secs: f64,
    digits: usize,
}

/// Shared state for the swarm
struct SwarmState {
    best_result: Mutex<Option<FactorizationResult>>,
    current_digit_target: AtomicU64,
    should_stop: AtomicBool,
    attempts: AtomicU64,
}

/// Generate a random prime of approximately n bits
fn generate_prime(bits: usize) -> BigUint {
    let mut rng = rand::thread_rng();
    loop {
        let candidate = rng.gen_biguint(bits);
        if is_prime(&candidate, None).probably() {
            return candidate;
        }
    }
}

/// Generate a semiprime with balanced factors
fn generate_balanced_semiprime(total_digits: usize) -> (BigUint, BigUint, BigUint) {
    // For balanced factors, each should be about half the total digits
    let factor_digits = total_digits / 2;
    let factor_bits = (factor_digits as f64 * 3.322) as usize; // log2(10) ≈ 3.322
    
    let p1 = generate_prime(factor_bits);
    let p2 = generate_prime(factor_bits);
    let semiprime = &p1 * &p2;
    
    (semiprime, p1, p2)
}

/// Worker function for each swarm member
fn swarm_worker(
    worker_id: usize,
    state: Arc<SwarmState>,
    device: Arc<CudaDevice>,
) {
    println!("Worker {} starting...", worker_id);
    
    let factorizer = match CudaCompositeFactorizer::new(&device) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Worker {} failed to create factorizer: {}", worker_id, e);
            return;
        }
    };
    
    let mut rng = rand::thread_rng();
    
    while !state.should_stop.load(Ordering::Relaxed) {
        // Get current target digit count
        let target_digits = state.current_digit_target.load(Ordering::Relaxed) as usize;
        
        // Add some randomness to explore nearby sizes
        let digits = if rng.gen_bool(0.7) {
            target_digits
        } else {
            target_digits + rng.gen_range(-2..=2)
        }.max(MIN_DIGITS).min(MAX_DIGITS);
        
        // Generate a semiprime
        let (semiprime, true_p1, true_p2) = generate_balanced_semiprime(digits);
        let semiprime_str = semiprime.to_string();
        let actual_digits = semiprime_str.len();
        
        // Skip if we can't convert to u64 (too large)
        let semiprime_u64 = match semiprime_str.parse::<u64>() {
            Ok(n) => n,
            Err(_) => {
                println!("Worker {}: Number too large for u64, skipping", worker_id);
                continue;
            }
        };
        
        println!("Worker {}: Testing {}-digit semiprime", worker_id, actual_digits);
        
        // Attempt factorization
        let start = Instant::now();
        let result = factorizer.factorize(semiprime_u64);
        let elapsed = start.elapsed();
        
        state.attempts.fetch_add(1, Ordering::Relaxed);
        
        match result {
            Ok(factors) if factors.len() == 2 => {
                let time_secs = elapsed.as_secs_f64();
                
                println!(
                    "Worker {}: Factored {}-digit number in {:.2}s",
                    worker_id, actual_digits, time_secs
                );
                
                // Check if this is close to our target time
                let time_diff = (time_secs - TARGET_TIME_SECS as f64).abs();
                
                if time_diff <= TOLERANCE_SECS as f64 {
                    // This is a good candidate!
                    let result = FactorizationResult {
                        number: semiprime.clone(),
                        factor1: BigUint::from(factors[0]),
                        factor2: BigUint::from(factors[1]),
                        time_secs,
                        digits: actual_digits,
                    };
                    
                    // Update best result if this is better
                    let mut best = state.best_result.lock().unwrap();
                    let should_update = match &*best {
                        None => true,
                        Some(current) => {
                            // Prefer results closer to target time
                            let current_diff = (current.time_secs - TARGET_TIME_SECS as f64).abs();
                            time_diff < current_diff
                        }
                    };
                    
                    if should_update {
                        println!(
                            "\n🎯 Worker {}: Found candidate! {}-digit number solved in {:.2}s",
                            worker_id, actual_digits, time_secs
                        );
                        *best = Some(result);
                        
                        // Signal other workers to stop if we're very close
                        if time_diff < 5.0 {
                            state.should_stop.store(true, Ordering::Relaxed);
                        }
                    }
                }
                
                // Adjust target based on timing
                if time_secs < (TARGET_TIME_SECS - TOLERANCE_SECS) as f64 {
                    // Too fast, try larger numbers
                    state.current_digit_target.fetch_max(
                        (actual_digits + 1) as u64,
                        Ordering::Relaxed
                    );
                } else if time_secs > (TARGET_TIME_SECS + TOLERANCE_SECS) as f64 {
                    // Too slow, try smaller numbers
                    state.current_digit_target.fetch_min(
                        (actual_digits - 1) as u64,
                        Ordering::Relaxed
                    );
                }
            }
            Ok(_) => {
                println!("Worker {}: Not a semiprime, got {} factors", worker_id, factors.len());
            }
            Err(e) => {
                eprintln!("Worker {}: Factorization failed: {}", worker_id, e);
            }
        }
        
        // Small delay to prevent CPU spinning
        thread::sleep(Duration::from_millis(100));
    }
    
    println!("Worker {} stopped", worker_id);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 CUDA Semiprime Seeker");
    println!("Target: Find largest semiprime that takes ~10 minutes to factor");
    println!("Swarm size: {} workers", SWARM_SIZE);
    println!("="*80);
    
    // Check CUDA availability
    if !pleiotropy::cuda::cuda_available() {
        eprintln!("Error: CUDA is not available");
        return Ok(());
    }
    
    // Initialize shared state
    let state = Arc::new(SwarmState {
        best_result: Mutex::new(None),
        current_digit_target: AtomicU64::new(35), // Start with 35-digit numbers
        should_stop: AtomicBool::new(false),
        attempts: AtomicU64::new(0),
    });
    
    // Create shared CUDA device
    let device = Arc::new(CudaDevice::new(0)?);
    
    // Launch swarm workers
    let mut handles = vec![];
    for i in 0..SWARM_SIZE {
        let state_clone = Arc::clone(&state);
        let device_clone = Arc::clone(&device);
        
        let handle = thread::spawn(move || {
            swarm_worker(i, state_clone, device_clone);
        });
        
        handles.push(handle);
    }
    
    // Monitor progress
    let start_time = Instant::now();
    loop {
        thread::sleep(Duration::from_secs(10));
        
        let attempts = state.attempts.load(Ordering::Relaxed);
        let current_target = state.current_digit_target.load(Ordering::Relaxed);
        let elapsed = start_time.elapsed().as_secs();
        
        println!(
            "\n[{:02}:{:02}] Progress: {} attempts, targeting ~{} digits",
            elapsed / 60, elapsed % 60, attempts, current_target
        );
        
        // Check if we have a good result
        let best = state.best_result.lock().unwrap();
        if let Some(ref result) = *best {
            println!(
                "  Best so far: {}-digit number in {:.2}s (target: {}s ±{}s)",
                result.digits, result.time_secs, TARGET_TIME_SECS, TOLERANCE_SECS
            );
            
            // Stop if we've been running for over an hour
            if elapsed > 3600 {
                println!("\nReached time limit, stopping search...");
                state.should_stop.store(true, Ordering::Relaxed);
                break;
            }
        }
        
        // Also stop if all workers have naturally stopped
        if state.should_stop.load(Ordering::Relaxed) {
            break;
        }
    }
    
    // Wait for all workers to finish
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Report final results
    println!("\n" + &"="*80);
    println!("🏁 SEARCH COMPLETE");
    
    let best = state.best_result.lock().unwrap();
    match &*best {
        Some(result) => {
            println!("\n✅ Found optimal semiprime:");
            println!("  Number: {} ({} digits)", result.number, result.digits);
            println!("  Factor 1: {} ({} digits)", result.factor1, result.factor1.to_string().len());
            println!("  Factor 2: {} ({} digits)", result.factor2, result.factor2.to_string().len());
            println!("  Factorization time: {:.2} seconds", result.time_secs);
            println!("  Target deviation: {:+.2} seconds", result.time_secs - TARGET_TIME_SECS as f64);
            
            // Save result to file
            let report = format!(
                "CUDA Semiprime Seeker Results\n\
                ============================\n\n\
                Target Time: {} seconds (10 minutes)\n\
                Tolerance: ±{} seconds\n\n\
                Optimal Semiprime Found:\n\
                Number: {}\n\
                Digits: {}\n\
                Factor 1: {}\n\
                Factor 2: {}\n\
                Factorization Time: {:.2} seconds\n\
                Workers: {}\n\
                Total Attempts: {}\n",
                TARGET_TIME_SECS, TOLERANCE_SECS,
                result.number, result.digits,
                result.factor1, result.factor2,
                result.time_secs, SWARM_SIZE,
                state.attempts.load(Ordering::Relaxed)
            );
            
            std::fs::write("semiprime_seeker_results.txt", report)?;
            println!("\n📄 Results saved to semiprime_seeker_results.txt");
        }
        None => {
            println!("\n❌ No suitable semiprime found within constraints");
        }
    }
    
    Ok(())
}