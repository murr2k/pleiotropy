#!/usr/bin/env python3
"""
CUDA Semiprime Seeker - Swarm Edition
Finds the largest semiprime that takes approximately 10 minutes to factor
Uses a swarm of parallel processes to explore the search space
"""

import multiprocessing as mp
import time
import random
import json
import os
import signal
import sys
from datetime import datetime
import subprocess
import tempfile
from typing import Tuple, Optional, Dict
import gmpy2
from gmpy2 import mpz

# Target solving time
TARGET_TIME_SECS = 600  # 10 minutes
TOLERANCE_SECS = 30     # ±30 seconds

# Search parameters
MIN_DIGITS = 30
MAX_DIGITS = 80
SWARM_SIZE = mp.cpu_count()  # One worker per CPU core

# Shared memory for coordination
manager = mp.Manager()
shared_state = {
    'best_result': manager.dict(),
    'current_target_digits': manager.Value('i', 40),  # Start with 40 digits
    'should_stop': manager.Value('b', False),
    'attempts': manager.Value('i', 0),
    'worker_stats': manager.dict(),
}

def generate_prime(bits: int) -> mpz:
    """Generate a random prime number with approximately the given number of bits"""
    while True:
        # Generate random number with exact bit length
        candidate = mpz(random.getrandbits(bits))
        candidate |= (1 << (bits - 1))  # Ensure MSB is set
        candidate |= 1  # Ensure odd
        
        if gmpy2.is_prime(candidate):
            return candidate

def generate_balanced_semiprime(total_digits: int) -> Tuple[mpz, mpz, mpz]:
    """Generate a semiprime with two prime factors of roughly equal size"""
    # For balanced factors, each should be about half the total digits
    factor_digits = total_digits // 2
    factor_bits = int(factor_digits * 3.322)  # log2(10) ≈ 3.322
    
    # Add some randomness to bit length
    bits1 = factor_bits + random.randint(-5, 5)
    bits2 = factor_bits + random.randint(-5, 5)
    
    p1 = generate_prime(bits1)
    p2 = generate_prime(bits2)
    
    # Ensure p1 < p2 for consistency
    if p1 > p2:
        p1, p2 = p2, p1
    
    semiprime = p1 * p2
    return semiprime, p1, p2

def factor_with_cuda(number: mpz) -> Optional[Tuple[mpz, mpz, float]]:
    """
    Attempt to factor a number using the CUDA factorizer
    Returns (factor1, factor2, time_seconds) or None if failed
    """
    # Create a temporary Rust program to factor the number
    rust_code = f"""
use pleiotropy::cuda::composite_factorizer::factorize_composite_cuda;
use std::time::Instant;

fn main() {{
    if !pleiotropy::cuda::cuda_available() {{
        eprintln!("CUDA not available");
        std::process::exit(1);
    }}
    
    // Parse the large number
    let n_str = "{number}";
    let n = match n_str.parse::<u128>() {{
        Ok(num) => num,
        Err(_) => {{
            eprintln!("Number too large for u128");
            std::process::exit(2);
        }}
    }};
    
    let start = Instant::now();
    
    match factorize_composite_cuda(n as u64) {{
        Ok(factors) => {{
            let elapsed = start.elapsed();
            if factors.len() == 2 {{
                println!("{{}},{{}},{{}}", factors[0], factors[1], elapsed.as_secs_f64());
            }} else {{
                eprintln!("Not a semiprime: {{}} factors", factors.len());
                std::process::exit(3);
            }}
        }}
        Err(e) => {{
            eprintln!("Factorization failed: {{}}", e);
            std::process::exit(4);
        }}
    }}
}}
"""
    
    try:
        # Write temporary Rust file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rs', delete=False) as f:
            f.write(rust_code)
            temp_file = f.name
        
        # Compile
        exe_file = temp_file.replace('.rs', '')
        compile_result = subprocess.run([
            'rustc', temp_file,
            '-L', 'rust_impl/target/release/deps',
            '-o', exe_file,
            '--edition', '2021'
        ], capture_output=True, text=True)
        
        if compile_result.returncode != 0:
            print(f"Compilation failed: {compile_result.stderr}")
            return None
        
        # Run with timeout
        start_time = time.time()
        try:
            result = subprocess.run(
                [exe_file],
                capture_output=True,
                text=True,
                timeout=TARGET_TIME_SECS + 60  # Give some extra time
            )
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                # Parse output
                parts = result.stdout.strip().split(',')
                if len(parts) == 3:
                    f1 = mpz(parts[0])
                    f2 = mpz(parts[1])
                    return (f1, f2, elapsed)
            else:
                print(f"Factorization failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"Factorization timed out after {elapsed:.1f} seconds")
            return None
            
    finally:
        # Cleanup
        for f in [temp_file, exe_file]:
            if os.path.exists(f):
                os.remove(f)
    
    return None

def swarm_worker(worker_id: int, shared_state: Dict):
    """Worker process for the swarm"""
    print(f"Worker {worker_id} starting...")
    
    random.seed(os.getpid())  # Ensure different random seeds
    local_attempts = 0
    
    while not shared_state['should_stop'].value:
        # Get current target
        target_digits = shared_state['current_target_digits'].value
        
        # Add some exploration
        if random.random() < 0.3:
            digits = target_digits + random.randint(-3, 3)
        else:
            digits = target_digits
        
        digits = max(MIN_DIGITS, min(MAX_DIGITS, digits))
        
        # Generate semiprime
        semiprime, true_p1, true_p2 = generate_balanced_semiprime(digits)
        actual_digits = len(str(semiprime))
        
        print(f"Worker {worker_id}: Testing {actual_digits}-digit semiprime")
        
        # Attempt factorization
        result = factor_with_cuda(semiprime)
        
        local_attempts += 1
        shared_state['attempts'].value += 1
        
        if result:
            factor1, factor2, time_secs = result
            
            print(f"Worker {worker_id}: Factored {actual_digits}-digit number in {time_secs:.2f}s")
            
            # Update worker stats
            shared_state['worker_stats'][worker_id] = {
                'attempts': local_attempts,
                'last_digits': actual_digits,
                'last_time': time_secs
            }
            
            # Check if this is close to target
            time_diff = abs(time_secs - TARGET_TIME_SECS)
            
            if time_diff <= TOLERANCE_SECS:
                # Good candidate!
                current_best = shared_state['best_result']
                
                should_update = False
                if not current_best:
                    should_update = True
                else:
                    current_diff = abs(current_best['time'] - TARGET_TIME_SECS)
                    if time_diff < current_diff:
                        should_update = True
                
                if should_update:
                    print(f"\n🎯 Worker {worker_id}: Found candidate! {actual_digits}-digit in {time_secs:.2f}s")
                    shared_state['best_result'] = {
                        'number': str(semiprime),
                        'digits': actual_digits,
                        'factor1': str(factor1),
                        'factor2': str(factor2),
                        'time': time_secs,
                        'worker': worker_id,
                        'attempts': shared_state['attempts'].value
                    }
                    
                    # Stop if very close
                    if time_diff < 5:
                        shared_state['should_stop'].value = True
            
            # Adjust target based on timing
            if time_secs < TARGET_TIME_SECS - TOLERANCE_SECS:
                # Too fast, go bigger
                new_target = actual_digits + 2
                if new_target > shared_state['current_target_digits'].value:
                    shared_state['current_target_digits'].value = new_target
            elif time_secs > TARGET_TIME_SECS + TOLERANCE_SECS:
                # Too slow, go smaller
                new_target = actual_digits - 2
                if new_target < shared_state['current_target_digits'].value:
                    shared_state['current_target_digits'].value = new_target
        
        # Small delay
        time.sleep(0.1)
    
    print(f"Worker {worker_id} stopped after {local_attempts} attempts")

def monitor_progress(shared_state: Dict):
    """Monitor and report progress"""
    start_time = time.time()
    
    while not shared_state['should_stop'].value:
        time.sleep(10)
        
        elapsed = time.time() - start_time
        attempts = shared_state['attempts'].value
        target = shared_state['current_target_digits'].value
        
        print(f"\n[{elapsed/60:.1f} min] Progress: {attempts} attempts, targeting ~{target} digits")
        
        # Show best result
        best = shared_state['best_result']
        if best:
            print(f"  Best: {best['digits']}-digit in {best['time']:.2f}s (target: {TARGET_TIME_SECS}s ±{TOLERANCE_SECS}s)")
        
        # Show worker stats
        stats = shared_state['worker_stats']
        if stats:
            active = sum(1 for w in stats.values() if w.get('last_time', 0) > 0)
            print(f"  Active workers: {active}/{SWARM_SIZE}")
        
        # Stop after 2 hours
        if elapsed > 7200:
            print("\nTime limit reached, stopping...")
            shared_state['should_stop'].value = True
            break

def main():
    print("🔍 CUDA Semiprime Seeker - Swarm Edition")
    print(f"Target: Find largest semiprime that takes ~{TARGET_TIME_SECS/60:.0f} minutes to factor")
    print(f"Swarm size: {SWARM_SIZE} workers")
    print("="*80)
    
    # Start workers
    processes = []
    for i in range(SWARM_SIZE):
        p = mp.Process(target=swarm_worker, args=(i, shared_state))
        p.start()
        processes.append(p)
    
    # Monitor progress in main thread
    try:
        monitor_progress(shared_state)
    except KeyboardInterrupt:
        print("\nInterrupted by user, stopping...")
        shared_state['should_stop'].value = True
    
    # Wait for workers
    for p in processes:
        p.join()
    
    # Report results
    print("\n" + "="*80)
    print("🏁 SEARCH COMPLETE")
    
    best = shared_state['best_result']
    if best:
        print(f"\n✅ Found optimal semiprime:")
        print(f"  Number: {best['number'][:50]}{'...' if len(best['number']) > 50 else ''}")
        print(f"  Digits: {best['digits']}")
        print(f"  Factor 1: {best['factor1'][:30]}{'...' if len(best['factor1']) > 30 else ''}")
        print(f"  Factor 2: {best['factor2'][:30]}{'...' if len(best['factor2']) > 30 else ''}")
        print(f"  Factorization time: {best['time']:.2f} seconds")
        print(f"  Target deviation: {best['time'] - TARGET_TIME_SECS:+.2f} seconds")
        print(f"  Found by: Worker {best['worker']}")
        print(f"  Total attempts: {best['attempts']}")
        
        # Save results
        result_data = {
            'timestamp': datetime.now().isoformat(),
            'target_seconds': TARGET_TIME_SECS,
            'tolerance_seconds': TOLERANCE_SECS,
            'swarm_size': SWARM_SIZE,
            'result': best
        }
        
        with open('semiprime_seeker_results.json', 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print("\n📄 Results saved to semiprime_seeker_results.json")
    else:
        print("\n❌ No suitable semiprime found")

if __name__ == "__main__":
    main()