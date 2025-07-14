#!/usr/bin/env python3
"""
CUDA Semiprime Seeker - Real GPU Version
Finds the largest semiprime that takes ~10 minutes to factor on GTX 2070
"""

import subprocess
import time
import random
import json
import os
import sys
from datetime import datetime
import multiprocessing as mp
from typing import Tuple, Optional, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

# Target parameters
TARGET_TIME_SECS = 600  # 10 minutes
TOLERANCE_SECS = 30     # ±30 seconds

# Search parameters based on our benchmarks
MIN_DIGITS = 45  # Start higher based on simulation
MAX_DIGITS = 55
INITIAL_DIGITS = 50  # Our best estimate

def generate_prime_rust(bits: int) -> str:
    """Generate a prime number using Rust"""
    rust_code = f"""
use rand::Rng;
use num_bigint::{{BigUint, RandBigInt}};
use num_prime::nt_funcs::is_prime;

fn main() {{
    let mut rng = rand::thread_rng();
    loop {{
        let candidate = rng.gen_biguint({bits});
        if is_prime(&candidate, None).probably() {{
            println!("{{}}", candidate);
            break;
        }}
    }}
}}
"""
    
    try:
        # Write temporary Rust file
        with open('/tmp/gen_prime.rs', 'w') as f:
            f.write(rust_code)
        
        # Compile
        compile_result = subprocess.run([
            'rustc', '/tmp/gen_prime.rs',
            '-o', '/tmp/gen_prime',
            '--edition', '2021',
            '-O'
        ], capture_output=True, text=True, timeout=10)
        
        if compile_result.returncode != 0:
            logging.error(f"Prime generation compilation failed: {compile_result.stderr}")
            # Fallback to Python
            import gmpy2
            p = gmpy2.mpz(2) ** bits + random.randint(1, 2**bits - 1)
            while not gmpy2.is_prime(p):
                p = gmpy2.next_prime(p)
            return str(p)
        
        # Run
        result = subprocess.run(['/tmp/gen_prime'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
            
    except Exception as e:
        logging.error(f"Prime generation error: {e}")
    
    # Fallback
    import gmpy2
    p = gmpy2.mpz(2) ** bits + random.randint(1, 2**bits - 1)
    while not gmpy2.is_prime(p):
        p = gmpy2.next_prime(p)
    return str(p)

def generate_semiprime(digits: int) -> Tuple[str, str, str]:
    """Generate a semiprime with specified number of digits"""
    # Calculate bits for each prime factor
    total_bits = int(digits * 3.322)  # log2(10) ≈ 3.322
    bits1 = total_bits // 2 + random.randint(-2, 2)
    bits2 = total_bits - bits1
    
    logging.info(f"Generating {digits}-digit semiprime ({bits1} + {bits2} bits)")
    
    # Generate two primes
    p1 = generate_prime_rust(max(bits1, 20))
    p2 = generate_prime_rust(max(bits2, 20))
    
    # Calculate product
    import gmpy2
    n = gmpy2.mpz(p1) * gmpy2.mpz(p2)
    
    actual_digits = len(str(n))
    logging.info(f"Generated {actual_digits}-digit semiprime")
    
    return str(n), p1, p2

def factor_with_cuda(number: str) -> Optional[Tuple[str, str, float]]:
    """Factor a number using the CUDA composite factorizer"""
    start_time = time.time()
    
    # Use the factor_large_number.py script we created
    try:
        result = subprocess.run([
            sys.executable, 'factor_large_number.py'
        ], input=number, capture_output=True, text=True, timeout=TARGET_TIME_SECS + 60)
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            # Parse output for prime factors
            output = result.stdout
            factor1 = None
            factor2 = None
            
            lines = output.split('\n')
            for line in lines:
                if 'Prime 1:' in line:
                    factor1 = line.split(':')[1].strip().replace(',', '')
                elif 'Prime 2:' in line:
                    factor2 = line.split(':')[1].strip().replace(',', '')
            
            if factor1 and factor2:
                logging.info(f"Factorization successful in {elapsed:.1f}s")
                return (factor1, factor2, elapsed)
            else:
                logging.warning("Could not parse factors from output")
                
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        logging.warning(f"Factorization timed out after {elapsed:.1f}s")
    except Exception as e:
        logging.error(f"Factorization error: {e}")
    
    return None

def seeker_worker(worker_id: int, shared_data: Dict):
    """Worker process for the semiprime seeker"""
    logging.info(f"Worker {worker_id} starting")
    
    attempts = 0
    best_result = None
    best_diff = float('inf')
    
    while attempts < 10:  # Limit attempts per worker
        # Get current target
        target_digits = shared_data['target_digits']
        
        # Add some exploration
        test_digits = target_digits + random.randint(-2, 2)
        test_digits = max(MIN_DIGITS, min(MAX_DIGITS, test_digits))
        
        # Generate semiprime
        logging.info(f"Worker {worker_id}: Generating {test_digits}-digit semiprime")
        n, p1, p2 = generate_semiprime(test_digits)
        actual_digits = len(n)
        
        # Factor it
        logging.info(f"Worker {worker_id}: Factoring {actual_digits}-digit number")
        result = factor_with_cuda(n)
        
        attempts += 1
        shared_data['total_attempts'] += 1
        
        if result:
            f1, f2, time_secs = result
            
            # Check if this is close to target
            time_diff = abs(time_secs - TARGET_TIME_SECS)
            
            status = ""
            if time_diff <= TOLERANCE_SECS:
                status = "✓ TARGET!"
            elif time_secs < TARGET_TIME_SECS - TOLERANCE_SECS:
                status = "Too fast"
                shared_data['target_digits'] = min(MAX_DIGITS, actual_digits + 1)
            else:
                status = "Too slow"
                shared_data['target_digits'] = max(MIN_DIGITS, actual_digits - 1)
            
            logging.info(f"Worker {worker_id}: {actual_digits} digits → {time_secs:.1f}s {status}")
            
            # Update best if closer to target
            if time_diff < best_diff:
                best_diff = time_diff
                best_result = {
                    'number': n,
                    'digits': actual_digits,
                    'factor1': f1,
                    'factor2': f2,
                    'time': time_secs,
                    'worker': worker_id
                }
                
                # Update global best
                if 'best_result' not in shared_data or time_diff < abs(shared_data['best_result']['time'] - TARGET_TIME_SECS):
                    shared_data['best_result'] = best_result
                    logging.info(f"🎯 New best: {actual_digits} digits in {time_secs:.1f}s")
                    
                    # Stop if very close
                    if time_diff < 5:
                        logging.info("Excellent result found, stopping search")
                        break
        else:
            logging.warning(f"Worker {worker_id}: Factorization failed")
        
        # Save intermediate results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_data = {
            'timestamp': timestamp,
            'worker': worker_id,
            'attempt': attempts,
            'digits': actual_digits,
            'time': time_secs if result else None,
            'success': result is not None
        }
        
        with open(f'seeker_worker_{worker_id}_log.json', 'a') as f:
            json.dump(result_data, f)
            f.write('\n')
    
    logging.info(f"Worker {worker_id} completed {attempts} attempts")

def main():
    print("🔍 CUDA Semiprime Seeker - GTX 2070")
    print(f"Target: Find semiprime that takes ~{TARGET_TIME_SECS}s to factor")
    print(f"GPU: NVIDIA GTX 2070")
    print("="*80)
    
    # Check CUDA availability
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_name = result.stdout.strip()
            print(f"✓ GPU detected: {gpu_name}")
        else:
            print("⚠️  Warning: Could not detect GPU")
    except:
        print("⚠️  Warning: nvidia-smi not available")
    
    # Shared data
    manager = mp.Manager()
    shared_data = manager.dict()
    shared_data['target_digits'] = INITIAL_DIGITS
    shared_data['total_attempts'] = 0
    
    # Start workers
    num_workers = min(4, mp.cpu_count())
    processes = []
    
    print(f"\nStarting {num_workers} workers...")
    start_time = time.time()
    
    for i in range(num_workers):
        p = mp.Process(target=seeker_worker, args=(i, shared_data))
        p.start()
        processes.append(p)
    
    # Monitor progress
    try:
        while any(p.is_alive() for p in processes):
            time.sleep(10)
            
            elapsed = time.time() - start_time
            attempts = shared_data.get('total_attempts', 0)
            target = shared_data.get('target_digits', INITIAL_DIGITS)
            
            print(f"\n[{elapsed/60:.1f} min] Progress: {attempts} attempts, targeting {target} digits")
            
            if 'best_result' in shared_data:
                best = shared_data['best_result']
                print(f"  Best: {best['digits']}-digit in {best['time']:.1f}s (Δ{best['time']-TARGET_TIME_SECS:+.1f}s)")
            
            # Stop after 30 minutes
            if elapsed > 1800:
                print("\nTime limit reached")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    # Wait for workers
    for p in processes:
        p.terminate()
        p.join()
    
    # Report results
    print("\n" + "="*80)
    print("🏁 SEARCH COMPLETE")
    
    if 'best_result' in shared_data:
        best = shared_data['best_result']
        print(f"\n✅ Optimal Semiprime Found:")
        print(f"  Digits: {best['digits']}")
        print(f"  Time: {best['time']:.1f} seconds")
        print(f"  Target deviation: {best['time'] - TARGET_TIME_SECS:+.1f} seconds")
        print(f"  Found by: Worker {best['worker']}")
        
        # Save full results
        results = {
            'timestamp': datetime.now().isoformat(),
            'gpu': 'GTX 2070',
            'target_seconds': TARGET_TIME_SECS,
            'tolerance_seconds': TOLERANCE_SECS,
            'total_attempts': shared_data.get('total_attempts', 0),
            'best_result': {
                'digits': best['digits'],
                'time': best['time'],
                'factor1': best['factor1'][:50] + '...' if len(best['factor1']) > 50 else best['factor1'],
                'factor2': best['factor2'][:50] + '...' if len(best['factor2']) > 50 else best['factor2'],
                'worker': best['worker']
            }
        }
        
        with open('cuda_seeker_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n📄 Results saved to cuda_seeker_results.json")
    else:
        print("\n❌ No suitable semiprime found")

if __name__ == "__main__":
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    main()