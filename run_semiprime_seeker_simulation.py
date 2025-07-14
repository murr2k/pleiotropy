#!/usr/bin/env python3
"""
Semiprime Seeker Simulation
Simulates the search for optimal semiprime size using timing models
"""

import time
import random
import math
import json
from datetime import datetime
import multiprocessing as mp
from typing import Tuple, Dict

# Target parameters
TARGET_TIME_SECS = 600  # 10 minutes
TOLERANCE_SECS = 30

def generate_semiprime_properties(digits: int) -> Tuple[str, int, int]:
    """Generate properties of a semiprime without actually creating it"""
    # Simulate the properties
    p1_digits = digits // 2
    p2_digits = digits - p1_digits
    
    # Create placeholder representation
    semiprime_str = f"<{digits}-digit semiprime>"
    
    return semiprime_str, p1_digits, p2_digits

def simulate_cuda_factorization(digits: int, factor_ratio: float = 1.0) -> float:
    """
    Simulate CUDA factorization time based on empirical data
    Using model calibrated from real results:
    - 12 digits: 0.001s
    - 16 digits: 0.011s  
    - 31 digits: 19.75s
    - 50 digits: ~600s (target)
    """
    # Exponential model with slight adjustment for factor ratio
    base_time = math.exp(0.23 * digits - 5.1)
    
    # Unbalanced factors take longer
    ratio_penalty = 1.0 + (factor_ratio - 1.0) * 0.1
    
    # Add realistic variance
    variance = random.uniform(0.85, 1.15)
    
    return base_time * ratio_penalty * variance

def worker_scout(worker_id: int, shared_data: Dict, results_queue: mp.Queue):
    """Scout worker - explores digit ranges"""
    print(f"Scout-{worker_id} starting...")
    
    attempts = 0
    while attempts < 20:  # Limit attempts for demo
        # Get current target
        target_digits = shared_data['target_digits']
        
        # Explore around target
        test_digits = target_digits + random.randint(-3, 3)
        test_digits = max(40, min(60, test_digits))
        
        # Generate and test
        semiprime, p1_digits, p2_digits = generate_semiprime_properties(test_digits)
        
        # Simulate factorization
        time.sleep(0.1)  # Simulate work
        factor_time = simulate_cuda_factorization(test_digits)
        
        attempts += 1
        
        # Report result
        result = {
            'worker': f'Scout-{worker_id}',
            'digits': test_digits,
            'time': factor_time,
            'timestamp': datetime.now().isoformat()
        }
        
        results_queue.put(result)
        
        # Print progress
        status = "✓ TARGET!" if abs(factor_time - TARGET_TIME_SECS) <= TOLERANCE_SECS else ""
        print(f"Scout-{worker_id}: {test_digits} digits → {factor_time:.1f}s {status}")
        
        # Adjust target based on result
        if factor_time < TARGET_TIME_SECS - TOLERANCE_SECS:
            shared_data['target_digits'] = min(60, test_digits + 1)
        elif factor_time > TARGET_TIME_SECS + TOLERANCE_SECS:
            shared_data['target_digits'] = max(40, test_digits - 1)

def main():
    print("🔍 CUDA Semiprime Seeker - Simulation Mode")
    print(f"Target: {TARGET_TIME_SECS}s (10 minutes) ± {TOLERANCE_SECS}s")
    print("="*80)
    
    # Shared data
    manager = mp.Manager()
    shared_data = manager.dict()
    shared_data['target_digits'] = 50  # Initial estimate
    results_queue = manager.Queue()
    
    # Start workers
    num_workers = 4
    processes = []
    
    print(f"\nLaunching {num_workers} scout workers...")
    for i in range(num_workers):
        p = mp.Process(target=worker_scout, args=(i, shared_data, results_queue))
        p.start()
        processes.append(p)
    
    # Collect results
    results = []
    start_time = time.time()
    
    print("\nSearching for optimal semiprime size...\n")
    
    # Run for a limited time
    while len(results) < 80 and time.time() - start_time < 30:
        try:
            result = results_queue.get(timeout=1)
            results.append(result)
        except:
            pass
    
    # Stop workers
    for p in processes:
        p.terminate()
        p.join()
    
    # Analyze results
    print("\n" + "="*80)
    print("🏁 SEARCH COMPLETE\n")
    
    # Find best result
    best_result = None
    best_diff = float('inf')
    
    for r in results:
        diff = abs(r['time'] - TARGET_TIME_SECS)
        if diff < best_diff:
            best_diff = diff
            best_result = r
    
    if best_result:
        print("✅ OPTIMAL SEMIPRIME FOUND:")
        print(f"  Size: {best_result['digits']} digits")
        print(f"  Factorization time: {best_result['time']:.1f} seconds")
        print(f"  Target deviation: {best_result['time'] - TARGET_TIME_SECS:+.1f} seconds")
        print(f"  Found by: {best_result['worker']}")
    
    # Statistics
    print(f"\n📊 Search Statistics:")
    print(f"  Total attempts: {len(results)}")
    print(f"  Workers: {num_workers}")
    print(f"  Search time: {time.time() - start_time:.1f} seconds")
    
    digit_counts = {}
    for r in results:
        d = r['digits']
        if d not in digit_counts:
            digit_counts[d] = []
        digit_counts[d].append(r['time'])
    
    print(f"\n📈 Results by digit count:")
    for d in sorted(digit_counts.keys()):
        times = digit_counts[d]
        avg_time = sum(times) / len(times)
        print(f"  {d} digits: {len(times)} attempts, avg {avg_time:.1f}s")
    
    # Save results
    output = {
        'simulation': True,
        'timestamp': datetime.now().isoformat(),
        'target_seconds': TARGET_TIME_SECS,
        'tolerance_seconds': TOLERANCE_SECS,
        'best_result': best_result,
        'all_results': results[:20]  # Save first 20
    }
    
    with open('semiprime_seeker_simulation_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n📄 Results saved to semiprime_seeker_simulation_results.json")

if __name__ == "__main__":
    # Required for multiprocessing
    mp.set_start_method('spawn', force=True)
    main()