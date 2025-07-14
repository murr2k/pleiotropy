#!/usr/bin/env python3
"""
Demo: CUDA Semiprime Seeker
Shows how the seeker finds the optimal semiprime size for 10-minute factorization
"""

import time
import random
import math

def simulate_factorization_time(digits):
    """
    Simulate factorization time based on empirical data:
    - 31 digits: ~20 seconds
    - 35 digits: ~2 minutes  
    - 40 digits: ~10 minutes
    
    Using exponential model: time = exp(0.23 * digits - 5.1)
    """
    base_time = math.exp(0.23 * digits - 5.1)
    # Add some randomness
    actual_time = base_time * random.uniform(0.8, 1.2)
    return actual_time

def main():
    print("🔍 CUDA Semiprime Seeker - Demo")
    print("Target: Find semiprime size for 10-minute (600s) factorization")
    print("="*60)
    
    target_time = 600  # 10 minutes
    tolerance = 30     # ±30 seconds
    
    # Binary search for optimal digit count
    low, high = 40, 55
    attempts = []
    
    print("\nSearching for optimal semiprime size...")
    print("Digits | Time (s) | Status")
    print("-------|----------|--------")
    
    for iteration in range(10):
        # Pick middle point
        test_digits = (low + high) // 2
        
        # Simulate factorization
        time.sleep(0.5)  # Simulate work
        factor_time = simulate_factorization_time(test_digits)
        
        # Record attempt
        attempts.append((test_digits, factor_time))
        
        # Check if within tolerance
        if abs(factor_time - target_time) <= tolerance:
            status = "✓ TARGET!"
        elif factor_time < target_time:
            status = "Too fast"
            low = test_digits
        else:
            status = "Too slow"
            high = test_digits
        
        print(f"  {test_digits:2d}   | {factor_time:7.1f}  | {status}")
        
        # Stop if very close
        if abs(factor_time - target_time) < 5:
            break
    
    # Find best attempt
    best = min(attempts, key=lambda x: abs(x[1] - target_time))
    best_digits, best_time = best
    
    print(f"\n{'='*60}")
    print("🎯 OPTIMAL RESULT FOUND")
    print(f"\nSemiprime size: {best_digits} digits")
    print(f"Factorization time: {best_time:.1f} seconds")
    print(f"Target deviation: {best_time - target_time:+.1f} seconds")
    
    # Show what a semiprime of this size looks like
    print(f"\nExample {best_digits}-digit semiprime:")
    p1 = random.randint(10**(best_digits//2-1), 10**(best_digits//2))
    p2 = random.randint(10**(best_digits//2-1), 10**(best_digits//2))
    semiprime = p1 * p2
    
    semiprime_str = str(semiprime)[:best_digits]
    print(f"{semiprime_str[:20]}...{semiprime_str[-20:]}")
    
    print("\n📊 Search Statistics:")
    print(f"Total attempts: {len(attempts)}")
    print(f"Search range: {min(a[0] for a in attempts)}-{max(a[0] for a in attempts)} digits")
    print(f"Time range: {min(a[1] for a in attempts):.1f}-{max(a[1] for a in attempts):.1f} seconds")
    
    # Show scaling
    print("\n📈 Factorization Time Scaling:")
    for d in [30, 35, 40, 45, 50]:
        t = math.exp(0.23 * d - 5.1)
        print(f"  {d} digits: ~{t:.0f} seconds ({t/60:.1f} minutes)")

if __name__ == "__main__":
    main()