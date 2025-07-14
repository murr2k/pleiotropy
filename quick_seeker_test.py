#!/usr/bin/env python3
"""
Quick test to estimate the right semiprime size
"""

import subprocess
import time
import gmpy2
import random

def test_size(digits):
    """Test a specific digit size"""
    print(f"\nTesting {digits}-digit semiprime...")
    
    # Generate primes
    prime_bits = int(digits * 3.322 / 2)
    
    p1 = gmpy2.mpz(2) ** prime_bits + random.randint(10**5, 10**6)
    p1 = gmpy2.next_prime(p1)
    
    p2 = gmpy2.mpz(2) ** prime_bits + random.randint(10**5, 10**6)  
    p2 = gmpy2.next_prime(p2)
    
    n = p1 * p2
    n_str = str(n)
    actual_digits = len(n_str)
    
    print(f"Generated {actual_digits}-digit number")
    print(f"First 30 digits: {n_str[:30]}...")
    
    # Time the factorization
    start = time.time()
    
    try:
        result = subprocess.run(
            ['python3', 'factor_large_number.py'],
            input=n_str,
            capture_output=True,
            text=True,
            timeout=1200  # 20 minute timeout
        )
        
        elapsed = time.time() - start
        
        if result.returncode == 0 and "✓ This IS a semiprime!" in result.stdout:
            print(f"✓ SUCCESS: Factored in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
            
            # Extract factors
            lines = result.stdout.split('\n')
            for line in lines:
                if "Prime 1:" in line or "Prime 2:" in line:
                    print(f"  {line.strip()}")
            
            return actual_digits, elapsed
        else:
            print(f"✗ FAILED after {elapsed:.1f} seconds")
            return actual_digits, None
            
    except subprocess.TimeoutExpired:
        print("✗ TIMEOUT after 20 minutes")
        return actual_digits, None
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return actual_digits, None

def main():
    print("🔍 Quick Semiprime Size Test")
    print("Target: ~600 seconds (10 minutes)")
    print("="*60)
    
    # Based on our data:
    # 31 digits: ~20 seconds
    # We expect ~50 digits for 600 seconds
    
    # Test a few sizes
    test_digits = [35, 40, 45, 48, 50]
    results = []
    
    for d in test_digits:
        actual_d, time_secs = test_size(d)
        
        if time_secs:
            results.append((actual_d, time_secs))
            
            # Stop if we're taking too long
            if time_secs > 900:  # 15 minutes
                print("\nStopping - taking too long")
                break
            
            # Also stop if we found the target
            if 570 < time_secs < 630:
                print(f"\n🎯 FOUND TARGET RANGE!")
                break
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    
    for d, t in results:
        print(f"  {d} digits: {t:.1f}s ({t/60:.1f} min)")
    
    # Extrapolate
    if len(results) >= 2:
        # Simple exponential fit
        import math
        d1, t1 = results[-2]
        d2, t2 = results[-1]
        
        # log(t) = a*d + b
        a = (math.log(t2) - math.log(t1)) / (d2 - d1)
        
        # Predict for 600s
        target_digits = d2 + (math.log(600) - math.log(t2)) / a
        print(f"\nPredicted optimal size: {target_digits:.0f} digits")

if __name__ == "__main__":
    main()