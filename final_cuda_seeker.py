#!/usr/bin/env python3
"""
Final CUDA Semiprime Seeker
Target: Find semiprime that takes ~600 seconds on GTX 2070
Based on calibration: expecting ~42 digits
"""

import subprocess
import time
import random
import json
from datetime import datetime
import gmpy2

TARGET_TIME = 600  # 10 minutes
TOLERANCE = 30     # ±30 seconds

def generate_semiprime(digits):
    """Generate a semiprime with approximately 'digits' digits"""
    # Each prime should be about digits/2
    prime_bits = int(digits * 3.322 / 2)
    
    # Add some randomness
    bits1 = prime_bits + random.randint(-2, 2)
    bits2 = prime_bits + random.randint(-2, 2)
    
    # Generate primes
    p1 = gmpy2.mpz(2) ** bits1 + random.randint(10**5, 10**7)
    p1 = gmpy2.next_prime(p1)
    
    p2 = gmpy2.mpz(2) ** bits2 + random.randint(10**5, 10**7)
    p2 = gmpy2.next_prime(p2)
    
    n = p1 * p2
    
    return str(n), str(p1), str(p2)

def test_factorization(n_str):
    """Factor using the CUDA implementation"""
    digits = len(n_str)
    print(f"\nTesting {digits}-digit semiprime")
    print(f"First 40 chars: {n_str[:40]}...")
    
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
            # Extract factors
            lines = result.stdout.split('\n')
            factor1, factor2 = None, None
            
            for line in lines:
                if "Prime 1:" in line:
                    factor1 = line.split(":")[1].strip()
                elif "Prime 2:" in line:
                    factor2 = line.split(":")[1].strip()
            
            print(f"✓ SUCCESS in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
            
            if abs(elapsed - TARGET_TIME) <= TOLERANCE:
                print("🎯 TARGET HIT! Within 10 minutes ± 30 seconds")
            
            return True, elapsed, factor1, factor2
        else:
            print(f"✗ FAILED after {elapsed:.1f} seconds")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}")
            return False, elapsed, None, None
            
    except subprocess.TimeoutExpired:
        print("✗ TIMEOUT after 20 minutes")
        return False, 1200, None, None
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False, 0, None, None

def main():
    print("🔍 Final CUDA Semiprime Seeker - GTX 2070")
    print(f"Target: {TARGET_TIME} seconds (10 minutes) ± {TOLERANCE} seconds")
    print("Based on calibration: expecting ~42 digits")
    print("="*70)
    
    # Test range based on calibration
    test_range = [40, 41, 42, 43, 44]
    results = []
    best_result = None
    best_diff = float('inf')
    
    for target_digits in test_range:
        # Try 2 samples per size
        for attempt in range(2):
            n, p1, p2 = generate_semiprime(target_digits)
            actual_digits = len(n)
            
            success, time_secs, f1, f2 = test_factorization(n)
            
            if success:
                result = {
                    'digits': actual_digits,
                    'time': time_secs,
                    'number': n[:50] + '...' if len(n) > 50 else n,
                    'factor1': f1[:30] + '...' if f1 and len(f1) > 30 else f1,
                    'factor2': f2[:30] + '...' if f2 and len(f2) > 30 else f2,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
                
                # Check if this is the best so far
                diff = abs(time_secs - TARGET_TIME)
                if diff < best_diff:
                    best_diff = diff
                    best_result = result
                
                # Stop if we hit the target
                if diff <= TOLERANCE:
                    print("\n✅ Found optimal semiprime!")
                    break
                
                # Also stop if taking too long
                if time_secs > 900:  # 15 minutes
                    print("\nStopping - factorization taking too long")
                    break
        
        # Stop outer loop if we found target or taking too long
        if best_diff <= TOLERANCE or (results and results[-1]['time'] > 900):
            break
    
    # Report results
    print("\n" + "="*70)
    print("FINAL RESULTS:\n")
    
    for r in results:
        status = ""
        if abs(r['time'] - TARGET_TIME) <= TOLERANCE:
            status = " ← TARGET!"
        print(f"{r['digits']} digits: {r['time']:.1f}s ({r['time']/60:.1f} min){status}")
    
    if best_result:
        print(f"\n🏆 BEST RESULT:")
        print(f"Digits: {best_result['digits']}")
        print(f"Time: {best_result['time']:.1f} seconds ({best_result['time']/60:.1f} minutes)")
        print(f"Deviation from target: {best_result['time'] - TARGET_TIME:+.1f} seconds")
        
        # Save results
        output = {
            'gpu': 'GTX 2070',
            'target_seconds': TARGET_TIME,
            'tolerance_seconds': TOLERANCE,
            'best_result': best_result,
            'all_results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('cuda_seeker_final_results.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print("\n📄 Results saved to cuda_seeker_final_results.json")
    
    # Show scaling
    if len(results) >= 2:
        print("\n📊 Observed Scaling:")
        import math
        
        # Sort by digits
        sorted_results = sorted(results, key=lambda r: r['digits'])
        
        for i in range(len(sorted_results) - 1):
            r1 = sorted_results[i]
            r2 = sorted_results[i + 1]
            
            if r1['digits'] != r2['digits']:
                ratio = r2['time'] / r1['time']
                print(f"  {r1['digits']}→{r2['digits']} digits: {ratio:.2f}× slower")

if __name__ == "__main__":
    main()