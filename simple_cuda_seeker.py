#!/usr/bin/env python3
"""
Simple CUDA Semiprime Seeker
Direct approach to find 10-minute factorization target
"""

import subprocess
import time
import random
import json
from datetime import datetime

def generate_semiprime_pair(digits):
    """Generate two primes that multiply to approximately 'digits' digits"""
    import gmpy2
    
    # Each prime should be about digits/2
    prime_bits = int(digits * 3.322 / 2)
    
    # Generate first prime
    p1 = gmpy2.mpz(2) ** prime_bits + random.randint(10**6, 10**7)
    p1 = gmpy2.next_prime(p1)
    
    # Generate second prime  
    p2 = gmpy2.mpz(2) ** prime_bits + random.randint(10**6, 10**7)
    p2 = gmpy2.next_prime(p2)
    
    # Ensure p1 < p2
    if p1 > p2:
        p1, p2 = p2, p1
    
    n = p1 * p2
    
    return str(n), str(p1), str(p2)

def test_factorization(number_str):
    """Test factorization using our Python script"""
    print(f"Testing {len(number_str)}-digit number...")
    
    start = time.time()
    
    # Call the factorization script
    result = subprocess.run(
        ['python3', 'factor_large_number.py'],
        input=number_str,
        capture_output=True,
        text=True,
        timeout=900  # 15 minute timeout
    )
    
    elapsed = time.time() - start
    
    if result.returncode == 0:
        # Check if it found exactly 2 prime factors
        output = result.stdout
        if "✓ This IS a semiprime!" in output:
            return True, elapsed
        else:
            print("Not a semiprime!")
            return False, elapsed
    else:
        print(f"Factorization failed: {result.stderr}")
        return False, elapsed

def main():
    print("🔍 Simple CUDA Semiprime Seeker")
    print("Target: 600 seconds (10 minutes)")
    print("="*60)
    
    # Test a few sizes based on our knowledge
    test_sizes = [48, 49, 50, 51, 52]
    results = []
    
    for digits in test_sizes:
        print(f"\nTesting {digits}-digit semiprime:")
        
        # Generate semiprime
        n, p1, p2 = generate_semiprime_pair(digits)
        actual_digits = len(n)
        print(f"Generated {actual_digits}-digit number")
        print(f"First 20 digits: {n[:20]}...")
        
        # Test factorization
        try:
            success, time_secs = test_factorization(n)
            
            if success:
                print(f"✓ Factored in {time_secs:.1f} seconds")
                
                results.append({
                    'digits': actual_digits,
                    'time': time_secs,
                    'number': n[:50] + '...' if len(n) > 50 else n,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Check if close to target
                if abs(time_secs - 600) < 30:
                    print(f"🎯 FOUND TARGET! {actual_digits} digits = {time_secs:.1f}s")
                    break
                elif time_secs > 900:
                    print("Taking too long, stopping search")
                    break
            else:
                print("✗ Factorization failed")
                
        except subprocess.TimeoutExpired:
            print("✗ Timeout!")
            results.append({
                'digits': actual_digits,
                'time': 'timeout',
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            print(f"Error: {e}")
    
    # Report results
    print("\n" + "="*60)
    print("RESULTS:")
    
    for r in results:
        if isinstance(r['time'], float):
            print(f"  {r['digits']} digits: {r['time']:.1f} seconds")
        else:
            print(f"  {r['digits']} digits: {r['time']}")
    
    # Find best
    valid_results = [r for r in results if isinstance(r['time'], float)]
    if valid_results:
        best = min(valid_results, key=lambda r: abs(r['time'] - 600))
        print(f"\nClosest to 10 minutes: {best['digits']} digits in {best['time']:.1f}s")
    
    # Save results
    with open('simple_cuda_seeker_results.json', 'w') as f:
        json.dump({
            'target_seconds': 600,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print("\n📄 Results saved to simple_cuda_seeker_results.json")

if __name__ == "__main__":
    main()