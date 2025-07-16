#!/usr/bin/env python3
"""
Thoroughly verify the prime factorization of 2133019384970323
"""

import math

def is_prime_exhaustive(n):
    """Exhaustive primality check"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    print(f"Verifying primality of {n:,}...")
    limit = int(math.sqrt(n)) + 1
    print(f"  Checking divisors up to {limit:,}")
    
    # Check all odd divisors
    for i in range(3, limit, 2):
        if n % i == 0:
            print(f"  Found divisor: {i} (so {n} = {i} × {n//i})")
            return False
        
        if i % 1000000 == 1 and i > 1000000:
            print(f"  Progress: checked up to {i:,}")
    
    print(f"  No divisors found - {n:,} is prime!")
    return True

def main():
    n = 2133019384970323
    factor1 = 37094581
    factor2 = 57502183
    
    print(f"Verifying factorization of {n:,}")
    print("="*60)
    
    # Step 1: Verify multiplication
    product = factor1 * factor2
    print(f"\nStep 1: Verify multiplication")
    print(f"{factor1:,} × {factor2:,} = {product:,}")
    print(f"Target number: {n:,}")
    print(f"Match: {'✓' if product == n else '✗'}")
    
    # Step 2: Verify primality
    print(f"\nStep 2: Verify primality of factors\n")
    
    is_f1_prime = is_prime_exhaustive(factor1)
    print()
    is_f2_prime = is_prime_exhaustive(factor2)
    
    # Final conclusion
    print(f"\n{'='*60}")
    print("FINAL VERIFICATION:")
    
    if product == n and is_f1_prime and is_f2_prime:
        print(f"✓ CONFIRMED: {n:,} is a semiprime!")
        print(f"\nThe two prime factors are:")
        print(f"  Prime 1: {factor1:,}")
        print(f"  Prime 2: {factor2:,}")
        print(f"\nFactorization: {n:,} = {factor1:,} × {factor2:,}")
    else:
        print("✗ Verification failed!")
        if product != n:
            print(f"  Product mismatch: {product:,} ≠ {n:,}")
        if not is_f1_prime:
            print(f"  {factor1:,} is not prime")
        if not is_f2_prime:
            print(f"  {factor2:,} is not prime")

if __name__ == "__main__":
    main()