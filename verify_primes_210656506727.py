#!/usr/bin/env python3
"""
Verify the primality of the factors of 210656506727
"""

import math

def is_prime_thorough(n):
    """Thorough primality test"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    print(f"Checking primality of {n:,}...")
    
    # Check all odd numbers up to sqrt(n)
    limit = int(math.sqrt(n)) + 1
    print(f"  Checking divisors up to {limit:,}")
    
    for i in range(3, limit, 2):
        if n % i == 0:
            print(f"  Found divisor: {i} (so {n} = {i} × {n//i})")
            return False
        
        if i % 10000 == 1 and i > 10000:
            print(f"  Progress: checked up to {i:,}")
    
    print(f"  No divisors found - {n:,} is prime!")
    return True

def main():
    n = 210656506727
    factor1 = 387743
    factor2 = 543289
    
    print(f"Verifying factorization of {n:,}")
    print("="*60)
    
    # Verify multiplication
    product = factor1 * factor2
    print(f"\nStep 1: Verify multiplication")
    print(f"{factor1:,} × {factor2:,} = {product:,}")
    print(f"Matches target: {product == n} ✓\n")
    
    # Check primality
    print("Step 2: Verify primality of factors\n")
    
    is_f1_prime = is_prime_thorough(factor1)
    print()
    is_f2_prime = is_prime_thorough(factor2)
    
    print(f"\n{'='*60}")
    print("CONCLUSION:")
    
    if is_f1_prime and is_f2_prime:
        print(f"✓ Confirmed: {n:,} is a semiprime!")
        print(f"\nThe two prime factors are:")
        print(f"  Prime 1: {factor1:,}")
        print(f"  Prime 2: {factor2:,}")
    else:
        print("✗ At least one factor is not prime")

if __name__ == "__main__":
    main()