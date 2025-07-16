#!/usr/bin/env python3
"""
Verify the prime factorization of 4349182478874450510265070424251
"""

import math
import random

def miller_rabin_test(n, k=30):
    """
    Miller-Rabin primality test with k rounds
    More rounds = higher confidence
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # Run k rounds of testing
    for round in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        
        if x == 1 or x == n - 1:
            continue
        
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    
    return True

def verify_multiplication(p1, p2, expected):
    """Verify that p1 * p2 = expected"""
    product = p1 * p2
    return product == expected, product

def main():
    n = 4349182478874450510265070424251
    factor1 = 1184650163880919
    factor2 = 3671280021290429
    
    print(f"Verifying factorization of:")
    print(f"{n:,}")
    print(f"(31-digit number)")
    print("="*80)
    
    # Step 1: Verify multiplication
    print("\nStep 1: Verify multiplication")
    print(f"Factor 1: {factor1:,} ({len(str(factor1))} digits)")
    print(f"Factor 2: {factor2:,} ({len(str(factor2))} digits)")
    
    is_correct, product = verify_multiplication(factor1, factor2, n)
    print(f"\nProduct: {product:,}")
    print(f"Target:  {n:,}")
    print(f"Match: {'✓' if is_correct else '✗'}")
    
    if not is_correct:
        print(f"ERROR: Product doesn't match!")
        return
    
    # Step 2: Verify primality with high confidence
    print("\nStep 2: Verify primality (using Miller-Rabin with 30 rounds)")
    
    print(f"\nTesting {factor1:,}...")
    is_p1_prime = miller_rabin_test(factor1, 30)
    print(f"  Result: {'PRIME' if is_p1_prime else 'COMPOSITE'}")
    print(f"  Confidence: >99.999999999%")
    
    print(f"\nTesting {factor2:,}...")
    is_p2_prime = miller_rabin_test(factor2, 30)
    print(f"  Result: {'PRIME' if is_p2_prime else 'COMPOSITE'}")
    print(f"  Confidence: >99.999999999%")
    
    # Step 3: Additional verification - check a few small divisors
    print("\nStep 3: Quick divisibility check for small primes")
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    for f_name, f_value in [("Factor 1", factor1), ("Factor 2", factor2)]:
        print(f"\n{f_name}: ", end="")
        divisible = False
        for p in small_primes:
            if f_value % p == 0:
                print(f"divisible by {p}")
                divisible = True
                break
        if not divisible:
            print("not divisible by any small prime ✓")
    
    # Final conclusion
    print(f"\n{'='*80}")
    print("FINAL VERIFICATION:")
    
    if is_correct and is_p1_prime and is_p2_prime:
        print(f"\n✓ CONFIRMED: {n:,} is a semiprime!")
        print(f"\nThe two prime factors are:")
        print(f"  Prime 1: {factor1:,} (16 digits)")
        print(f"  Prime 2: {factor2:,} (16 digits)")
        print(f"\nFactorization: {n:,} = {factor1:,} × {factor2:,}")
    else:
        print("\n✗ Verification failed!")

if __name__ == "__main__":
    main()