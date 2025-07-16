#!/usr/bin/env python3
"""
Direct solution for finding two prime factors of 2539123152460219
Using optimized Pollard's rho and ECM-like methods
"""

import math
import random
import time

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def is_prime(n, k=5):
    """Miller-Rabin primality test"""
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
    
    # Test with k random witnesses
    for _ in range(k):
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

def pollard_rho_optimized(n):
    """Optimized Pollard's rho with multiple polynomials"""
    if n % 2 == 0:
        return 2
    
    # Try different polynomials
    for c in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]:
        print(f"  Trying polynomial x² + {c}")
        
        x = random.randint(2, n - 1)
        y = x
        d = 1
        
        # Use Floyd's cycle detection with Brent's improvement
        saved = y
        for i in range(1, 100000):
            x = (pow(x, 2, n) + c) % n
            d = gcd(abs(x - y), n)
            
            if d > 1 and d < n:
                return d
            
            # Brent's improvement: update y less frequently
            if i & (i - 1) == 0:  # i is a power of 2
                y = x
                saved = y
        
        # Try batch GCD for efficiency
        product = 1
        for _ in range(100):
            x = (pow(x, 2, n) + c) % n
            product = (product * abs(x - saved)) % n
            
            if _ % 10 == 0:
                d = gcd(product, n)
                if d > 1 and d < n:
                    return d
    
    return None

def main():
    n = 2539123152460219
    print(f"Finding the two prime factors of {n}")
    print(f"Target: Find p, q such that p × q = {n} where both p and q are prime\n")
    
    start_time = time.time()
    
    # First, let's try the optimized Pollard's rho
    print("Running optimized Pollard's rho algorithm...")
    factor = pollard_rho_optimized(n)
    
    if factor and factor != n:
        factor2 = n // factor
        
        print(f"\nFound potential factorization:")
        print(f"{n} = {factor} × {factor2}")
        
        # Verify the factorization
        if factor * factor2 == n:
            print("\n✓ Factorization verified!")
            
            # Check primality of both factors
            print("\nChecking primality of factors...")
            print(f"  Checking {factor}...")
            p1_prime = is_prime(factor, k=10)
            print(f"  {factor} is prime: {p1_prime}")
            
            print(f"  Checking {factor2}...")
            p2_prime = is_prime(factor2, k=10)
            print(f"  {factor2} is prime: {p2_prime}")
            
            if p1_prime and p2_prime:
                elapsed = time.time() - start_time
                print(f"\n{'='*60}")
                print("SUCCESS! Found the two prime factors:")
                print(f"Prime 1: {factor:,}")
                print(f"Prime 2: {factor2:,}")
                print(f"Time taken: {elapsed:.3f} seconds")
                print(f"\nVerification: {factor} × {factor2} = {factor * factor2}")
            else:
                # One or both factors are composite, need to factor further
                print("\nFactors are not both prime. Attempting complete factorization...")
                
                # If one is prime and the other isn't, factor the composite one
                if p1_prime and not p2_prime:
                    print(f"\n{factor} is prime, factoring {factor2}...")
                    subfactor = pollard_rho_optimized(factor2)
                    if subfactor:
                        print(f"Found subfactor: {subfactor}")
                        other_subfactor = factor2 // subfactor
                        print(f"Complete factorization: {factor} × {subfactor} × {other_subfactor}")
                elif not p1_prime and p2_prime:
                    print(f"\n{factor2} is prime, factoring {factor}...")
                    subfactor = pollard_rho_optimized(factor)
                    if subfactor:
                        print(f"Found subfactor: {subfactor}")
                        other_subfactor = factor // subfactor
                        print(f"Complete factorization: {subfactor} × {other_subfactor} × {factor2}")
                else:
                    print("\nBoth factors are composite. This suggests the number has more than 2 prime factors.")
    else:
        print("\nPollard's rho failed. The number might be prime or require more advanced methods.")

if __name__ == "__main__":
    main()