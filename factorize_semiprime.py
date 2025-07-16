#!/usr/bin/env python3
"""
Find the two prime factors of 2539123152460219 using various factorization methods
"""

import math
import time
from typing import Tuple, Optional

def is_prime(n: int) -> bool:
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
    
    # Witnesses for deterministic test
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    for a in witnesses:
        if a >= n:
            continue
        
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

def fermat_factorization(n: int) -> Optional[Tuple[int, int]]:
    """Fermat's factorization method - efficient for factors close in size"""
    print(f"\nTrying Fermat's method on {n}...")
    
    # Start with ceiling of sqrt(n)
    a = math.isqrt(n)
    if a * a < n:
        a += 1
    
    start_time = time.time()
    iterations = 0
    
    while iterations < 10000000:  # Limit iterations
        b_squared = a * a - n
        b = math.isqrt(b_squared)
        
        if b * b == b_squared:
            # Found factorization
            factor1 = a - b
            factor2 = a + b
            elapsed = time.time() - start_time
            print(f"Success! Found factors after {iterations} iterations in {elapsed:.3f} seconds")
            return (factor1, factor2)
        
        a += 1
        iterations += 1
        
        if iterations % 100000 == 0:
            print(f"  Progress: {iterations} iterations, a = {a}")
    
    print(f"Fermat's method failed after {iterations} iterations")
    return None

def pollard_rho(n: int, max_iterations: int = 10000000) -> Optional[int]:
    """Pollard's rho algorithm for finding a factor"""
    print(f"\nTrying Pollard's rho on {n}...")
    
    def f(x):
        return (x * x + 1) % n
    
    start_time = time.time()
    
    # Try multiple starting values
    for start in [2, 3, 5, 7, 11]:
        x = start
        y = start
        d = 1
        iterations = 0
        
        while d == 1 and iterations < max_iterations:
            x = f(x)
            y = f(f(y))
            d = math.gcd(abs(x - y), n)
            iterations += 1
            
            if iterations % 100000 == 0:
                print(f"  Start={start}, iterations={iterations}")
        
        if d != 1 and d != n:
            elapsed = time.time() - start_time
            print(f"Success! Found factor {d} in {elapsed:.3f} seconds")
            return d
    
    print("Pollard's rho failed")
    return None

def trial_division_smart(n: int, limit: int = 10000000) -> Optional[int]:
    """Smart trial division checking small primes first"""
    print(f"\nTrying smart trial division up to {limit}...")
    
    start_time = time.time()
    
    # Check small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    for p in small_primes:
        if n % p == 0:
            print(f"Found small prime factor: {p}")
            return p
    
    # Check larger primes
    checked = 0
    for candidate in range(51, min(limit, math.isqrt(n) + 1), 2):
        if n % candidate == 0:
            elapsed = time.time() - start_time
            print(f"Found factor {candidate} in {elapsed:.3f} seconds")
            return candidate
        
        checked += 1
        if checked % 100000 == 0:
            print(f"  Checked up to {candidate}")
    
    return None

def main():
    n = 2539123152460219
    print(f"Factoring {n}")
    print(f"Number of digits: {len(str(n))}")
    print(f"Approximate size: {n:.2e}")
    
    # First, let's check if it's even close to a perfect square
    sqrt_n = math.isqrt(n)
    print(f"\nSquare root ≈ {sqrt_n}")
    print(f"If factors are close: they're around {sqrt_n:,}")
    
    # Try Fermat's method (best for factors close in size)
    result = fermat_factorization(n)
    
    if result:
        factor1, factor2 = result
        print(f"\nFactorization: {n} = {factor1} × {factor2}")
        
        # Verify
        if factor1 * factor2 == n:
            print("✓ Verification: Product is correct")
        else:
            print("✗ Verification: Product is incorrect!")
        
        # Check primality
        print(f"\nChecking primality:")
        print(f"  {factor1} is prime: {is_prime(factor1)}")
        print(f"  {factor2} is prime: {is_prime(factor2)}")
        
        # Show the size difference
        ratio = factor2 / factor1 if factor1 < factor2 else factor1 / factor2
        print(f"\nFactor ratio: {ratio:.6f}")
        print(f"Factors differ by: {abs(factor2 - factor1):,}")
        
    else:
        # Try Pollard's rho
        factor = pollard_rho(n)
        if factor:
            factor2 = n // factor
            print(f"\nFactorization: {n} = {factor} × {factor2}")
            print(f"Checking primality:")
            print(f"  {factor} is prime: {is_prime(factor)}")
            print(f"  {factor2} is prime: {is_prime(factor2)}")
        else:
            # Last resort - trial division
            factor = trial_division_smart(n)
            if factor:
                factor2 = n // factor
                print(f"\nFactorization: {n} = {factor} × {factor2}")

if __name__ == "__main__":
    main()