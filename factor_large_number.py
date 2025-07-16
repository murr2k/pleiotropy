#!/usr/bin/env python3
"""
Find the prime factors of 4349182478874450510265070424251
This is a 31-digit number, so we'll need efficient algorithms
"""

import math
import time
import random

def gcd(a, b):
    """Greatest common divisor"""
    while b:
        a, b = b, a % b
    return a

def is_prime(n, k=20):
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

def pollard_rho_brent(n, max_iterations=100000000):
    """Improved Pollard's rho using Brent's algorithm"""
    if n % 2 == 0:
        return 2
    
    # Try multiple polynomials and starting values
    polynomials = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    for c in polynomials:
        print(f"  Trying polynomial x² + {c}")
        
        y = random.randint(2, min(n - 1, 10**9))
        m = random.randint(1, min(n - 1, 10**6))
        g = 1
        r = 1
        q = 1
        iterations = 0
        
        while g == 1 and iterations < max_iterations:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = (q * abs(x - y)) % n
                
                g = gcd(q, n)
                k += m
                iterations += m
                
                if iterations % 1000000 == 0:
                    print(f"    Progress: {iterations:,} iterations")
            
            r *= 2
            
            if g == n:
                # Backtrack
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = gcd(abs(x - ys), n)
        
        if g != n and g != 1:
            return g
    
    return None

def trial_division_limited(n, limit=10000000):
    """Trial division up to a limit"""
    factors = []
    
    # Check small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    for p in small_primes:
        while n % p == 0:
            factors.append(p)
            n //= p
            print(f"  Found small prime factor: {p}")
    
    # Continue with larger primes up to limit
    if n > 1:
        i = 101
        while i <= limit and i * i <= n:
            if n % i == 0:
                factors.append(i)
                n //= i
                print(f"  Found factor: {i}")
            else:
                i += 2
            
            if i % 1000000 == 1:
                print(f"  Checked up to {i:,}")
    
    return factors, n

def fermat_factorization(n, max_iterations=10000000):
    """Fermat's factorization method"""
    a = int(n**0.5)
    if a * a < n:
        a += 1
    
    print(f"  Starting Fermat's method from a={a:,}")
    
    for i in range(max_iterations):
        b_squared = a * a - n
        
        # Check if b_squared is a perfect square
        b = int(b_squared**0.5)
        if b * b == b_squared:
            return (a - b, a + b)
        
        a += 1
        
        if i % 1000000 == 0 and i > 0:
            print(f"    Progress: {i:,} iterations, a={a:,}")
    
    return None

def factorize_large_number(n):
    """Complete factorization of a very large number"""
    print(f"Factoring {n:,}")
    print(f"Number size: {len(str(n))} digits")
    print("="*80)
    
    factors = []
    remaining = n
    
    # Step 1: Check small factors
    print("\nStep 1: Checking small prime factors...")
    small_factors, remaining = trial_division_limited(remaining, 1000000)
    
    if small_factors:
        factors.extend(small_factors)
        print(f"  Remaining after small factors: {remaining:,}")
        print(f"  Remaining size: {len(str(remaining))} digits")
    else:
        print("  No small factors found")
    
    # Step 2: Advanced factorization
    if remaining > 1:
        print(f"\nStep 2: Factoring large remaining number...")
        
        # For very large numbers, check if it might be a semiprime
        sqrt_remaining = int(remaining**0.5)
        print(f"  Square root of remaining ≈ {sqrt_remaining:,}")
        
        # Try Pollard's rho first
        print("\n  Attempting Pollard's rho algorithm...")
        factor = pollard_rho_brent(remaining)
        
        if factor and factor != remaining and factor > 1:
            print(f"\n  Found factor: {factor:,}")
            factors.append(factor)
            quotient = remaining // factor
            print(f"  Quotient: {quotient:,}")
            
            # Check if quotient is prime
            print(f"\n  Checking if {quotient:,} is prime...")
            if is_prime(quotient):
                print(f"  {quotient:,} is prime!")
                factors.append(quotient)
            else:
                # Try to factor the quotient
                print(f"  Attempting to factor {quotient:,}...")
                subfactor = pollard_rho_brent(quotient)
                if subfactor and subfactor != quotient and subfactor > 1:
                    factors.append(subfactor)
                    factors.append(quotient // subfactor)
                else:
                    factors.append(quotient)
        else:
            # Try Fermat's method
            print("\n  Pollard's rho didn't find factors, trying Fermat's method...")
            print("  (This works well if the factors are close in size)")
            
            fermat_result = fermat_factorization(remaining, 5000000)
            
            if fermat_result:
                f1, f2 = fermat_result
                print(f"\n  Found factors using Fermat's method:")
                print(f"  Factor 1: {f1:,}")
                print(f"  Factor 2: {f2:,}")
                
                # Check primality
                for f in [f1, f2]:
                    print(f"\n  Checking if {f:,} is prime...")
                    if is_prime(f):
                        print(f"  {f:,} is prime!")
                        factors.append(f)
                    else:
                        print(f"  {f:,} is composite, needs further factorization")
                        # Try to factor it
                        subfactor = pollard_rho_brent(f)
                        if subfactor and subfactor != f and subfactor > 1:
                            factors.append(subfactor)
                            factors.append(f // subfactor)
                        else:
                            factors.append(f)
            else:
                print("  All factorization methods exhausted")
                print("  The number might be prime or have very large factors")
                factors.append(remaining)
    
    return sorted(factors)

def main():
    import sys
    
    # Check if number provided via stdin or argument
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    elif not sys.stdin.isatty():
        n = int(sys.stdin.read().strip())
    else:
        n = 4349182478874450510265070424251
    
    start_time = time.time()
    factors = factorize_large_number(n)
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("RESULTS:")
    
    # Verify factorization
    product = 1
    for f in factors:
        product *= f
    
    print(f"\nFactorization found:")
    if len(factors) <= 10:
        print(f"  {' × '.join(map(str, factors))}")
    else:
        print(f"  {len(factors)} factors found")
    
    print(f"\nVerification: Product = {product:,}")
    print(f"Original number:        {n:,}")
    print(f"Match: {'✓' if product == n else '✗'}")
    
    # Check primality
    print(f"\nPrimality verification:")
    all_prime = True
    for f in sorted(set(factors)):
        is_f_prime = is_prime(f)
        print(f"  {f:,} is prime: {is_f_prime}")
        if not is_f_prime:
            all_prime = False
    
    # Summary
    from collections import Counter
    factor_counts = Counter(factors)
    
    print(f"\nSummary:")
    print(f"  Time taken: {elapsed:.2f} seconds")
    print(f"  Total prime factors: {len(factors)}")
    print(f"  Distinct prime factors: {len(factor_counts)}")
    
    if len(factors) == 2 and all_prime:
        print(f"\n✓ This IS a semiprime!")
        print(f"\nThe two prime factors are:")
        print(f"  Prime 1: {factors[0]:,}")
        print(f"  Prime 2: {factors[1]:,}")

if __name__ == "__main__":
    main()