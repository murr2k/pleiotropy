#!/usr/bin/env python3
"""
Find the prime factors of 2133019384970323
"""

import math
import time
import random

def gcd(a, b):
    """Greatest common divisor"""
    while b:
        a, b = b, a % b
    return a

def is_prime(n, k=10):
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

def pollard_rho_brent(n):
    """Improved Pollard's rho using Brent's algorithm"""
    if n % 2 == 0:
        return 2
    
    # Try multiple starting values and polynomials
    for c in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]:
        print(f"  Trying Pollard-Brent with c={c}...")
        
        y = random.randint(2, n - 1)
        m = random.randint(1, n - 1)
        g = 1
        r = 1
        q = 1
        
        while g == 1:
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
            
            r *= 2
            
            if g == n:
                # Backtrack to find the exact factor
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = gcd(abs(x - ys), n)
        
        if g != n:
            return g
    
    return None

def trial_division_limited(n, limit=1000000):
    """Trial division up to a limit"""
    factors = []
    
    # Check small primes first
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    for p in small_primes:
        while n % p == 0:
            factors.append(p)
            n //= p
    
    # Check up to limit
    if n > 1:
        i = 101
        while i <= limit and i * i <= n:
            while n % i == 0:
                factors.append(i)
                n //= i
            i += 2
    
    return factors, n

def fermat_factorization(n, max_iterations=1000000):
    """Fermat's factorization method"""
    a = math.isqrt(n)
    if a * a < n:
        a += 1
    
    print(f"  Starting Fermat's method from a={a:,}")
    
    for i in range(max_iterations):
        b_squared = a * a - n
        b = math.isqrt(b_squared)
        
        if b * b == b_squared:
            return (a - b, a + b)
        
        a += 1
        
        if i % 100000 == 0 and i > 0:
            print(f"    Progress: {i:,} iterations, a={a:,}")
    
    return None

def factorize_large_number(n):
    """Complete factorization of a large number"""
    print(f"Factoring {n:,}")
    print(f"Number size: {len(str(n))} digits")
    print("="*60)
    
    factors = []
    remaining = n
    
    # Step 1: Trial division for small factors
    print("\nStep 1: Checking small prime factors...")
    small_factors, remaining = trial_division_limited(remaining, 100000)
    
    if small_factors:
        factors.extend(small_factors)
        for f in small_factors:
            print(f"  Found factor: {f}")
        print(f"  Remaining: {remaining:,}")
    else:
        print("  No small factors found")
    
    # Step 2: If remaining is still large, try advanced methods
    if remaining > 1:
        print(f"\nStep 2: Factoring {remaining:,} using advanced methods...")
        
        # Check if it might be prime first
        if remaining < 10**12:  # Can check primality for smaller numbers
            print("  Checking if remaining number is prime...")
            if is_prime(remaining):
                print(f"  {remaining:,} is prime!")
                factors.append(remaining)
                return factors
        
        # Try Pollard's rho
        print("  Trying Pollard's rho algorithm...")
        factor = pollard_rho_brent(remaining)
        
        if factor and factor != remaining:
            print(f"  Found factor: {factor:,}")
            factors.append(factor)
            remaining //= factor
            
            # Check if the quotient is prime
            print(f"  Checking if {remaining:,} is prime...")
            if is_prime(remaining):
                print(f"  {remaining:,} is prime!")
                factors.append(remaining)
            else:
                # Try to factor the quotient
                print(f"  Attempting to factor {remaining:,}...")
                subfactor = pollard_rho_brent(remaining)
                if subfactor and subfactor != remaining:
                    factors.append(subfactor)
                    factors.append(remaining // subfactor)
                else:
                    factors.append(remaining)
        else:
            # Try Fermat's method (good for numbers that are products of two close primes)
            print("  Pollard's rho failed, trying Fermat's method...")
            fermat_result = fermat_factorization(remaining, 500000)
            
            if fermat_result:
                f1, f2 = fermat_result
                print(f"  Found factors: {f1:,} and {f2:,}")
                
                # Check if each is prime
                for f in [f1, f2]:
                    if is_prime(f):
                        factors.append(f)
                    else:
                        # Try to factor further
                        subfactor = pollard_rho_brent(f)
                        if subfactor and subfactor != f:
                            factors.append(subfactor)
                            factors.append(f // subfactor)
                        else:
                            factors.append(f)
            else:
                print("  All methods failed, number might be prime")
                factors.append(remaining)
    
    return sorted(factors)

def main():
    n = 2133019384970323
    
    start_time = time.time()
    factors = factorize_large_number(n)
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("RESULTS:")
    
    # Verify factorization
    product = 1
    for f in factors:
        product *= f
    
    print(f"\nFactorization: {' × '.join(map(str, factors))}")
    print(f"Verification: {product:,} = {n:,}")
    print(f"Correct: {'✓' if product == n else '✗'}")
    
    # Check primality of factors
    print(f"\nPrimality check of factors:")
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
    print(f"  Total factors: {len(factors)}")
    print(f"  Distinct prime factors: {len(factor_counts)}")
    print(f"  Time taken: {elapsed:.3f} seconds")
    
    if len(factors) == 2 and all_prime:
        print(f"\n✓ This IS a semiprime!")
        print(f"  {factors[0]:,} × {factors[1]:,} = {n:,}")
    elif all_prime:
        print(f"\n✓ Complete prime factorization found")
        factorization_str = ' × '.join(f"{p}^{c}" if c > 1 else str(p) for p, c in sorted(factor_counts.items()))
        print(f"  {n:,} = {factorization_str}")
    else:
        print(f"\n⚠ Some factors may not be prime, further factorization needed")

if __name__ == "__main__":
    main()