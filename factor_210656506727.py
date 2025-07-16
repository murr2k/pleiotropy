#!/usr/bin/env python3
"""
Find the prime factors of 210656506727
"""

import math
import time

def is_prime(n):
    """Check if n is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def trial_division(n):
    """Find factors using trial division"""
    factors = []
    
    # Check for factor of 2
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    
    # Check odd factors
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        i += 2
        
        # Progress indicator for large numbers
        if i % 1000000 == 1 and i > 1000000:
            print(f"  Checking up to {i:,}, remaining: {n:,}")
    
    # If n is still greater than 1, it's prime
    if n > 1:
        factors.append(n)
    
    return factors

def pollard_rho(n):
    """Pollard's rho algorithm for finding a factor"""
    if n % 2 == 0:
        return 2
    
    x = 2
    y = 2
    d = 1
    
    # f(x) = x^2 + 1 mod n
    f = lambda x: (x * x + 1) % n
    
    while d == 1:
        x = f(x)
        y = f(f(y))
        d = math.gcd(abs(x - y), n)
    
    return d if d != n else None

def factorize(n):
    """Complete factorization using multiple methods"""
    print(f"Factoring {n:,}")
    print("="*60)
    
    # First try small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    factors = []
    remaining = n
    
    print("\nChecking small prime factors...")
    for p in small_primes:
        while remaining % p == 0:
            factors.append(p)
            remaining //= p
            print(f"  Found factor: {p}, remaining: {remaining:,}")
    
    # If we still have a large number, try Pollard's rho
    if remaining > 1 and remaining > 1000000:
        print(f"\nTrying Pollard's rho on {remaining:,}...")
        factor = pollard_rho(remaining)
        
        if factor and factor != remaining:
            print(f"  Found factor: {factor:,}")
            factors.append(factor)
            remaining //= factor
            
            # Check if remaining is prime
            if remaining > 1:
                print(f"  Checking if {remaining:,} is prime...")
                if is_prime(remaining):
                    print(f"  {remaining:,} is prime!")
                    factors.append(remaining)
                else:
                    # Factor the remaining part
                    print(f"  Factoring {remaining:,}...")
                    more_factors = trial_division(remaining)
                    factors.extend(more_factors)
        else:
            # Use trial division
            print(f"  Pollard's rho didn't find factors, using trial division...")
            more_factors = trial_division(remaining)
            factors.extend(more_factors)
    elif remaining > 1:
        factors.append(remaining)
    
    return sorted(factors)

def main():
    n = 210656506727
    
    start_time = time.time()
    factors = factorize(n)
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"\nComplete factorization of {n:,}:")
    
    # Count occurrences of each prime
    from collections import Counter
    factor_counts = Counter(factors)
    
    # Display in exponential form
    factorization_parts = []
    for prime, count in sorted(factor_counts.items()):
        if count == 1:
            factorization_parts.append(str(prime))
        else:
            factorization_parts.append(f"{prime}^{count}")
    
    print(f"{n:,} = {' × '.join(factorization_parts)}")
    
    # Verify
    product = 1
    for f in factors:
        product *= f
    
    print(f"\nVerification: {' × '.join(map(str, factors))} = {product:,}")
    print(f"Correct: {product == n} ✓" if product == n else f"Error: {product} ≠ {n}")
    
    # Summary
    print(f"\nSummary:")
    print(f"  Number of prime factors: {len(factors)}")
    print(f"  Number of distinct primes: {len(factor_counts)}")
    print(f"  Prime factors: {sorted(set(factors))}")
    print(f"  Time taken: {elapsed:.3f} seconds")
    
    # Check if it's a semiprime
    if len(factors) == 2 and len(factor_counts) == 2:
        print(f"\n✓ This IS a semiprime (product of exactly two primes):")
        print(f"  {factors[0]:,} × {factors[1]:,} = {n:,}")

if __name__ == "__main__":
    main()