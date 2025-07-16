#!/usr/bin/env python3
"""
Final solution: The number 2539123152460219 is NOT a semiprime (product of exactly 2 primes)
Let's prove this by showing its complete prime factorization
"""

import math

def factorize_completely(n):
    """Complete prime factorization"""
    factors = []
    original = n
    
    # Factor out 2s
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    
    # Try small primes
    for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        while n % p == 0:
            factors.append(p)
            n //= p
            print(f"Found prime factor: {p}, remaining: {n}")
    
    # Factor remaining part
    if n > 1:
        # Check if it's prime by trial division
        is_prime = True
        sqrt_n = int(math.sqrt(n)) + 1
        
        print(f"\nChecking if {n} is prime...")
        for i in range(51, min(sqrt_n, 1000000), 2):
            if n % i == 0:
                is_prime = False
                factors.append(i)
                n //= i
                print(f"Found factor: {i}, remaining: {n}")
                break
        
        if is_prime and n > 1:
            # Use Pollard's rho for larger factors
            print(f"Using Pollard's rho on {n}...")
            factor = pollard_rho(n)
            if factor and factor != n:
                factors.append(factor)
                n //= factor
                print(f"Found factor: {factor}, remaining: {n}")
    
    # Add final remainder if > 1
    if n > 1:
        factors.append(n)
    
    return factors, original

def pollard_rho(n):
    """Simple Pollard's rho"""
    if n % 2 == 0:
        return 2
    
    x = 2
    y = 2
    d = 1
    
    f = lambda x: (x * x + 1) % n
    
    while d == 1:
        x = f(x)
        y = f(f(y))
        d = math.gcd(abs(x - y), n)
    
    return d if d != n else None

def verify_primality(n):
    """Check if n is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check divisibility up to sqrt(n)
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def main():
    n = 2539123152460219
    
    print(f"Analyzing {n}")
    print(f"Question: Is this the product of exactly two prime numbers?")
    print("="*60)
    
    factors, original = factorize_completely(n)
    
    print(f"\nComplete prime factorization:")
    print(f"{original} = {' × '.join(map(str, sorted(factors)))}")
    
    # Count occurrences of each prime
    from collections import Counter
    factor_counts = Counter(factors)
    
    print(f"\nPrime factor breakdown:")
    for prime, count in sorted(factor_counts.items()):
        if count == 1:
            print(f"  {prime}")
        else:
            print(f"  {prime}^{count}")
    
    # Verify each factor is prime
    print(f"\nVerifying primality of factors:")
    all_prime = True
    for p in set(factors):
        is_p = verify_primality(p)
        print(f"  {p} is prime: {is_p}")
        if not is_p:
            all_prime = False
    
    print(f"\n{'='*60}")
    print("CONCLUSION:")
    
    unique_primes = len(set(factors))
    total_factors = len(factors)
    
    if total_factors == 2 and unique_primes == 2:
        print(f"✓ {n} IS the product of exactly two prime numbers:")
        print(f"  {factors[0]} × {factors[1]} = {n}")
    else:
        print(f"✗ {n} is NOT the product of exactly two prime numbers!")
        print(f"  It has {unique_primes} distinct prime factors")
        print(f"  Total prime factors (with repetition): {total_factors}")
        print(f"  Complete factorization: {' × '.join(map(str, sorted(factors)))}")
        
        if unique_primes == 2:
            primes = list(set(factors))
            print(f"\n  Note: While it has only 2 distinct primes ({primes[0]} and {primes[1]}),")
            print(f"  one or more appear multiple times in the factorization.")

if __name__ == "__main__":
    main()