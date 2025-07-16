#!/usr/bin/env python3
"""
Complete factorization to find the two prime factors
"""

import math
from typing import List, Tuple

def factorize_completely(n: int) -> List[int]:
    """Find all prime factors of n"""
    factors = []
    
    # Check for factor of 2
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    
    # Check odd factors up to sqrt(n)
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        i += 2
    
    # If n is still greater than 1, it's prime
    if n > 1:
        factors.append(n)
    
    return factors

def main():
    n = 2539123152460219
    
    # Factor the two numbers we found
    factor1 = 32176519
    factor2 = 78912301
    
    print(f"Factoring {factor1}...")
    factors1 = factorize_completely(factor1)
    print(f"{factor1} = {' × '.join(map(str, factors1))}")
    
    print(f"\nFactoring {factor2}...")
    factors2 = factorize_completely(factor2)
    print(f"{factor2} = {' × '.join(map(str, factors2))}")
    
    # Combine all factors
    all_factors = factors1 + factors2
    print(f"\nComplete factorization of {n}:")
    print(f"{n} = {' × '.join(map(str, sorted(all_factors)))}")
    
    # Find unique prime factors
    unique_primes = sorted(set(all_factors))
    print(f"\nUnique prime factors: {unique_primes}")
    
    # If we're looking for exactly two primes, let's try a different approach
    print("\n" + "="*60)
    print("Direct factorization approach:")
    
    # Try to find two prime factors directly
    print(f"Factoring {n} directly...")
    
    # First check if it's divisible by small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    remaining = n
    found_factors = []
    
    for p in small_primes:
        if remaining % p == 0:
            found_factors.append(p)
            remaining //= p
            print(f"Found factor: {p}, remaining: {remaining}")
            
            # Check if remaining is prime
            if is_prime_simple(remaining):
                found_factors.append(remaining)
                print(f"Remaining {remaining} is prime!")
                break
    
    if len(found_factors) == 2:
        print(f"\nThe two prime factors are: {found_factors[0]} and {found_factors[1]}")
        print(f"Verification: {found_factors[0]} × {found_factors[1]} = {found_factors[0] * found_factors[1]}")

def is_prime_simple(n: int) -> bool:
    """Simple primality test"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check odd divisors up to sqrt(n)
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    
    return True

if __name__ == "__main__":
    main()