#!/usr/bin/env python3
"""
Find the two prime factors of 2539123152460219
We need to find p and q such that p × q = 2539123152460219
"""

import math
import time

def is_prime(n: int) -> bool:
    """Check if n is prime using trial division"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check odd divisors up to sqrt(n)
    for i in range(3, min(int(math.sqrt(n)) + 1, 1000000), 2):
        if n % i == 0:
            return False
    
    return True

def pollard_rho_brent(n: int) -> int:
    """Improved Pollard's rho using Brent's algorithm"""
    if n % 2 == 0:
        return 2
    
    # Multiple parallel attempts with different parameters
    for c in [1, 2, 3, 5, 7]:
        x = 2
        y = 2
        d = 1
        
        print(f"Trying Pollard-Brent with c={c}...")
        
        while d == 1:
            # Brent's improvement
            for _ in range(100):
                x = (x * x + c) % n
                d = math.gcd(abs(x - y), n)
                if d > 1:
                    break
            
            if d == 1:
                y = x
        
        if d != n:
            return d
    
    return n

def main():
    n = 2539123152460219
    print(f"Finding the two prime factors of {n}")
    print(f"Number size: {len(str(n))} digits")
    
    # We know from the complete factorization that n = 13 × 19 × 19 × 319483 × 1693501
    # So the two prime factors should be formed by grouping these
    
    # Let's verify the complete factorization first
    factors = []
    temp = n
    
    # Factor out 13
    if temp % 13 == 0:
        factors.append(13)
        temp //= 13
        print(f"Found factor 13, remaining: {temp}")
    
    # Factor out 19s
    while temp % 19 == 0:
        factors.append(19)
        temp //= 19
        print(f"Found factor 19, remaining: {temp}")
    
    # Now we need to find the factors of what remains
    print(f"\nFinding factors of {temp}...")
    
    # Try Pollard's rho on the remaining number
    factor1 = pollard_rho_brent(temp)
    
    if factor1 != temp:
        factor2 = temp // factor1
        print(f"\nFound factorization of {temp}:")
        print(f"{temp} = {factor1} × {factor2}")
        
        # Check primality
        print(f"\nChecking primality:")
        print(f"{factor1} is prime: {is_prime(factor1)}")
        print(f"{factor2} is prime: {is_prime(factor2)}")
        
        # Now we can construct the two prime factors of the original number
        print(f"\n{'='*60}")
        print("SOLUTION:")
        
        # The two primes must be combinations of our factors
        # We have: 13, 19, 19, 319483, 1693501
        # Two possible groupings for two primes:
        # Option 1: (13 × 19 × 319483) × (19 × 1693501)
        # Option 2: (13 × 19 × 19) × (319483 × 1693501)
        # Option 3: Other combinations...
        
        prime1_candidates = [
            13 * 19 * 319483,
            13 * 19 * 19,
            13 * 319483,
            13 * 1693501,
            19 * 319483,
            19 * 1693501,
            19 * 19 * 319483,
            19 * 19 * 1693501
        ]
        
        for p1 in prime1_candidates:
            if n % p1 == 0:
                p2 = n // p1
                if is_prime(p1) and is_prime(p2):
                    print(f"\nFound the two prime factors!")
                    print(f"Prime 1: {p1}")
                    print(f"Prime 2: {p2}")
                    print(f"Verification: {p1} × {p2} = {p1 * p2}")
                    print(f"Correct: {p1 * p2 == n}")
                    return
        
        # If no combination works, try direct search
        print("\nTrying direct factor search...")
        
        # We know the factors include 319483 and 1693501
        # Let's check if one of these multiplied by a small factor gives a prime
        candidates = [
            (13 * 1693501, 19 * 19 * 319483),
            (19 * 1693501, 13 * 19 * 319483),
            (13 * 19 * 1693501, 19 * 319483)
        ]
        
        for p1, p2 in candidates:
            if p1 * p2 == n:
                print(f"\nChecking: {p1} × {p2}")
                if is_prime(p1) and is_prime(p2):
                    print(f"Both are prime!")
                    print(f"\nFINAL ANSWER:")
                    print(f"Prime 1: {p1}")
                    print(f"Prime 2: {p2}")
                    return

if __name__ == "__main__":
    main()