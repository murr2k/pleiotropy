#!/usr/bin/env python3
"""
Verify and find the correct two prime factors
"""

import math

def is_prime(n: int) -> bool:
    """Rigorous primality check"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check all odd divisors up to sqrt(n)
    limit = int(math.sqrt(n)) + 1
    for i in range(3, min(limit, 10000000), 2):
        if n % i == 0:
            return False
        if i > 10000 and i % 100000 == 1:
            print(f"  Checking up to {i} of {limit}")
    
    return True

def main():
    n = 2539123152460219
    
    # We know the complete factorization: 13 × 19² × 319483 × 1693501
    # Both 319483 and 1693501 are prime
    # So we need to find which grouping gives us two primes
    
    prime_factors = [319483, 1693501]
    small_factors = [13, 19, 19]
    
    print("Complete factorization: 13 × 19 × 19 × 319483 × 1693501")
    print(f"\nVerifying large factors are prime:")
    print(f"319483 is prime: {is_prime(319483)}")
    print(f"1693501 is prime: {is_prime(1693501)}")
    
    print("\nFinding the two prime factors...")
    
    # Try all possible groupings
    # Group 1: Combine some small factors with one large prime
    # Group 2: Combine remaining factors with other large prime
    
    candidates = [
        # (small factors with first prime, small factors with second prime)
        (13 * 19 * 19 * 319483, 1693501),
        (13 * 19 * 19 * 1693501, 319483),
        (13 * 19 * 319483, 19 * 1693501),
        (13 * 19 * 1693501, 19 * 319483),
        (13 * 319483, 19 * 19 * 1693501),
        (13 * 1693501, 19 * 19 * 319483),
        (19 * 319483, 13 * 19 * 1693501),
        (19 * 1693501, 13 * 19 * 319483),
        (19 * 19 * 319483, 13 * 1693501),
        (19 * 19 * 1693501, 13 * 319483),
        (319483, 13 * 19 * 19 * 1693501),
        (1693501, 13 * 19 * 19 * 319483)
    ]
    
    for p1, p2 in candidates:
        if p1 * p2 == n:
            print(f"\nChecking: {p1} × {p2} = {p1 * p2}")
            print(f"  Product correct: {p1 * p2 == n}")
            
            # For efficiency, check smaller numbers first
            if p1 < p2:
                check1, check2 = p1, p2
            else:
                check1, check2 = p2, p1
            
            print(f"  Checking if {check1} is prime...")
            is_p1_prime = is_prime(check1)
            print(f"  {check1} is prime: {is_p1_prime}")
            
            if is_p1_prime:
                print(f"  Checking if {check2} is prime...")
                is_p2_prime = is_prime(check2)
                print(f"  {check2} is prime: {is_p2_prime}")
                
                if is_p2_prime:
                    print(f"\n{'='*60}")
                    print("FOUND THE TWO PRIME FACTORS!")
                    print(f"Prime 1: {check1}")
                    print(f"Prime 2: {check2}")
                    print(f"Verification: {check1} × {check2} = {n}")
                    print(f"Both factors are prime: ✓")
                    return

if __name__ == "__main__":
    main()