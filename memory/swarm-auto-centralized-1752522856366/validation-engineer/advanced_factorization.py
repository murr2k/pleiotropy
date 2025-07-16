#!/usr/bin/env python3
"""
Advanced Factorization for 2539123152460219
Using multiple sophisticated algorithms
"""

import time
import math
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

TARGET_NUMBER = 2539123152460219

def is_prime(n: int) -> bool:
    """Miller-Rabin primality test"""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    
    # Write n-1 as 2^r * d
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    
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

def find_all_factors(n: int) -> List[int]:
    """Find all factors of n"""
    factors = []
    sqrt_n = int(math.sqrt(n)) + 1
    
    for i in range(2, min(sqrt_n, 1000000)):  # Limit search to avoid too long runtime
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)
    
    return sorted(factors)

def pollard_rho_brent(n: int) -> Optional[int]:
    """Brent's improvement to Pollard's rho algorithm"""
    if n % 2 == 0:
        return 2
    
    # Try multiple starting values and c parameters
    for x0 in [2, 3, 5]:
        for c in [1, 2, 3, 5, 7, 11, 13, 17]:
            y, r, q = x0, 1, 1
            g, x, ys = 1, x0, x0
            
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(100, r - k)):
                        y = (y * y + c) % n
                        q = (q * abs(x - y)) % n
                    
                    g = math.gcd(q, n)
                    k += 100
                
                r *= 2
                
                # Prevent infinite loop
                if r > 100000:
                    break
            
            if g != n and g != 1:
                # Verify it's a proper factor
                if n % g == 0:
                    return g
    
    return None

def fermat_factorization(n: int) -> Optional[Tuple[int, int]]:
    """Fermat's factorization method - works well when factors are close"""
    a = int(math.ceil(math.sqrt(n)))
    b2 = a * a - n
    
    max_iterations = min(n, 10000000)  # Limit iterations
    
    for _ in range(max_iterations):
        b = int(math.sqrt(b2))
        if b * b == b2:
            factor1 = a - b
            factor2 = a + b
            if factor1 > 1 and factor2 > 1 and factor1 * factor2 == n:
                return (factor1, factor2)
        
        a += 1
        b2 = a * a - n
        
        if a % 100000 == 0:
            print(f"  Fermat method: a = {a}")
    
    return None

def quadratic_sieve_simple(n: int, B: int = 10000) -> Optional[int]:
    """Simplified quadratic sieve - find a factor"""
    # This is a very simplified version
    sqrt_n = int(math.sqrt(n))
    
    # Try to find smooth numbers
    for x in range(sqrt_n, sqrt_n + B):
        y2 = x * x - n
        if y2 > 0:
            y = int(math.sqrt(y2))
            if y * y == y2:
                # Found a perfect square
                factor = math.gcd(x - y, n)
                if factor > 1 and factor < n:
                    return factor
    
    return None

def ecm_simple(n: int, curves: int = 100) -> Optional[int]:
    """Simplified Elliptic Curve Method"""
    for _ in range(curves):
        # Random curve parameters
        a = random.randint(1, n - 1)
        x = random.randint(1, n - 1)
        y = random.randint(1, n - 1)
        
        # Try to find a factor
        for k in range(2, min(1000, n)):
            try:
                # Simplified point multiplication
                d = math.gcd(k * x - y, n)
                if d > 1 and d < n:
                    return d
            except:
                pass
    
    return None

def check_perfect_power(n: int) -> Optional[Tuple[int, int]]:
    """Check if n is a perfect power"""
    for exp in range(2, 65):  # Up to 64-bit
        root = int(n ** (1.0 / exp))
        
        # Check around the floating point approximation
        for r in [root - 1, root, root + 1]:
            if r > 0 and r ** exp == n:
                return (r, exp)
    
    return None

def main():
    print("=" * 60)
    print("Advanced Prime Factorization Analysis")
    print("=" * 60)
    print(f"Target Number: {TARGET_NUMBER}")
    print(f"Binary: {bin(TARGET_NUMBER)}")
    print(f"Bits: {TARGET_NUMBER.bit_length()}")
    print()
    
    # First check if it's prime
    print("Checking if the number is prime...")
    if is_prime(TARGET_NUMBER):
        print("✓ The number is PRIME! No factorization possible.")
        return
    else:
        print("✗ The number is composite.")
    
    # Check if it's a perfect power
    print("\nChecking if it's a perfect power...")
    power_result = check_perfect_power(TARGET_NUMBER)
    if power_result:
        base, exp = power_result
        print(f"✓ Perfect power found: {TARGET_NUMBER} = {base}^{exp}")
        return
    else:
        print("✗ Not a perfect power.")
    
    # Try to find small factors
    print("\nSearching for small factors...")
    small_factors = []
    n_remaining = TARGET_NUMBER
    
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        while n_remaining % p == 0:
            small_factors.append(p)
            n_remaining //= p
    
    if small_factors:
        print(f"Found small factors: {small_factors}")
        print(f"Remaining to factor: {n_remaining}")
    
    # Try Fermat's method
    print("\nTrying Fermat's factorization method...")
    fermat_result = fermat_factorization(n_remaining)
    if fermat_result:
        f1, f2 = fermat_result
        print(f"✓ Fermat's method found: {f1} × {f2}")
        
        # Check if these are prime
        if is_prime(f1) and is_prime(f2):
            print(f"✓ Both factors are prime!")
            print(f"\n🎯 FINAL RESULT: {TARGET_NUMBER} = {f1} × {f2}")
            
            # Save results
            save_results(f1, f2, "fermat", small_factors)
            return
    
    # Try Pollard's rho with Brent's improvement
    print("\nTrying Pollard's rho (Brent variant)...")
    factor = pollard_rho_brent(n_remaining)
    if factor:
        other = n_remaining // factor
        print(f"✓ Found factor: {factor}")
        print(f"Other factor: {other}")
        
        if is_prime(factor) and is_prime(other):
            print(f"✓ Both factors are prime!")
            print(f"\n🎯 FINAL RESULT: {TARGET_NUMBER} = {factor} × {other}")
            save_results(factor, other, "pollard_brent", small_factors)
            return
    
    # Try quadratic sieve
    print("\nTrying simplified quadratic sieve...")
    qs_factor = quadratic_sieve_simple(n_remaining)
    if qs_factor:
        other = n_remaining // qs_factor
        print(f"✓ Found factor: {qs_factor}")
        print(f"Other factor: {other}")
        
        if is_prime(qs_factor) and is_prime(other):
            print(f"✓ Both factors are prime!")
            print(f"\n🎯 FINAL RESULT: {TARGET_NUMBER} = {qs_factor} × {other}")
            save_results(qs_factor, other, "quadratic_sieve", small_factors)
            return
    
    # Last resort - check if it might have more than 2 prime factors
    print("\nAnalyzing factorization structure...")
    print("The number might have more than 2 prime factors.")
    
    # Try to find any factors
    print("\nSearching for any factors (this may take a while)...")
    factors = find_all_factors(n_remaining)
    if factors:
        print(f"Found factors: {factors[:10]}...")  # Show first 10
        
        # Check for semiprime factors
        for f in factors:
            if is_prime(f):
                other = n_remaining // f
                if is_prime(other):
                    print(f"\n✓ Found semiprime factorization: {f} × {other}")
                    save_results(f, other, "exhaustive", small_factors)
                    return

def save_results(factor1: int, factor2: int, method: str, small_factors: List[int]):
    """Save the factorization results"""
    memory_dir = Path("memory/swarm-auto-centralized-1752522856366/validation-engineer")
    memory_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "target_number": TARGET_NUMBER,
        "factor1": int(factor1),
        "factor2": int(factor2),
        "method": method,
        "small_factors": small_factors,
        "verification": {
            "product": int(factor1 * factor2),
            "factor1_prime": is_prime(factor1),
            "factor2_prime": is_prime(factor2),
            "correct": factor1 * factor2 == TARGET_NUMBER
        },
        "timestamp": datetime.now().isoformat()
    }
    
    json_path = memory_dir / f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: {json_path}")
    
    # Generate final report
    report = f"""# FINAL VALIDATION REPORT

## Target Number
{TARGET_NUMBER}

## SUCCESSFUL FACTORIZATION
✓ Factor 1: {factor1}
✓ Factor 2: {factor2}

## Verification
- Product: {factor1} × {factor2} = {factor1 * factor2}
- Expected: {TARGET_NUMBER}
- Match: {factor1 * factor2 == TARGET_NUMBER}
- Factor 1 is prime: {is_prime(factor1)}
- Factor 2 is prime: {is_prime(factor2)}

## Method Used
{method}

## Timestamp
{datetime.now().isoformat()}
"""
    
    report_path = memory_dir / "FINAL_FACTORIZATION_RESULT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"📄 Final report saved to: {report_path}")

if __name__ == "__main__":
    main()