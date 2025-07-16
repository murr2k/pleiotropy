#!/usr/bin/env python3
"""
Correct verification of the factorization
"""

def is_prime_thorough(n):
    """Thorough primality test"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check all odd numbers up to sqrt(n)
    import math
    limit = int(math.sqrt(n)) + 1
    for i in range(3, limit, 2):
        if n % i == 0:
            print(f"    Found divisor: {i} (so {n} = {i} × {n//i})")
            return False
        if i % 1000000 == 1 and i > 1000000:
            print(f"    Checked up to {i:,} of {limit:,}")
    
    return True

def main():
    n = 2539123152460219
    factor1 = 78912301  
    factor2 = 32176519
    
    print("VERIFICATION OF FACTORIZATION")
    print("="*60)
    
    # Verify the multiplication
    product = factor1 * factor2
    print(f"\nStep 1: Verify multiplication")
    print(f"{factor1:,} × {factor2:,} = {product:,}")
    print(f"Target number: {n:,}")
    print(f"Match: {product == n} ✓" if product == n else f"No match ✗")
    
    # Check primality of each factor
    print(f"\nStep 2: Check if {factor1:,} is prime")
    is_f1_prime = is_prime_thorough(factor1)
    
    print(f"\nStep 3: Check if {factor2:,} is prime")  
    is_f2_prime = is_prime_thorough(factor2)
    
    print(f"\n{'='*60}")
    print("FINAL CONCLUSION:")
    
    if product == n:
        if is_f1_prime and is_f2_prime:
            print(f"✓ {n} IS a semiprime!")
            print(f"  It is the product of exactly two prime numbers:")
            print(f"  {factor1} × {factor2} = {n}")
        else:
            print(f"✗ {n} is NOT a semiprime")
            print(f"  While {factor1} × {factor2} = {n},")
            print(f"  at least one of these factors is composite (not prime)")
            
            # Show the complete factorization
            print(f"\nComplete prime factorization:")
            # We know from earlier: 78912301 = 13 × 19 × 319483
            #                      32176519 = 19 × 1693501
            print(f"  78912301 = 13 × 19 × 319483")
            print(f"  32176519 = 19 × 1693501")
            print(f"  Therefore: {n} = 13 × 19² × 319483 × 1693501")

if __name__ == "__main__":
    main()