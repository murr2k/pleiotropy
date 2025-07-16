#!/usr/bin/env python3
"""
Verify if 78912301 and 32176519 are factors of 2539123152460219
"""

def check_divisibility(n, potential_factor):
    """Check if potential_factor divides n evenly"""
    quotient, remainder = divmod(n, potential_factor)
    return remainder == 0, quotient

def is_prime(n):
    """Check if n is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check odd divisors up to sqrt(n)
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

def main():
    n = 2539123152460219
    factor1 = 78912301
    factor2 = 32176519
    
    print(f"Verifying if {factor1} and {factor2} are factors of {n}")
    print("="*60)
    
    # Check if factor1 divides n
    divides1, quotient1 = check_divisibility(n, factor1)
    print(f"\nChecking {factor1}:")
    print(f"  {n} ÷ {factor1} = {quotient1} remainder {n - factor1 * quotient1}")
    print(f"  Is {factor1} a factor? {divides1}")
    
    if divides1:
        print(f"  Quotient: {quotient1}")
        print(f"  Is quotient {quotient1} equal to {factor2}? {quotient1 == factor2}")
    
    # Check if factor2 divides n
    divides2, quotient2 = check_divisibility(n, factor2)
    print(f"\nChecking {factor2}:")
    print(f"  {n} ÷ {factor2} = {quotient2} remainder {n - factor2 * quotient2}")
    print(f"  Is {factor2} a factor? {divides2}")
    
    if divides2:
        print(f"  Quotient: {quotient2}")
        print(f"  Is quotient {quotient2} equal to {factor1}? {quotient2 == factor1}")
    
    # Check if their product equals n
    product = factor1 * factor2
    print(f"\nChecking product:")
    print(f"  {factor1} × {factor2} = {product}")
    print(f"  Does this equal {n}? {product == n}")
    
    if product == n:
        print("\n✓ VERIFIED: These two numbers multiply to give the target!")
        
        # Check if they are prime
        print(f"\nChecking primality:")
        print(f"  Checking if {factor1} is prime...")
        prime1 = is_prime(factor1)
        print(f"  {factor1} is prime: {prime1}")
        
        print(f"  Checking if {factor2} is prime...")
        prime2 = is_prime(factor2)
        print(f"  {factor2} is prime: {prime2}")
        
        if prime1 and prime2:
            print(f"\n{'='*60}")
            print("CONCLUSION: The number IS a semiprime!")
            print(f"{n} = {factor1} × {factor2}")
            print("Both factors are prime numbers.")
        else:
            print("\nAt least one factor is not prime.")
            
            # Factor the non-prime ones
            if not prime1:
                print(f"\nFactoring {factor1}...")
                factors1 = []
                temp = factor1
                for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
                    while temp % p == 0:
                        factors1.append(p)
                        temp //= p
                if temp > 1:
                    factors1.append(temp)
                print(f"  {factor1} = {' × '.join(map(str, factors1))}")
            
            if not prime2:
                print(f"\nFactoring {factor2}...")
                factors2 = []
                temp = factor2
                for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
                    while temp % p == 0:
                        factors2.append(p)
                        temp //= p
                if temp > 1:
                    factors2.append(temp)
                print(f"  {factor2} = {' × '.join(map(str, factors2))}")
    else:
        print(f"\n✗ These numbers do NOT multiply to give {n}")
        print(f"  Difference: {abs(product - n)}")

if __name__ == "__main__":
    main()