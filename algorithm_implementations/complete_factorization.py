"""
Complete factorization to find the two prime factors of 2539123152460219
"""

from hybrid_factorization import HybridFactorizer, is_prime
import time


def complete_factorization(n: int):
    """
    Completely factorize n into prime factors.
    """
    print(f"\nComplete Factorization of {n}")
    print("=" * 60)
    
    factors = []
    remaining = n
    
    while remaining > 1:
        print(f"\nFactorizing: {remaining}")
        
        factorizer = HybridFactorizer(remaining)
        f1, f2, elapsed, method = factorizer.factorize()
        
        if f1 is None:
            print(f"Cannot factorize {remaining} - it might be prime")
            if is_prime(remaining):
                print(f"Confirmed: {remaining} is PRIME")
                factors.append(remaining)
            break
        
        print(f"Found factor: {f1} (method: {method}, time: {elapsed:.4f}s)")
        
        # Check if f1 is prime
        if is_prime(f1):
            print(f"{f1} is PRIME")
            factors.append(f1)
        else:
            print(f"{f1} is COMPOSITE - will factor further")
            # We'll factor this later
            
        remaining = f2
        
        # Check if remaining factor is prime
        if is_prime(remaining):
            print(f"{remaining} is PRIME")
            factors.append(remaining)
            break
    
    return factors


if __name__ == "__main__":
    target = 2539123152460219
    
    print("COMPLETE FACTORIZATION ANALYSIS")
    print("=" * 60)
    
    start_time = time.time()
    
    # First, we know 13 is a factor
    print(f"\nInitial factorization: {target} = 13 × 195317165573863")
    
    # Now factor the composite part
    composite = 195317165573863
    print(f"\nNeed to factor: {composite}")
    
    prime_factors = complete_factorization(composite)
    
    # Add 13 to the factors
    all_factors = [13] + prime_factors
    all_factors.sort()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    print(f"\nPrime factorization of {target}:")
    print(f"{' × '.join(map(str, all_factors))}")
    
    # Verify
    product = 1
    for f in all_factors:
        product *= f
    
    print(f"\nVerification: {product} = {target}")
    print(f"Correct: {product == target}")
    
    # Check for two prime factors as requested
    if len(all_factors) == 2:
        print(f"\nThe two prime factors are: {all_factors[0]} and {all_factors[1]}")
    else:
        print(f"\nNote: Found {len(all_factors)} prime factors, not 2 as expected")
    
    print(f"\nTotal time: {time.time() - start_time:.4f} seconds")