"""
Pollard's Rho Algorithm with Brent's Improvement
Optimized for finding factors of 2539123152460219
"""

import math
import random
from typing import Optional, Tuple
import time


def gcd(a: int, b: int) -> int:
    """Compute greatest common divisor using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return a


def f(x: int, n: int, c: int = 1) -> int:
    """Polynomial function for Pollard's rho: f(x) = (x^2 + c) mod n"""
    return (x * x + c) % n


def pollard_rho_brent(n: int, max_iterations: int = 10**7) -> Optional[int]:
    """
    Pollard's rho algorithm with Brent's improvement.
    More efficient than the classic Floyd's cycle detection.
    
    Args:
        n: Number to factorize
        max_iterations: Maximum iterations before giving up
        
    Returns:
        A non-trivial factor of n, or None if not found
    """
    if n <= 1:
        return None
    if n % 2 == 0:
        return 2
    
    # Small primes check
    small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    for p in small_primes:
        if n % p == 0:
            return p
    
    # Multiple attempts with different parameters
    for attempt in range(10):
        x = random.randint(2, n - 2)
        c = random.randint(1, n - 1)
        y = x
        d = 1
        
        power = 1
        lam = 1
        
        while d == 1 and lam < max_iterations:
            # Brent's cycle detection
            for _ in range(min(lam, max_iterations - lam)):
                x = f(x, n, c)
                d = gcd(abs(x - y), n)
                
                if d != 1 and d != n:
                    return d
            
            # Update y and double lambda
            y = x
            lam *= 2
            power *= 2
        
        if d != 1 and d != n:
            return d
    
    return None


def pollard_rho_parallel_friendly(n: int, start_x: int, c: int, iterations: int) -> Optional[int]:
    """
    GPU-friendly version of Pollard's rho that can be parallelized.
    Each thread can run with different starting values.
    
    Args:
        n: Number to factorize
        start_x: Starting value for this instance
        c: Constant for the polynomial
        iterations: Number of iterations to perform
        
    Returns:
        A factor or None
    """
    x = start_x
    y = start_x
    d = 1
    
    # Product accumulation for efficiency
    product = 1
    
    for i in range(iterations):
        x = f(x, n, c)
        if i % 2 == 0:
            y = f(f(y, n, c), n, c)
        
        # Accumulate products to reduce GCD calls
        product = (product * abs(x - y)) % n
        
        # Check GCD periodically
        if i % 100 == 0:
            d = gcd(product, n)
            if d != 1 and d != n:
                return d
            product = 1
    
    # Final GCD check
    d = gcd(product, n)
    if d != 1 and d != n:
        return d
    
    return None


def factorize_with_pollard_rho(n: int) -> Tuple[Optional[int], Optional[int], float]:
    """
    Complete factorization using Pollard's rho with Brent's improvement.
    
    Args:
        n: Number to factorize
        
    Returns:
        Tuple of (factor1, factor2, time_taken)
    """
    start_time = time.time()
    
    # Check if n is prime (basic primality test)
    if n < 2:
        return None, None, 0
    
    # Try to find first factor
    factor1 = pollard_rho_brent(n)
    
    if factor1 is None:
        return None, None, time.time() - start_time
    
    factor2 = n // factor1
    
    # Verify factorization
    if factor1 * factor2 == n:
        return min(factor1, factor2), max(factor1, factor2), time.time() - start_time
    
    return None, None, time.time() - start_time


if __name__ == "__main__":
    # Test with the target number
    target = 2539123152460219
    print(f"Factorizing {target} using Pollard's Rho with Brent's improvement...")
    
    f1, f2, elapsed = factorize_with_pollard_rho(target)
    
    if f1 and f2:
        print(f"Factors found: {f1} × {f2} = {f1 * f2}")
        print(f"Verification: {f1 * f2 == target}")
        print(f"Time taken: {elapsed:.4f} seconds")
    else:
        print("Failed to find factors with Pollard's rho")