"""
Hybrid Factorization System
Combines multiple algorithms for optimal performance on 2539123152460219
"""

import math
import time
import multiprocessing as mp
from typing import Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from pollard_rho_brent import pollard_rho_brent, pollard_rho_parallel_friendly
from ecm_factorization import ecm_one_curve, ecm_parallel_friendly


def is_prime(n: int) -> bool:
    """Miller-Rabin primality test."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # Witnesses for deterministic test up to certain bounds
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


def optimized_trial_division(n: int, limit: int = 10**6) -> Optional[int]:
    """
    Optimized trial division for small factors.
    Uses wheel factorization to skip many composites.
    """
    if n <= 1:
        return None
    
    # Check small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    for p in small_primes:
        if n % p == 0:
            return p
    
    # Wheel factorization mod 30 (skips multiples of 2, 3, 5)
    wheel = [1, 7, 11, 13, 17, 19, 23, 29]
    
    # Start from 30
    base = 30
    while base < limit:
        for offset in wheel:
            candidate = base + offset
            if candidate > limit:
                break
            if n % candidate == 0:
                return candidate
        base += 30
    
    return None


def parallel_trial_division(n: int, num_threads: int = 4) -> Optional[int]:
    """
    Parallel trial division using multiple threads.
    Each thread checks a different range.
    """
    limit = min(int(math.sqrt(n)), 10**8)
    chunk_size = limit // num_threads
    
    def check_range(start: int, end: int) -> Optional[int]:
        """Check a range of potential factors."""
        # Ensure start is odd
        if start % 2 == 0:
            start += 1
        
        for candidate in range(start, end, 2):
            if n % candidate == 0:
                return candidate
        return None
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            start = max(3, i * chunk_size)
            end = min((i + 1) * chunk_size, limit)
            futures.append(executor.submit(check_range, start, end))
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                # Cancel remaining tasks
                for f in futures:
                    f.cancel()
                return result
    
    return None


def gpu_simulation_factorization(n: int, num_parallel: int = 1000) -> Optional[int]:
    """
    Simulate GPU parallelization using process pool.
    In real GPU implementation, these would be CUDA threads.
    """
    import random
    
    # Parameters for parallel Pollard's rho
    params = []
    for i in range(num_parallel):
        x0 = random.randint(2, n - 1)
        c = random.randint(1, n - 1)
        params.append((n, x0, c, 10000))  # 10k iterations per instance
    
    # Run parallel instances
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = []
        for param in params[:100]:  # Limit for CPU simulation
            futures.append(executor.submit(pollard_rho_parallel_friendly, *param))
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                # Cancel remaining tasks
                for f in futures:
                    f.cancel()
                return result
    
    return None


class HybridFactorizer:
    """
    Hybrid factorization system that intelligently selects algorithms.
    """
    
    def __init__(self, n: int):
        self.n = n
        self.size = len(str(n))
        self.sqrt_n = int(math.sqrt(n))
        
    def select_algorithm(self) -> str:
        """Select best algorithm based on number characteristics."""
        # Quick primality check
        if is_prime(self.n):
            return "prime"
        
        # For numbers with small factors, use trial division
        if self.sqrt_n < 10**6:
            return "trial_division"
        
        # For medium numbers, use Pollard's rho
        if self.n < 10**20:
            return "pollard_rho"
        
        # For larger numbers or when Pollard's rho fails, use ECM
        return "ecm"
    
    def factorize(self) -> Tuple[Optional[int], Optional[int], float, str]:
        """
        Main factorization method.
        Returns (factor1, factor2, time_taken, method_used)
        """
        start_time = time.time()
        
        print(f"Factorizing {self.n} ({self.size} digits)")
        print(f"sqrt(n) ≈ {self.sqrt_n} ({len(str(self.sqrt_n))} digits)")
        
        # Stage 1: Quick trial division for small factors
        print("\nStage 1: Checking small factors...")
        factor = optimized_trial_division(self.n, min(10**6, self.sqrt_n))
        if factor:
            factor2 = self.n // factor
            return factor, factor2, time.time() - start_time, "trial_division"
        
        # Stage 2: Parallel trial division for medium factors
        if self.sqrt_n < 10**9:
            print("\nStage 2: Parallel trial division...")
            factor = parallel_trial_division(self.n)
            if factor:
                factor2 = self.n // factor
                return factor, factor2, time.time() - start_time, "parallel_trial"
        
        # Stage 3: Pollard's rho for larger factors
        print("\nStage 3: Pollard's rho with Brent's improvement...")
        factor = pollard_rho_brent(self.n, max_iterations=10**7)
        if factor:
            factor2 = self.n // factor
            return factor, factor2, time.time() - start_time, "pollard_rho"
        
        # Stage 4: ECM for difficult semiprimes
        print("\nStage 4: Elliptic Curve Method...")
        for attempt in range(3):
            factor = ecm_one_curve(self.n, B1=50000)
            if factor:
                factor2 = self.n // factor
                return factor, factor2, time.time() - start_time, "ecm"
        
        # Stage 5: GPU simulation (would be actual GPU in production)
        print("\nStage 5: Simulated GPU parallelization...")
        factor = gpu_simulation_factorization(self.n, num_parallel=100)
        if factor:
            factor2 = self.n // factor
            return factor, factor2, time.time() - start_time, "gpu_parallel"
        
        return None, None, time.time() - start_time, "failed"


def analyze_factorization_strategy(n: int):
    """Analyze and recommend factorization strategy."""
    print(f"\n{'='*60}")
    print("FACTORIZATION STRATEGY ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\nTarget number: {n}")
    print(f"Number of digits: {len(str(n))}")
    print(f"Binary length: {n.bit_length()} bits")
    
    sqrt_n = int(math.sqrt(n))
    print(f"\nSquare root: {sqrt_n}")
    print(f"sqrt digits: {len(str(sqrt_n))}")
    
    # Estimate factor sizes if semiprime
    if len(str(n)) == 16:  # Our target is 16 digits
        print("\nIf this is a semiprime (product of two primes):")
        print("- Factors are likely 7-8 digits each")
        print("- Trial division effective up to ~10^6")
        print("- Pollard's rho effective for 7-8 digit factors")
        print("- ECM effective for up to ~20 digit factors")
    
    print("\nRecommended approach:")
    print("1. Quick trial division up to 10^6")
    print("2. Pollard's rho with multiple attempts")
    print("3. ECM with increasing bounds")
    print("4. GPU parallelization for massive search")


if __name__ == "__main__":
    # Target number
    target = 2539123152460219
    
    # Analyze strategy
    analyze_factorization_strategy(target)
    
    # Run hybrid factorization
    print(f"\n{'='*60}")
    print("STARTING HYBRID FACTORIZATION")
    print(f"{'='*60}")
    
    factorizer = HybridFactorizer(target)
    f1, f2, elapsed, method = factorizer.factorize()
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    
    if f1 and f2:
        print(f"SUCCESS! Factors found: {f1} × {f2}")
        print(f"Verification: {f1} × {f2} = {f1 * f2}")
        print(f"Correct: {f1 * f2 == target}")
        print(f"Method used: {method}")
        print(f"Time taken: {elapsed:.4f} seconds")
        
        # Check if factors are prime
        print(f"\nFactor analysis:")
        print(f"{f1}: {'PRIME' if is_prime(f1) else 'COMPOSITE'} ({len(str(f1))} digits)")
        print(f"{f2}: {'PRIME' if is_prime(f2) else 'COMPOSITE'} ({len(str(f2))} digits)")
    else:
        print(f"Failed to factorize after {elapsed:.4f} seconds")
        print("This might be a prime number or require more advanced methods")