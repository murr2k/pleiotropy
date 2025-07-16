"""
Elliptic Curve Method (ECM) for Integer Factorization
Optimized for finding factors of 2539123152460219
"""

import math
import random
from typing import Optional, Tuple, List
import time


def gcd(a: int, b: int) -> int:
    """Compute greatest common divisor."""
    while b:
        a, b = b, a % b
    return a


def mod_inverse(a: int, n: int) -> Optional[int]:
    """
    Compute modular inverse of a modulo n using extended Euclidean algorithm.
    Returns None if gcd(a, n) != 1.
    """
    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        if a == 0:
            return b, 0, 1
        gcd_val, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd_val, x, y
    
    gcd_val, x, _ = extended_gcd(a % n, n)
    if gcd_val != 1:
        return None
    return (x % n + n) % n


class EllipticCurve:
    """
    Elliptic curve in Montgomery form: B*y^2 = x^3 + A*x^2 + x (mod n)
    """
    def __init__(self, A: int, B: int, n: int):
        self.A = A % n
        self.B = B % n
        self.n = n


class Point:
    """Point on an elliptic curve."""
    def __init__(self, x: int, z: int, curve: EllipticCurve):
        self.x = x % curve.n
        self.z = z % curve.n
        self.curve = curve
    
    def is_infinity(self) -> bool:
        """Check if point is at infinity."""
        return self.z == 0


def point_add(P: Point, Q: Point, diff: Point) -> Optional[Point]:
    """
    Add two points on Montgomery curve.
    diff = P - Q must be known.
    Returns None if a factor is found.
    """
    curve = P.curve
    n = curve.n
    
    if P.is_infinity():
        return Q
    if Q.is_infinity():
        return P
    
    # Montgomery ladder addition
    u = (P.x - P.z) * (Q.x + Q.z) % n
    v = (P.x + P.z) * (Q.x - Q.z) % n
    
    add = u + v
    sub = u - v
    
    x = (add * add * diff.z) % n
    z = (sub * sub * diff.x) % n
    
    return Point(x, z, curve)


def point_double(P: Point) -> Optional[Point]:
    """
    Double a point on Montgomery curve.
    Returns None if a factor is found.
    """
    curve = P.curve
    n = curve.n
    
    if P.is_infinity():
        return P
    
    # Montgomery doubling
    s = (P.x + P.z) % n
    d = (P.x - P.z) % n
    s2 = (s * s) % n
    d2 = (d * d) % n
    
    x = (s2 * d2) % n
    
    # z = 4*x*z*(x^2 + A*x*z + z^2)
    t = s2 - d2
    z = (t * ((curve.A - 2) * t // 4 + d2)) % n
    
    return Point(x, z, curve)


def scalar_multiply(k: int, P: Point) -> Optional[Point]:
    """
    Compute k*P using Montgomery ladder.
    Returns None if a factor is found during computation.
    """
    if k == 0 or P.is_infinity():
        return Point(0, 0, P.curve)  # Point at infinity
    
    # Montgomery ladder
    R0 = P
    R1 = point_double(P)
    
    if R1 is None:
        return None
    
    # Process bits of k from second most significant to least
    bits = bin(k)[3:]  # Skip '0b' and the MSB
    
    for bit in bits:
        if bit == '0':
            R1 = point_add(R0, R1, P)
            R0 = point_double(R0)
        else:
            R0 = point_add(R0, R1, P)
            R1 = point_double(R1)
        
        if R0 is None or R1 is None:
            return None
    
    return R0


def ecm_stage1(n: int, B1: int, curve: EllipticCurve, P: Point) -> Optional[int]:
    """
    Stage 1 of ECM: compute k*P where k = lcm(1,2,...,B1).
    Returns a factor if found, None otherwise.
    """
    Q = P
    
    # Compute k = product of prime powers <= B1
    primes = sieve_of_eratosthenes(B1)
    
    for p in primes:
        # Compute highest power of p <= B1
        pp = p
        while pp <= B1:
            # Q = p*Q
            for _ in range(p):
                new_Q = scalar_multiply(p, Q)
                if new_Q is None:
                    # Check for factor
                    d = gcd(Q.z, n)
                    if 1 < d < n:
                        return d
                Q = new_Q
            pp *= p
    
    # Check final GCD
    d = gcd(Q.z, n)
    if 1 < d < n:
        return d
    
    return None


def sieve_of_eratosthenes(limit: int) -> List[int]:
    """Generate all primes up to limit using sieve."""
    if limit < 2:
        return []
    
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    
    return [i for i in range(2, limit + 1) if sieve[i]]


def ecm_one_curve(n: int, B1: int = 10000, B2: int = 100000) -> Optional[int]:
    """
    Run ECM with one random curve.
    
    Args:
        n: Number to factorize
        B1: Stage 1 bound
        B2: Stage 2 bound (not implemented yet)
        
    Returns:
        A factor if found, None otherwise
    """
    # Generate random curve parameters
    while True:
        # Random Montgomery curve
        A = random.randint(1, n - 1)
        x = random.randint(1, n - 1)
        z = 1
        
        # Compute B from curve equation
        B = (x * x * x + A * x * x + x) % n
        
        # Check that curve is valid
        d = gcd(4 * A * A * A + 27 * B * B, n)
        if d == n:
            continue  # Bad curve, try again
        if 1 < d < n:
            return d  # Lucky factor!
        
        break
    
    curve = EllipticCurve(A, B, n)
    P = Point(x, z, curve)
    
    # Run stage 1
    factor = ecm_stage1(n, B1, curve, P)
    
    # TODO: Implement stage 2 for better performance
    
    return factor


def ecm_parallel_friendly(n: int, curve_params: Tuple[int, int, int], B1: int) -> Optional[int]:
    """
    GPU-friendly ECM that can be parallelized.
    Each thread works on a different curve.
    
    Args:
        n: Number to factorize
        curve_params: (A, x0, z0) for the curve
        B1: Stage 1 bound
        
    Returns:
        A factor or None
    """
    A, x0, z0 = curve_params
    
    # Compute B from curve equation
    B = (x0 * x0 * x0 + A * x0 * x0 + x0) % n
    
    curve = EllipticCurve(A, B, n)
    P = Point(x0, z0, curve)
    
    return ecm_stage1(n, B1, curve, P)


def factorize_with_ecm(n: int, max_curves: int = 100) -> Tuple[Optional[int], Optional[int], float]:
    """
    Complete factorization using ECM.
    
    Args:
        n: Number to factorize
        max_curves: Maximum number of curves to try
        
    Returns:
        Tuple of (factor1, factor2, time_taken)
    """
    start_time = time.time()
    
    # Adjust bounds based on expected factor size
    # For ~7-8 digit factors, use moderate bounds
    B1_values = [2000, 11000, 50000, 250000]
    
    for B1 in B1_values:
        print(f"Trying ECM with B1={B1}")
        
        for curve_num in range(max_curves // len(B1_values)):
            factor = ecm_one_curve(n, B1)
            
            if factor:
                factor2 = n // factor
                if factor * factor2 == n:
                    return min(factor, factor2), max(factor, factor2), time.time() - start_time
    
    return None, None, time.time() - start_time


if __name__ == "__main__":
    # Test with the target number
    target = 2539123152460219
    print(f"Factorizing {target} using Elliptic Curve Method...")
    
    f1, f2, elapsed = factorize_with_ecm(target, max_curves=50)
    
    if f1 and f2:
        print(f"Factors found: {f1} × {f2} = {f1 * f2}")
        print(f"Verification: {f1 * f2 == target}")
        print(f"Time taken: {elapsed:.4f} seconds")
    else:
        print("Failed to find factors with ECM")