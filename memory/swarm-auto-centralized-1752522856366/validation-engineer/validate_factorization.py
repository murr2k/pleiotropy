#!/usr/bin/env python3
"""
Validation Engineer - Prime Factorization Validator
Target: 2539123152460219
Mission: Validate factorization and benchmark performance
"""

import time
import math
import json
from datetime import datetime
from pathlib import Path

TARGET_NUMBER = 2539123152460219

def is_prime(n):
    """Check if a number is prime using deterministic Miller-Rabin test"""
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

def trial_division(n):
    """Factorize using trial division"""
    print("Starting trial division factorization...")
    start_time = time.time()
    
    # Check if even
    if n % 2 == 0:
        other = n // 2
        if is_prime(2) and is_prime(other):
            elapsed = (time.time() - start_time) * 1000
            return 2, other, elapsed, "trial_even"
    
    # Check small primes
    small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73]
    for p in small_primes:
        if n % p == 0:
            other = n // p
            if is_prime(p) and is_prime(other):
                elapsed = (time.time() - start_time) * 1000
                return p, other, elapsed, "trial_small"
    
    # Full trial division
    sqrt_n = int(math.sqrt(n)) + 1
    progress_interval = 1000000
    checked = 0
    
    # Use 6k±1 optimization
    i = 5
    while i <= sqrt_n:
        if n % i == 0:
            other = n // i
            if is_prime(i) and is_prime(other):
                elapsed = (time.time() - start_time) * 1000
                return i, other, elapsed, "trial_division"
        
        if i + 2 <= sqrt_n and n % (i + 2) == 0:
            other = n // (i + 2)
            if is_prime(i + 2) and is_prime(other):
                elapsed = (time.time() - start_time) * 1000
                return i + 2, other, elapsed, "trial_division"
        
        i += 6
        checked += 2
        
        if checked % progress_interval == 0:
            progress = (i / sqrt_n) * 100
            print(f"  Progress: {progress:.2f}% (checked {checked:,} candidates)")
    
    elapsed = (time.time() - start_time) * 1000
    return None, None, elapsed, "trial_failed"

def pollards_rho(n):
    """Factorize using Pollard's rho algorithm"""
    print("Starting Pollard's rho factorization...")
    start_time = time.time()
    
    if n % 2 == 0:
        other = n // 2
        if is_prime(2) and is_prime(other):
            elapsed = (time.time() - start_time) * 1000
            return 2, other, elapsed, "pollard_even"
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    # Try different parameters
    c_values = [1, 2, 3, 5, 7, 11]
    
    for c in c_values:
        x = 2
        y = 2
        d = 1
        
        def f(x):
            return (x * x + c) % n
        
        iterations = 0
        max_iterations = 100000
        
        while d == 1 and iterations < max_iterations:
            x = f(x)
            y = f(f(y))
            d = gcd(abs(x - y), n)
            iterations += 1
            
            if iterations % 10000 == 0:
                print(f"  Pollard's rho iteration {iterations} with c={c}")
        
        if d != 1 and d != n:
            other = n // d
            if is_prime(d) and is_prime(other):
                elapsed = (time.time() - start_time) * 1000
                return min(d, other), max(d, other), elapsed, "pollard_rho"
    
    elapsed = (time.time() - start_time) * 1000
    return None, None, elapsed, "pollard_failed"

def validate_result(factor1, factor2):
    """Validate the factorization result"""
    print("\nValidating result...")
    
    # Check multiplication
    product = factor1 * factor2
    if product == TARGET_NUMBER:
        print(f"✓ Multiplication check: {factor1} × {factor2} = {product}")
    else:
        print(f"✗ Multiplication check failed: {factor1} × {factor2} = {product} (expected {TARGET_NUMBER})")
        return False
    
    # Check primality
    if is_prime(factor1):
        print(f"✓ Factor 1 ({factor1}) is prime")
    else:
        print(f"✗ Factor 1 ({factor1}) is NOT prime")
        return False
    
    if is_prime(factor2):
        print(f"✓ Factor 2 ({factor2}) is prime")
    else:
        print(f"✗ Factor 2 ({factor2}) is NOT prime")
        return False
    
    return True

def main():
    print("=" * 40)
    print("Prime Factorization Validation Engineer")
    print("=" * 40)
    print(f"Target Number: {TARGET_NUMBER}")
    print()
    
    results = {
        "target_number": TARGET_NUMBER,
        "timestamp": datetime.now().isoformat(),
        "methods_tried": [],
        "success": False,
        "factor1": None,
        "factor2": None,
        "total_time_ms": 0,
        "algorithm_used": None
    }
    
    total_start = time.time()
    
    # Try trial division
    f1, f2, elapsed, method = trial_division(TARGET_NUMBER)
    results["methods_tried"].append({
        "method": "trial_division",
        "time_ms": elapsed,
        "success": f1 is not None
    })
    
    if f1 is not None:
        print(f"\n✓ Trial division successful!")
        print(f"  Factors: {f1} × {f2}")
        print(f"  Time: {elapsed:.2f}ms")
        if validate_result(f1, f2):
            results["success"] = True
            results["factor1"] = f1
            results["factor2"] = f2
            results["algorithm_used"] = method
    else:
        print(f"\n✗ Trial division failed after {elapsed:.2f}ms")
        
        # Try Pollard's rho
        f1, f2, elapsed, method = pollards_rho(TARGET_NUMBER)
        results["methods_tried"].append({
            "method": "pollards_rho",
            "time_ms": elapsed,
            "success": f1 is not None
        })
        
        if f1 is not None:
            print(f"\n✓ Pollard's rho successful!")
            print(f"  Factors: {f1} × {f2}")
            print(f"  Time: {elapsed:.2f}ms")
            if validate_result(f1, f2):
                results["success"] = True
                results["factor1"] = f1
                results["factor2"] = f2
                results["algorithm_used"] = method
        else:
            print(f"\n✗ Pollard's rho failed after {elapsed:.2f}ms")
    
    results["total_time_ms"] = (time.time() - total_start) * 1000
    
    # Final result
    if results["success"]:
        print(f"\n🎯 FINAL RESULT: {TARGET_NUMBER} = {results['factor1']} × {results['factor2']}")
        print(f"   Algorithm: {results['algorithm_used']}")
        print(f"   Total time: {results['total_time_ms']:.2f}ms")
    else:
        print(f"\n✗ Failed to factorize {TARGET_NUMBER}")
    
    # Save results
    memory_dir = Path("memory/swarm-auto-centralized-1752522856366/validation-engineer")
    memory_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    json_path = memory_dir / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Results saved to: {json_path}")
    
    # Generate report
    report = f"""# Validation Engineer Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Target Number
{TARGET_NUMBER}

## Results
{'✓ SUCCESS' if results['success'] else '✗ FAILED'}

Factor 1: {results['factor1'] if results['factor1'] else 'Not found'}
Factor 2: {results['factor2'] if results['factor2'] else 'Not found'}

## Performance Metrics
- Total Time: {results['total_time_ms']:.2f}ms
- Algorithm Used: {results['algorithm_used'] if results['algorithm_used'] else 'None'}

## Methods Tried
"""
    
    for method in results["methods_tried"]:
        report += f"- {method['method']}: {'Success' if method['success'] else 'Failed'} ({method['time_ms']:.2f}ms)\n"
    
    if results['success']:
        report += f"""
## Validation Status
- [x] Multiplication verified: {results['factor1']} × {results['factor2']} = {TARGET_NUMBER}
- [x] Factor 1 primality verified: {results['factor1']} is prime
- [x] Factor 2 primality verified: {results['factor2']} is prime

## Conclusion
The factorization has been successfully validated.
"""
    else:
        report += """
## Conclusion
The factorization could not be completed with the available methods.
"""
    
    report_path = memory_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"📄 Report saved to: {report_path}")

if __name__ == "__main__":
    main()