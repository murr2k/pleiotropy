#!/usr/bin/env python3
"""
Test a single semiprime of specific size
"""

import time
import subprocess
import gmpy2
import random

# Generate a 40-digit semiprime
digits = 40
prime_bits = int(digits * 3.322 / 2)

print(f"Generating {digits}-digit semiprime...")

# Generate two primes
p1 = gmpy2.mpz(2) ** prime_bits + random.randint(10**5, 10**6)
p1 = gmpy2.next_prime(p1)

p2 = gmpy2.mpz(2) ** prime_bits + random.randint(10**5, 10**6)
p2 = gmpy2.next_prime(p2)

n = p1 * p2
n_str = str(n)

print(f"Generated {len(n_str)}-digit number")
print(f"Number: {n_str[:50]}...")
print(f"\nFactoring...")

start = time.time()

# Run factorization
result = subprocess.run(
    ['python3', 'factor_large_number.py'],
    input=n_str,
    capture_output=True,
    text=True,
    timeout=1200
)

elapsed = time.time() - start

print(f"\nCompleted in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

if "✓ This IS a semiprime!" in result.stdout:
    print("✓ Successfully factored!")
    
    # Show factors
    lines = result.stdout.split('\n')
    for line in lines:
        if "Prime" in line and ":" in line:
            print(f"  {line.strip()}")