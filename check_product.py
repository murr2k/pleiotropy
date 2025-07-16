#!/usr/bin/env python3
"""
Calculate the product of 78912301 and 32176519
"""

def main():
    num1 = 78912301
    num2 = 32176519
    
    print(f"Calculating: {num1:,} × {num2:,}")
    
    product = num1 * num2
    
    print(f"\nResult: {product:,}")
    print(f"\nIn standard form: {product}")
    
    # Also verify against the original number
    original = 2539123152460219
    print(f"\nOriginal number: {original:,}")
    print(f"Do they match? {product == original}")

if __name__ == "__main__":
    main()