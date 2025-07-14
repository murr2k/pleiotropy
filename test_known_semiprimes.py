#!/usr/bin/env python3
"""
Test with known semiprime sizes to calibrate
"""

import subprocess
import time

# Known semiprimes from our previous tests
known_semiprimes = [
    ("210656506727", 12, 0.001),  # 12 digits, 0.001s
    ("2133019384970323", 16, 0.011),  # 16 digits, 0.011s
    ("4349182478874450510265070424251", 31, 19.75),  # 31 digits, 19.75s
]

def test_semiprime(n_str, expected_digits, expected_time):
    """Test a known semiprime"""
    print(f"\nTesting {expected_digits}-digit semiprime:")
    print(f"Number: {n_str[:20]}...{n_str[-10:] if len(n_str) > 30 else ''}")
    print(f"Expected time: ~{expected_time}s")
    
    start = time.time()
    
    try:
        result = subprocess.run(
            ['python3', 'factor_large_number.py'],
            input=n_str,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"✓ Factored in {elapsed:.3f}s (expected ~{expected_time}s)")
            
            # Show factors
            lines = result.stdout.split('\n')
            for line in lines:
                if "Prime" in line and ":" in line:
                    print(f"  {line.strip()}")
            
            return True, elapsed
        else:
            print(f"✗ Failed: {result.stderr}")
            return False, elapsed
            
    except subprocess.TimeoutExpired:
        print("✗ Timeout!")
        return False, None
    except Exception as e:
        print(f"✗ Error: {e}")
        return False, None

def main():
    print("🔍 Testing Known Semiprimes")
    print("="*60)
    
    results = []
    
    for n_str, digits, expected_time in known_semiprimes:
        success, actual_time = test_semiprime(n_str, digits, expected_time)
        
        if success and actual_time:
            results.append((digits, actual_time))
    
    # Analyze scaling
    print("\n" + "="*60)
    print("SCALING ANALYSIS:")
    
    for d, t in results:
        print(f"  {d} digits: {t:.3f}s")
    
    if len(results) >= 2:
        # Calculate scaling factor
        import math
        
        print("\nExponential fit: time = exp(a * digits + b)")
        
        # Use first and last points
        d1, t1 = results[0]
        d2, t2 = results[-1]
        
        a = (math.log(t2) - math.log(t1)) / (d2 - d1)
        b = math.log(t1) - a * d1
        
        print(f"  a = {a:.4f}")
        print(f"  b = {b:.4f}")
        
        # Predict for various sizes
        print("\nPredictions:")
        for d in [35, 40, 45, 50, 55]:
            pred_time = math.exp(a * d + b)
            print(f"  {d} digits: {pred_time:.1f}s ({pred_time/60:.1f} min)")
        
        # Find target for 600s
        target_digits = (math.log(600) - b) / a
        print(f"\nTarget for 600s (10 min): {target_digits:.0f} digits")

if __name__ == "__main__":
    main()