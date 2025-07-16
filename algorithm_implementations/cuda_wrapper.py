"""
Python wrapper for CUDA factorization
Provides interface to GPU-accelerated factorization algorithms
"""

import ctypes
import os
import numpy as np
from pathlib import Path
import subprocess
import time
from typing import Optional, Tuple


class CUDAFactorizer:
    """
    Wrapper for CUDA factorization implementation.
    """
    
    def __init__(self):
        self.lib_path = None
        self.lib = None
        self._compile_cuda()
        self._load_library()
    
    def _compile_cuda(self):
        """Compile CUDA code to shared library."""
        cuda_file = Path(__file__).parent / "cuda_factorization.cu"
        lib_file = Path(__file__).parent / "cuda_factorization.so"
        
        if not lib_file.exists() or cuda_file.stat().st_mtime > lib_file.stat().st_mtime:
            print("Compiling CUDA kernels...")
            try:
                # Compile CUDA code
                cmd = [
                    "nvcc",
                    "-shared",
                    "-fPIC",
                    "-o", str(lib_file),
                    str(cuda_file),
                    "-lcurand"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"CUDA compilation failed: {result.stderr}")
                    raise RuntimeError("Failed to compile CUDA code")
                
                print("CUDA compilation successful!")
                self.lib_path = lib_file
            except FileNotFoundError:
                print("NVCC not found. Please ensure CUDA toolkit is installed.")
                raise
    
    def _load_library(self):
        """Load the compiled CUDA library."""
        if self.lib_path and self.lib_path.exists():
            self.lib = ctypes.CDLL(str(self.lib_path))
            
            # Set up function signature
            self.lib.cuda_factorize.argtypes = [
                ctypes.c_ulonglong,  # n
                ctypes.POINTER(ctypes.c_ulonglong),  # factor1
                ctypes.POINTER(ctypes.c_ulonglong)   # factor2
            ]
            self.lib.cuda_factorize.restype = None
    
    def factorize(self, n: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Factorize a number using GPU acceleration.
        
        Args:
            n: Number to factorize
            
        Returns:
            Tuple of (factor1, factor2) or (None, None) if failed
        """
        if not self.lib:
            raise RuntimeError("CUDA library not loaded")
        
        factor1 = ctypes.c_ulonglong(0)
        factor2 = ctypes.c_ulonglong(0)
        
        # Call CUDA function
        self.lib.cuda_factorize(
            ctypes.c_ulonglong(n),
            ctypes.byref(factor1),
            ctypes.byref(factor2)
        )
        
        if factor1.value > 0 and factor2.value > 0:
            return factor1.value, factor2.value
        
        return None, None


def gpu_accelerated_factorization(n: int) -> Tuple[Optional[int], Optional[int], float]:
    """
    High-level function for GPU-accelerated factorization.
    
    Args:
        n: Number to factorize
        
    Returns:
        Tuple of (factor1, factor2, time_taken)
    """
    start_time = time.time()
    
    try:
        factorizer = CUDAFactorizer()
        f1, f2 = factorizer.factorize(n)
        
        if f1 and f2:
            # Ensure factors are ordered
            return min(f1, f2), max(f1, f2), time.time() - start_time
        
    except Exception as e:
        print(f"GPU factorization failed: {e}")
        print("Falling back to CPU implementation...")
    
    return None, None, time.time() - start_time


# Alternative: Use CuPy for GPU acceleration (if CUDA compilation fails)
try:
    import cupy as cp
    
    def cupy_pollard_rho(n: int, max_attempts: int = 1000) -> Optional[int]:
        """
        GPU-accelerated Pollard's rho using CuPy.
        """
        # Generate random starting points on GPU
        x = cp.random.randint(2, n-1, size=max_attempts, dtype=cp.uint64)
        c = cp.random.randint(1, n, size=max_attempts, dtype=cp.uint64)
        
        # Initialize arrays
        y = x.copy()
        d = cp.ones(max_attempts, dtype=cp.uint64)
        
        # Run iterations
        for _ in range(10000):
            # f(x) = (x^2 + c) mod n
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            
            # Compute GCD for each attempt
            diff = cp.abs(x - y)
            
            # Batch GCD computation
            for i in range(max_attempts):
                if d[i] == 1:
                    d[i] = int(cp.gcd(int(diff[i]), n))
                    
                    if 1 < d[i] < n:
                        return int(d[i])
        
        return None
    
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("CuPy not available. GPU acceleration limited to CUDA kernels.")


def benchmark_algorithms(n: int):
    """
    Benchmark different factorization approaches.
    """
    print(f"\n{'='*60}")
    print("ALGORITHM BENCHMARKING")
    print(f"{'='*60}")
    print(f"Target: {n}")
    
    results = []
    
    # CPU Pollard's rho
    from pollard_rho_brent import factorize_with_pollard_rho
    print("\n1. CPU Pollard's Rho (Brent)...")
    f1, f2, elapsed = factorize_with_pollard_rho(n)
    results.append(("CPU Pollard's Rho", f1, f2, elapsed))
    
    # CPU ECM
    from ecm_factorization import factorize_with_ecm
    print("\n2. CPU Elliptic Curve Method...")
    f1, f2, elapsed = factorize_with_ecm(n, max_curves=20)
    results.append(("CPU ECM", f1, f2, elapsed))
    
    # GPU CUDA
    try:
        print("\n3. GPU CUDA Kernels...")
        f1, f2, elapsed = gpu_accelerated_factorization(n)
        results.append(("GPU CUDA", f1, f2, elapsed))
    except:
        results.append(("GPU CUDA", None, None, -1))
    
    # GPU CuPy
    if HAS_CUPY:
        print("\n4. GPU CuPy...")
        start = time.time()
        factor = cupy_pollard_rho(n)
        if factor:
            f1, f2 = factor, n // factor
            elapsed = time.time() - start
            results.append(("GPU CuPy", f1, f2, elapsed))
        else:
            results.append(("GPU CuPy", None, None, time.time() - start))
    
    # Print results
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"{'Algorithm':<20} {'Success':<10} {'Time (s)':<15} {'Speedup':<10}")
    print("-" * 60)
    
    best_time = min(r[3] for r in results if r[3] > 0 and r[1] is not None)
    
    for algo, f1, f2, elapsed in results:
        if elapsed < 0:
            print(f"{algo:<20} {'FAILED':<10} {'N/A':<15} {'N/A':<10}")
        elif f1 and f2:
            speedup = best_time / elapsed if elapsed > 0 else 0
            print(f"{algo:<20} {'YES':<10} {elapsed:<15.4f} {speedup:<10.2f}x")
        else:
            print(f"{algo:<20} {'NO':<10} {elapsed:<15.4f} {'N/A':<10}")


if __name__ == "__main__":
    # Test with target number
    target = 2539123152460219
    
    # Run benchmarks
    benchmark_algorithms(target)