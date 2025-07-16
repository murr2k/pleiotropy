#!/usr/bin/env python3
"""
Run and analyze CUDA vs CPU benchmark for semiprime factorization
Target: 225012420229 = 336151 × 669379
"""

import subprocess
import sys
import time
import re
from pathlib import Path

def run_rust_benchmark():
    """Build and run the Rust benchmark"""
    print("=== Building and Running CUDA Benchmark ===\n")
    
    rust_dir = Path("rust_impl")
    
    # First, try to build with CUDA
    print("Building with CUDA support...")
    cuda_build = subprocess.run(
        ["cargo", "build", "--release", "--features", "cuda", "--bin", "benchmark_semiprime"],
        cwd=rust_dir,
        capture_output=True,
        text=True
    )
    
    if cuda_build.returncode != 0:
        print("⚠ CUDA build failed, trying CPU-only build...")
        print(f"Error: {cuda_build.stderr}")
        
        cpu_build = subprocess.run(
            ["cargo", "build", "--release", "--bin", "benchmark_semiprime"],
            cwd=rust_dir,
            capture_output=True,
            text=True
        )
        
        if cpu_build.returncode != 0:
            print(f"✗ Build failed: {cpu_build.stderr}")
            return None
        
        features = ""
    else:
        print("✓ CUDA build successful")
        features = "--features cuda"
    
    # Run the benchmark
    print("\nRunning benchmark...")
    result = subprocess.run(
        ["cargo", "run", "--release"] + (features.split() if features else []) + ["--bin", "benchmark_semiprime"],
        cwd=rust_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"✗ Benchmark failed: {result.stderr}")
        return None
    
    return result.stdout

def analyze_results(output):
    """Parse and analyze benchmark results"""
    if not output:
        return
    
    print("\n=== Benchmark Results Analysis ===\n")
    
    # Extract single number results
    trial_match = re.search(r"CPU Trial Division:.*?Average time: ([\d.]+)ms", output, re.DOTALL)
    pollard_match = re.search(r"CPU Pollard's Rho:.*?Average time: ([\d.]+)ms", output, re.DOTALL)
    cuda_single_match = re.search(r"CUDA Single Number:.*?Average time: ([\d.]+)ms", output, re.DOTALL)
    
    if trial_match:
        trial_time = float(trial_match.group(1))
        print(f"CPU Trial Division: {trial_time:.3f}ms")
    
    if pollard_match:
        pollard_time = float(pollard_match.group(1))
        print(f"CPU Pollard's Rho:  {pollard_time:.3f}ms")
    
    if cuda_single_match:
        cuda_time = float(cuda_single_match.group(1))
        print(f"CUDA Single:        {cuda_time:.3f}ms")
        
        if trial_match:
            speedup_trial = trial_time / cuda_time
            print(f"\nSpeedup vs Trial Division: {speedup_trial:.2f}x")
        
        if pollard_match:
            speedup_pollard = pollard_time / cuda_time
            print(f"Speedup vs Pollard's Rho:  {speedup_pollard:.2f}x")
    
    # Extract batch results
    print("\n--- Batch Performance ---")
    batch_pattern = r"Batch size: (\d+).*?CPU:\s+([\d.]+)ms.*?(?:CUDA:\s+([\d.]+)ms.*?Speedup:\s+([\d.]+)x)?"
    
    batch_results = []
    for match in re.finditer(batch_pattern, output, re.DOTALL):
        size = int(match.group(1))
        cpu_time = float(match.group(2))
        cuda_time = float(match.group(3)) if match.group(3) else None
        speedup = float(match.group(4)) if match.group(4) else None
        
        batch_results.append({
            'size': size,
            'cpu_time': cpu_time,
            'cuda_time': cuda_time,
            'speedup': speedup
        })
    
    if batch_results:
        print(f"\n{'Batch Size':<12} {'CPU (ms)':<12} {'CUDA (ms)':<12} {'Speedup':<10}")
        print("-" * 50)
        for r in batch_results:
            cuda_str = f"{r['cuda_time']:.3f}" if r['cuda_time'] else "N/A"
            speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
            print(f"{r['size']:<12} {r['cpu_time']:<12.3f} {cuda_str:<12} {speedup_str:<10}")
    
    # Performance summary
    print("\n--- Performance Summary ---")
    print(f"Target number: 225012420229 = 336151 × 669379")
    
    if cuda_single_match and trial_match:
        print(f"\nFor single number factorization:")
        print(f"  CPU best time:  {min(trial_time, pollard_time if pollard_match else trial_time):.3f}ms")
        print(f"  CUDA time:      {cuda_time:.3f}ms")
        print(f"  Best speedup:   {max(trial_time/cuda_time, pollard_time/cuda_time if pollard_match else 0):.2f}x")
    
    if batch_results and any(r['speedup'] for r in batch_results):
        max_speedup = max(r['speedup'] for r in batch_results if r['speedup'])
        best_batch = next(r for r in batch_results if r['speedup'] == max_speedup)
        print(f"\nFor batch processing:")
        print(f"  Best speedup:   {max_speedup:.2f}x (batch size {best_batch['size']})")
        print(f"  CPU time:       {best_batch['cpu_time']:.3f}ms")
        print(f"  CUDA time:      {best_batch['cuda_time']:.3f}ms")

def main():
    # Check if we're in the right directory
    if not Path("rust_impl").exists():
        print("Error: rust_impl directory not found. Please run from the project root.")
        sys.exit(1)
    
    # Run the benchmark
    output = run_rust_benchmark()
    
    if output:
        # Print raw output
        print("\n=== Raw Benchmark Output ===")
        print(output)
        
        # Analyze results
        analyze_results(output)
    else:
        print("\n✗ Benchmark execution failed")
        sys.exit(1)

if __name__ == "__main__":
    main()