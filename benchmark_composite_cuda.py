#!/usr/bin/env python3
"""
CUDA Composite Number Factorizer Benchmark
Demonstrates the performance of GPU-accelerated composite number factorization
"""

import subprocess
import json
import time
import random
import math
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np

def generate_test_numbers() -> Dict[str, List[int]]:
    """Generate various types of composite numbers for testing"""
    
    test_numbers = {
        "small_composites": [
            24, 100, 128, 243, 720, 1001, 2310, 5040
        ],
        "semiprimes": [
            15, 21, 35, 77, 143, 391, 1517, 4189, 5767, 9797
        ],
        "powers_of_primes": [
            32, 81, 125, 343, 1024, 2187, 3125, 16807
        ],
        "highly_composite": [
            720, 1260, 5040, 10080, 25200, 50400, 100800, 720720
        ],
        "large_composites": [
            123456, 234567, 345678, 456789, 567890,
            1234567, 2345678, 3456789, 4567890, 5678901
        ],
        "rsa_like": [
            # Products of two primes of similar size
            100822548703,  # 317567 × 317569
            1000000007 * 1000000009,
            2147483647 * 2147483659,
        ]
    }
    
    return test_numbers

def run_cuda_factorization(number: int) -> Tuple[List[int], float]:
    """Run CUDA factorization and return factors and time"""
    
    # Create a simple Rust program to test factorization
    rust_code = f"""
use pleiotropy::cuda::composite_factorizer::factorize_composite_cuda;
use std::time::Instant;

fn main() {{
    if !pleiotropy::cuda::cuda_available() {{
        eprintln!("CUDA not available");
        std::process::exit(1);
    }}
    
    let n = {number}u64;
    let start = Instant::now();
    
    match factorize_composite_cuda(n) {{
        Ok(factors) => {{
            let elapsed = start.elapsed();
            println!("{{}}", serde_json::json!({{
                "number": n,
                "factors": factors,
                "time_ms": elapsed.as_secs_f64() * 1000.0,
                "success": true
            }}));
        }}
        Err(e) => {{
            eprintln!("Factorization failed: {{}}", e);
            std::process::exit(1);
        }}
    }}
}}
"""
    
    # Write temporary Rust file
    with open("/tmp/test_cuda_factor.rs", "w") as f:
        f.write(rust_code)
    
    # Compile and run
    try:
        # Compile
        subprocess.run([
            "rustc", 
            "/tmp/test_cuda_factor.rs",
            "-L", "rust_impl/target/release/deps",
            "-o", "/tmp/test_cuda_factor",
            "--edition", "2021"
        ], check=True, capture_output=True)
        
        # Run
        result = subprocess.run(
            ["/tmp/test_cuda_factor"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            output = json.loads(result.stdout)
            return output["factors"], output["time_ms"]
        else:
            print(f"Error factoring {number}: {result.stderr}")
            return [], 0.0
            
    except Exception as e:
        print(f"Failed to run factorization for {number}: {e}")
        return [], 0.0

def verify_factorization(n: int, factors: List[int]) -> bool:
    """Verify that the factorization is correct"""
    if not factors:
        return n <= 1
    
    product = 1
    for f in factors:
        product *= f
    
    return product == n

def analyze_performance(results: Dict[str, List[Dict]]) -> None:
    """Analyze and visualize performance results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("CUDA Composite Number Factorization Performance", fontsize=16)
    
    # 1. Time by number size
    ax1 = axes[0, 0]
    all_numbers = []
    all_times = []
    all_types = []
    
    for num_type, data in results.items():
        numbers = [d["number"] for d in data]
        times = [d["time_ms"] for d in data]
        all_numbers.extend(numbers)
        all_times.extend(times)
        all_types.extend([num_type] * len(numbers))
        
        ax1.scatter(numbers, times, label=num_type.replace("_", " ").title(), alpha=0.7)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel("Number")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Factorization Time vs Number Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Time by number of factors
    ax2 = axes[0, 1]
    factor_counts = []
    times_by_factors = []
    
    for num_type, data in results.items():
        for d in data:
            if d["factors"]:
                factor_counts.append(len(d["factors"]))
                times_by_factors.append(d["time_ms"])
    
    ax2.scatter(factor_counts, times_by_factors, alpha=0.6)
    ax2.set_xlabel("Number of Prime Factors")
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("Factorization Time vs Factor Count")
    ax2.grid(True, alpha=0.3)
    
    # 3. Success rate by type
    ax3 = axes[1, 0]
    success_rates = {}
    for num_type, data in results.items():
        success_count = sum(1 for d in data if d["success"])
        success_rates[num_type] = (success_count / len(data)) * 100 if data else 0
    
    types = list(success_rates.keys())
    rates = list(success_rates.values())
    bars = ax3.bar(range(len(types)), rates)
    ax3.set_xticks(range(len(types)))
    ax3.set_xticklabels([t.replace("_", " ").title() for t in types], rotation=45, ha='right')
    ax3.set_ylabel("Success Rate (%)")
    ax3.set_title("Factorization Success Rate by Number Type")
    ax3.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    # 4. Average time by type
    ax4 = axes[1, 1]
    avg_times = {}
    for num_type, data in results.items():
        times = [d["time_ms"] for d in data if d["success"]]
        avg_times[num_type] = np.mean(times) if times else 0
    
    types = list(avg_times.keys())
    times = list(avg_times.values())
    bars = ax4.bar(range(len(types)), times)
    ax4.set_xticks(range(len(types)))
    ax4.set_xticklabels([t.replace("_", " ").title() for t in types], rotation=45, ha='right')
    ax4.set_ylabel("Average Time (ms)")
    ax4.set_title("Average Factorization Time by Number Type")
    
    # Add value labels
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{time:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("cuda_composite_factorization_benchmark.png", dpi=150)
    print("\nBenchmark visualization saved to: cuda_composite_factorization_benchmark.png")

def main():
    """Run comprehensive CUDA composite factorization benchmark"""
    
    print("=== CUDA Composite Number Factorization Benchmark ===\n")
    
    # Check if CUDA is available
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    except:
        print("PyTorch not available for CUDA info, continuing with benchmark...\n")
    
    # Generate test numbers
    test_numbers = generate_test_numbers()
    
    # Run benchmarks
    results = {}
    total_numbers = sum(len(nums) for nums in test_numbers.values())
    current = 0
    
    print(f"Testing {total_numbers} composite numbers...\n")
    
    for num_type, numbers in test_numbers.items():
        print(f"\n{num_type.replace('_', ' ').title()}:")
        print("-" * 50)
        
        type_results = []
        
        for n in numbers:
            current += 1
            print(f"[{current}/{total_numbers}] Factoring {n}...", end=" ")
            
            start_time = time.time()
            factors, time_ms = run_cuda_factorization(n)
            
            if factors:
                success = verify_factorization(n, factors)
                if success:
                    print(f"✓ {n} = {' × '.join(map(str, factors))} ({time_ms:.2f} ms)")
                else:
                    print(f"✗ Invalid factorization!")
            else:
                success = False
                print(f"✗ Failed")
            
            type_results.append({
                "number": n,
                "factors": factors,
                "time_ms": time_ms,
                "success": success
            })
        
        results[num_type] = type_results
    
    # Print summary statistics
    print("\n\n=== Summary Statistics ===")
    print("-" * 50)
    
    total_success = 0
    total_time = 0
    
    for num_type, data in results.items():
        success_count = sum(1 for d in data if d["success"])
        total_success += success_count
        
        times = [d["time_ms"] for d in data if d["success"]]
        if times:
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            total_time += sum(times)
            
            print(f"\n{num_type.replace('_', ' ').title()}:")
            print(f"  Success rate: {success_count}/{len(data)} ({success_count/len(data)*100:.1f}%)")
            print(f"  Avg time: {avg_time:.2f} ms")
            print(f"  Min time: {min_time:.2f} ms")
            print(f"  Max time: {max_time:.2f} ms")
    
    print(f"\nOverall success rate: {total_success}/{total_numbers} ({total_success/total_numbers*100:.1f}%)")
    print(f"Total computation time: {total_time:.2f} ms")
    
    # Create visualizations
    analyze_performance(results)
    
    # Save detailed results
    with open("cuda_composite_factorization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nDetailed results saved to: cuda_composite_factorization_results.json")

if __name__ == "__main__":
    main()