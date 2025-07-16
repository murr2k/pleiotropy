//! Benchmark harness for prime factorization performance monitoring
//!
//! Run with: cargo bench --bench prime_factorization_bench

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use pleiotropy_rust::prime_factorization::{cpu, factorize};
use std::time::Duration;

/// Benchmark data structure
struct BenchmarkCase {
    number: u64,
    name: &'static str,
}

impl BenchmarkCase {
    const fn new(number: u64, name: &'static str) -> Self {
        Self { number, name }
    }
}

/// Get benchmark cases organized by size
fn get_benchmark_cases() -> Vec<BenchmarkCase> {
    vec![
        // Small numbers (< 1000)
        BenchmarkCase::new(12, "small_12"),
        BenchmarkCase::new(97, "small_prime_97"),
        BenchmarkCase::new(256, "small_power_256"),
        
        // Medium numbers (1000 - 1M)
        BenchmarkCase::new(10007, "medium_prime_10007"),
        BenchmarkCase::new(65537, "medium_fermat_65537"),
        BenchmarkCase::new(1000000, "medium_million"),
        
        // Large numbers (> 1M)
        BenchmarkCase::new(1299709, "large_prime_1299709"),
        BenchmarkCase::new(100822548703, "large_required_100822548703"),
        BenchmarkCase::new(2147483647, "large_mersenne_2147483647"),
    ]
}

/// Benchmark CPU implementation
fn bench_cpu_factorization(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_factorization");
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));
    
    for case in get_benchmark_cases() {
        group.bench_with_input(
            BenchmarkId::new("cpu", case.name),
            &case.number,
            |b, &number| {
                b.iter(|| {
                    cpu::factorize(black_box(number))
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark CUDA implementation (if available)
#[cfg(feature = "cuda")]
fn bench_cuda_factorization(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda_factorization");
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));
    
    for case in get_benchmark_cases() {
        group.bench_with_input(
            BenchmarkId::new("cuda", case.name),
            &case.number,
            |b, &number| {
                b.iter(|| {
                    match factorize(black_box(number)) {
                        Ok(result) if result.used_cuda => result,
                        _ => panic!("CUDA not available for benchmark"),
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark CPU vs CUDA comparison
#[cfg(feature = "cuda")]
fn bench_cpu_vs_cuda_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_vs_cuda");
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));
    
    // Focus on larger numbers where CUDA should show benefit
    let comparison_cases = vec![
        BenchmarkCase::new(1000000, "million"),
        BenchmarkCase::new(100822548703, "required_case"),
        BenchmarkCase::new(2147483647, "mersenne_prime"),
    ];
    
    for case in comparison_cases {
        // CPU benchmark
        group.bench_with_input(
            BenchmarkId::new("cpu", case.name),
            &case.number,
            |b, &number| {
                b.iter(|| {
                    cpu::factorize(black_box(number))
                });
            },
        );
        
        // CUDA benchmark
        group.bench_with_input(
            BenchmarkId::new("cuda", case.name),
            &case.number,
            |b, &number| {
                b.iter(|| {
                    match factorize(black_box(number)) {
                        Ok(result) if result.used_cuda => result,
                        _ => panic!("CUDA not available for benchmark"),
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory allocation overhead
fn bench_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_overhead");
    
    // Test repeated factorizations to measure allocation overhead
    group.bench_function("repeated_small", |b| {
        b.iter(|| {
            for i in 0..100 {
                cpu::factorize(black_box(12 + i));
            }
        });
    });
    
    group.bench_function("repeated_large", |b| {
        b.iter(|| {
            for i in 0..10 {
                cpu::factorize(black_box(100822548703 + i * 2));
            }
        });
    });
    
    group.finish();
}

/// Benchmark worst-case scenarios
fn bench_worst_case(c: &mut Criterion) {
    let mut group = c.benchmark_group("worst_case");
    group.measurement_time(Duration::from_secs(30)); // Longer time for worst cases
    
    // Prime numbers are worst case for trial division
    let worst_cases = vec![
        BenchmarkCase::new(1299709, "large_prime"),
        BenchmarkCase::new(15485863, "8_digit_prime"),
        BenchmarkCase::new(2147483647, "mersenne_prime"),
    ];
    
    for case in worst_cases {
        group.bench_with_input(
            BenchmarkId::new("worst_case", case.name),
            &case.number,
            |b, &number| {
                b.iter(|| {
                    cpu::factorize(black_box(number))
                });
            },
        );
    }
    
    group.finish();
}

// Configure criterion groups
#[cfg(not(feature = "cuda"))]
criterion_group!(
    benches,
    bench_cpu_factorization,
    bench_memory_overhead,
    bench_worst_case
);

#[cfg(feature = "cuda")]
criterion_group!(
    benches,
    bench_cpu_factorization,
    bench_cuda_factorization,
    bench_cpu_vs_cuda_comparison,
    bench_memory_overhead,
    bench_worst_case
);

criterion_main!(benches);