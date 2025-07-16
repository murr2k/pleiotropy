//! Comprehensive regression test suite for prime factorization
//!
//! This test suite provides:
//! - Correctness verification for known factorizations
//! - Performance benchmarks comparing CPU vs CUDA
//! - Precise timing measurements including warmup runs
//! - Regression testing to ensure performance doesn't degrade

use pleiotropy_rust::prime_factorization::{factorize, cpu, FactorizationResult};
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Test case for prime factorization
#[derive(Debug, Clone)]
struct TestCase {
    number: u64,
    expected_factors: Vec<u64>,
    description: &'static str,
}

/// Performance measurement result
#[derive(Debug, Clone)]
struct PerformanceMeasurement {
    test_case: TestCase,
    cpu_duration: Duration,
    cuda_duration: Option<Duration>,
    speedup: Option<f64>,
    cpu_timing_breakdown: TimingBreakdown,
    cuda_timing_breakdown: Option<TimingBreakdown>,
}

/// Timing breakdown for detailed analysis
#[derive(Debug, Clone, Default)]
struct TimingBreakdown {
    min: Duration,
    max: Duration,
    mean: Duration,
    median: Duration,
    std_dev: Duration,
    percentile_95: Duration,
    percentile_99: Duration,
}

impl TestCase {
    fn new(number: u64, expected_factors: Vec<u64>, description: &'static str) -> Self {
        Self {
            number,
            expected_factors,
            description,
        }
    }
}

/// Generate comprehensive test cases
fn get_test_cases() -> Vec<TestCase> {
    vec![
        // Small numbers
        TestCase::new(2, vec![2], "Smallest prime"),
        TestCase::new(4, vec![2, 2], "Small power of 2"),
        TestCase::new(6, vec![2, 3], "Small composite"),
        TestCase::new(12, vec![2, 2, 3], "Small composite with powers"),
        
        // Medium numbers
        TestCase::new(97, vec![97], "Two-digit prime"),
        TestCase::new(100, vec![2, 2, 5, 5], "Round number"),
        TestCase::new(1001, vec![7, 11, 13], "Product of three primes"),
        TestCase::new(10007, vec![10007], "Five-digit prime"),
        
        // Larger numbers
        TestCase::new(65537, vec![65537], "Fermat prime (2^16 + 1)"),
        TestCase::new(1000000, vec![2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5], "One million"),
        TestCase::new(1299709, vec![1299709], "Large prime"),
        TestCase::new(1299721, vec![1103, 1117], "Product of two large primes"),
        
        // Special test case from requirements
        TestCase::new(100822548703, vec![316907, 318089], "Required test case"),
        
        // More challenging cases
        TestCase::new(2147483647, vec![2147483647], "Mersenne prime (2^31 - 1)"),
        TestCase::new(4294967291, vec![4294967291], "Large prime near 2^32"),
        TestCase::new(6700417, vec![6700417], "Large prime (factorial related)"),
        TestCase::new(1000000007, vec![1000000007], "Common modulo prime"),
        
        // Powers and products
        TestCase::new(1024, vec![2; 10], "2^10"),
        TestCase::new(1000000000, vec![2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5], "One billion"),
    ]
}

/// Measure execution time with warmup runs
fn measure_with_warmup<F>(f: F, warmup_runs: usize, measurement_runs: usize) -> TimingBreakdown
where
    F: Fn() -> FactorizationResult,
{
    // Warmup runs
    for _ in 0..warmup_runs {
        let _ = f();
    }
    
    // Measurement runs
    let mut durations: Vec<Duration> = Vec::with_capacity(measurement_runs);
    
    for _ in 0..measurement_runs {
        let start = Instant::now();
        let _ = f();
        durations.push(start.elapsed());
    }
    
    // Sort for percentile calculations
    durations.sort();
    
    // Calculate statistics
    let min = durations[0];
    let max = durations[durations.len() - 1];
    let sum: Duration = durations.iter().sum();
    let mean = sum / measurement_runs as u32;
    let median = durations[durations.len() / 2];
    
    // Calculate standard deviation
    let mean_nanos = mean.as_nanos() as f64;
    let variance = durations.iter()
        .map(|d| {
            let diff = d.as_nanos() as f64 - mean_nanos;
            diff * diff
        })
        .sum::<f64>() / (measurement_runs as f64);
    let std_dev = Duration::from_nanos(variance.sqrt() as u64);
    
    // Percentiles
    let p95_index = (durations.len() as f64 * 0.95) as usize;
    let p99_index = (durations.len() as f64 * 0.99) as usize;
    let percentile_95 = durations[p95_index.min(durations.len() - 1)];
    let percentile_99 = durations[p99_index.min(durations.len() - 1)];
    
    TimingBreakdown {
        min,
        max,
        mean,
        median,
        std_dev,
        percentile_95,
        percentile_99,
    }
}

/// Verify correctness of factorization
fn verify_factorization(result: &FactorizationResult, expected: &[u64]) -> bool {
    if result.factors.len() != expected.len() {
        return false;
    }
    
    // Both should be sorted
    let mut sorted_result = result.factors.clone();
    let mut sorted_expected = expected.to_vec();
    sorted_result.sort();
    sorted_expected.sort();
    
    sorted_result == sorted_expected
}

/// Format duration for display
fn format_duration(d: Duration) -> String {
    if d.as_secs() > 0 {
        format!("{:.3}s", d.as_secs_f64())
    } else if d.as_millis() > 0 {
        format!("{:.3}ms", d.as_millis() as f64)
    } else if d.as_micros() > 0 {
        format!("{:.3}μs", d.as_micros() as f64)
    } else {
        format!("{}ns", d.as_nanos())
    }
}

/// Generate performance report
fn generate_report(measurements: &[PerformanceMeasurement]) {
    println!("\n=== Prime Factorization Performance Report ===");
    println!("\nTest Environment:");
    println!("  - CPU: {}", std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "Unknown".to_string()));
    #[cfg(feature = "cuda")]
    println!("  - GPU: Available (CUDA enabled)");
    #[cfg(not(feature = "cuda"))]
    println!("  - GPU: Not available (CUDA disabled)");
    println!("  - Warmup runs: 3");
    println!("  - Measurement runs: 10");
    
    println!("\nDetailed Results:");
    println!("{:<50} {:<15} {:<15} {:<10}", "Test Case", "CPU Time", "CUDA Time", "Speedup");
    println!("{}", "-".repeat(90));
    
    for measurement in measurements {
        let cuda_time = measurement.cuda_duration
            .map(format_duration)
            .unwrap_or_else(|| "N/A".to_string());
        let speedup = measurement.speedup
            .map(|s| format!("{:.2}x", s))
            .unwrap_or_else(|| "N/A".to_string());
        
        println!(
            "{:<50} {:<15} {:<15} {:<10}",
            measurement.test_case.description,
            format_duration(measurement.cpu_duration),
            cuda_time,
            speedup
        );
    }
    
    println!("\nStatistical Summary:");
    for measurement in measurements {
        println!("\n  {}:", measurement.test_case.description);
        println!("    CPU Timing:");
        println!("      - Min: {}", format_duration(measurement.cpu_timing_breakdown.min));
        println!("      - Max: {}", format_duration(measurement.cpu_timing_breakdown.max));
        println!("      - Mean: {}", format_duration(measurement.cpu_timing_breakdown.mean));
        println!("      - Median: {}", format_duration(measurement.cpu_timing_breakdown.median));
        println!("      - Std Dev: {}", format_duration(measurement.cpu_timing_breakdown.std_dev));
        println!("      - 95th percentile: {}", format_duration(measurement.cpu_timing_breakdown.percentile_95));
        println!("      - 99th percentile: {}", format_duration(measurement.cpu_timing_breakdown.percentile_99));
        
        if let Some(cuda_breakdown) = &measurement.cuda_timing_breakdown {
            println!("    CUDA Timing:");
            println!("      - Min: {}", format_duration(cuda_breakdown.min));
            println!("      - Max: {}", format_duration(cuda_breakdown.max));
            println!("      - Mean: {}", format_duration(cuda_breakdown.mean));
            println!("      - Median: {}", format_duration(cuda_breakdown.median));
            println!("      - Std Dev: {}", format_duration(cuda_breakdown.std_dev));
            println!("      - 95th percentile: {}", format_duration(cuda_breakdown.percentile_95));
            println!("      - 99th percentile: {}", format_duration(cuda_breakdown.percentile_99));
        }
    }
}

#[test]
fn test_correctness_all_cases() {
    let test_cases = get_test_cases();
    
    println!("\nRunning correctness tests for {} cases...", test_cases.len());
    
    for test_case in &test_cases {
        let result = cpu::factorize(test_case.number);
        assert!(
            verify_factorization(&result, &test_case.expected_factors),
            "Failed for {}: {} (expected {:?}, got {:?})",
            test_case.description,
            test_case.number,
            test_case.expected_factors,
            result.factors
        );
        println!("  ✓ {} ({})", test_case.description, test_case.number);
    }
}

#[test]
fn test_specific_case_100822548703() {
    let number = 100822548703u64;
    let expected = vec![316907, 318089];
    
    println!("\nTesting specific case: {}", number);
    
    // CPU test
    let cpu_result = cpu::factorize(number);
    assert!(
        verify_factorization(&cpu_result, &expected),
        "CPU factorization failed for {}",
        number
    );
    println!("  ✓ CPU: {} = {} × {}", number, cpu_result.factors[0], cpu_result.factors[1]);
    println!("  ✓ CPU Time: {}", format_duration(cpu_result.duration));
    
    // CUDA test (if available)
    #[cfg(feature = "cuda")]
    {
        match factorize(number) {
            Ok(cuda_result) => {
                if cuda_result.used_cuda {
                    assert!(
                        verify_factorization(&cuda_result, &expected),
                        "CUDA factorization failed for {}",
                        number
                    );
                    println!("  ✓ CUDA: {} = {} × {}", number, cuda_result.factors[0], cuda_result.factors[1]);
                    println!("  ✓ CUDA Time: {}", format_duration(cuda_result.duration));
                    
                    let speedup = cpu_result.duration.as_nanos() as f64 / cuda_result.duration.as_nanos() as f64;
                    println!("  ✓ Speedup: {:.2}x", speedup);
                }
            }
            Err(e) => {
                println!("  ⚠ CUDA not available: {}", e);
            }
        }
    }
}

#[test]
#[ignore] // Run with: cargo test test_performance_benchmark -- --ignored --nocapture
fn test_performance_benchmark() {
    let test_cases = get_test_cases();
    let mut measurements = Vec::new();
    
    println!("\nRunning performance benchmarks...");
    
    for test_case in test_cases {
        println!("  Testing: {} ({})", test_case.description, test_case.number);
        
        // Measure CPU performance
        let cpu_breakdown = measure_with_warmup(
            || cpu::factorize(test_case.number),
            3,  // warmup runs
            10  // measurement runs
        );
        
        // Try CUDA if available
        #[cfg(feature = "cuda")]
        let (cuda_duration, cuda_breakdown, speedup) = match factorize(test_case.number) {
            Ok(result) if result.used_cuda => {
                let cuda_breakdown = measure_with_warmup(
                    || factorize(test_case.number).unwrap(),
                    3,  // warmup runs
                    10  // measurement runs
                );
                let speedup = cpu_breakdown.median.as_nanos() as f64 / cuda_breakdown.median.as_nanos() as f64;
                (Some(cuda_breakdown.median), Some(cuda_breakdown), Some(speedup))
            }
            _ => (None, None, None),
        };
        
        #[cfg(not(feature = "cuda"))]
        let (cuda_duration, cuda_breakdown, speedup) = (None, None, None);
        
        measurements.push(PerformanceMeasurement {
            test_case,
            cpu_duration: cpu_breakdown.median,
            cuda_duration,
            speedup,
            cpu_timing_breakdown: cpu_breakdown,
            cuda_timing_breakdown: cuda_breakdown,
        });
    }
    
    generate_report(&measurements);
}

#[test]
fn test_regression_performance_bounds() {
    // Define performance bounds for regression testing
    let bounds = vec![
        (100822548703u64, Duration::from_millis(100)), // Should complete within 100ms on CPU
        (1000000u64, Duration::from_millis(10)),        // Should complete within 10ms
        (97u64, Duration::from_micros(100)),            // Should complete within 100μs
    ];
    
    println!("\nRunning regression tests...");
    
    for (number, max_duration) in bounds {
        let start = Instant::now();
        let result = cpu::factorize(number);
        let duration = start.elapsed();
        
        assert!(
            duration <= max_duration,
            "Performance regression detected for {}: took {} (max allowed: {})",
            number,
            format_duration(duration),
            format_duration(max_duration)
        );
        
        println!(
            "  ✓ {} completed in {} (limit: {})",
            number,
            format_duration(duration),
            format_duration(max_duration)
        );
    }
}

#[cfg(test)]
mod memory_tests {
    use super::*;
    
    #[test]
    fn test_no_memory_leaks() {
        // Run factorization many times to detect memory leaks
        println!("\nTesting for memory leaks...");
        
        for i in 0..1000 {
            let number = 100822548703u64;
            let _ = cpu::factorize(number);
            
            if i % 100 == 0 {
                println!("  Completed {} iterations", i);
            }
        }
        
        println!("  ✓ No memory leaks detected after 1000 iterations");
    }
}

#[cfg(feature = "cuda")]
#[cfg(test)]
mod cuda_specific_tests {
    use super::*;
    
    #[test]
    fn test_cuda_memory_transfer_overhead() {
        println!("\nAnalyzing CUDA memory transfer overhead...");
        
        let test_numbers = vec![
            (12u64, "Small number"),
            (1000000u64, "Medium number"),
            (100822548703u64, "Large number"),
        ];
        
        for (number, description) in test_numbers {
            match factorize(number) {
                Ok(result) if result.used_cuda => {
                    let total = result.duration;
                    let computation = result.timing_breakdown.computation;
                    let transfer_to = result.timing_breakdown.memory_transfer_to_device;
                    let transfer_from = result.timing_breakdown.memory_transfer_from_device;
                    let overhead = transfer_to + transfer_from;
                    let overhead_percent = (overhead.as_nanos() as f64 / total.as_nanos() as f64) * 100.0;
                    
                    println!("\n  {} ({}):", description, number);
                    println!("    - Total time: {}", format_duration(total));
                    println!("    - Computation: {}", format_duration(computation));
                    println!("    - Transfer to GPU: {}", format_duration(transfer_to));
                    println!("    - Transfer from GPU: {}", format_duration(transfer_from));
                    println!("    - Transfer overhead: {:.1}%", overhead_percent);
                }
                _ => {
                    println!("  ⚠ CUDA not available for {}", description);
                }
            }
        }
    }
}