# Prime Factorization Test Specifications

## Target Number Analysis

**Target**: 100822548703
**Factors**: 317213 × 317879
**Properties**: 
- Semi-prime (product of two 6-digit primes)
- Binary representation: 1011110011011111111001010100011111111 (38 bits)
- Decimal digits: 12

## Test Categories

### 1. Correctness Tests

```rust
#[test]
fn test_target_factorization_correctness() {
    let target = 100822548703u64;
    let expected_factors = vec![317213, 317879];
    
    // Test all algorithms
    let algorithms = vec![
        FactorizationAlgorithm::TrialDivision,
        FactorizationAlgorithm::PollardRho,
        FactorizationAlgorithm::ParallelTrialDivision,
    ];
    
    for algo in algorithms {
        let factorizer = CpuFactorizer::new(algo);
        let result = factorizer.factorize(target).unwrap();
        assert_eq!(result.factors, expected_factors);
        assert_eq!(result.factors[0] * result.factors[1], target);
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_factorization_correctness() {
    let target = 100822548703u64;
    let expected_factors = vec![317213, 317879];
    
    let mut cuda_factorizer = CudaFactorizer::new().unwrap();
    let result = cuda_factorizer.cuda_trial_division(target).unwrap();
    assert_eq!(result, expected_factors);
}
```

### 2. Performance Benchmarks

```rust
#[bench]
fn bench_cpu_trial_division(b: &mut Bencher) {
    let factorizer = CpuFactorizer::new(FactorizationAlgorithm::TrialDivision);
    b.iter(|| {
        factorizer.factorize(100822548703)
    });
}

#[bench]
fn bench_cpu_pollard_rho(b: &mut Bencher) {
    let factorizer = CpuFactorizer::new(FactorizationAlgorithm::PollardRho);
    b.iter(|| {
        factorizer.factorize(100822548703)
    });
}

#[bench]
#[cfg(feature = "cuda")]
fn bench_cuda_trial_division(b: &mut Bencher) {
    let mut cuda_factorizer = CudaFactorizer::new().unwrap();
    b.iter(|| {
        cuda_factorizer.cuda_trial_division(100822548703)
    });
}
```

### 3. Edge Cases

```rust
pub fn get_edge_case_suite() -> Vec<TestCase> {
    vec![
        // Smallest cases
        TestCase { input: 1, expected_factors: vec![1], category: TestCategory::EdgeCase },
        TestCase { input: 2, expected_factors: vec![2], category: TestCategory::SmallPrime },
        TestCase { input: 3, expected_factors: vec![3], category: TestCategory::SmallPrime },
        
        // Perfect squares
        TestCase { input: 4, expected_factors: vec![2, 2], category: TestCategory::Composite },
        TestCase { input: 9, expected_factors: vec![3, 3], category: TestCategory::Composite },
        TestCase { input: 121, expected_factors: vec![11, 11], category: TestCategory::Composite },
        
        // Mersenne primes
        TestCase { input: 31, expected_factors: vec![31], category: TestCategory::SmallPrime },
        TestCase { input: 127, expected_factors: vec![127], category: TestCategory::SmallPrime },
        TestCase { input: 8191, expected_factors: vec![8191], category: TestCategory::MediumPrime },
        
        // Large primes near target factors
        TestCase { input: 317207, expected_factors: vec![317207], category: TestCategory::LargePrime },
        TestCase { input: 317881, expected_factors: vec![317881], category: TestCategory::LargePrime },
        
        // Products of primes with varying sizes
        TestCase { input: 6, expected_factors: vec![2, 3], category: TestCategory::SemiPrime },
        TestCase { input: 15, expected_factors: vec![3, 5], category: TestCategory::SemiPrime },
        TestCase { input: 10001357, expected_factors: vec![2027, 4931], category: TestCategory::SemiPrime },
        
        // Target case
        TestCase { 
            input: 100822548703, 
            expected_factors: vec![317213, 317879], 
            category: TestCategory::TargetCase 
        },
    ]
}
```

### 4. Memory Usage Tests

```rust
#[test]
fn test_memory_bounded_execution() {
    let factorizer = CpuFactorizer::new(FactorizationAlgorithm::TrialDivision);
    let initial_memory = get_current_memory_usage();
    
    let _result = factorizer.factorize(100822548703).unwrap();
    
    let peak_memory = get_peak_memory_usage();
    let memory_increase = peak_memory - initial_memory;
    
    // Assert reasonable memory bounds (< 10MB for CPU implementation)
    assert!(memory_increase < 10 * 1024 * 1024, 
            "Memory usage {} exceeds limit", memory_increase);
}

#[test]
#[cfg(feature = "cuda")]
fn test_gpu_memory_usage() {
    let mut cuda_factorizer = CudaFactorizer::new().unwrap();
    let gpu_memory_before = cuda_factorizer.get_gpu_memory_usage();
    
    let _result = cuda_factorizer.cuda_trial_division(100822548703).unwrap();
    
    let gpu_memory_after = cuda_factorizer.get_gpu_memory_usage();
    let gpu_memory_used = gpu_memory_after - gpu_memory_before;
    
    // Assert GPU memory bounds (< 100MB)
    assert!(gpu_memory_used < 100 * 1024 * 1024,
            "GPU memory usage {} exceeds limit", gpu_memory_used);
}
```

### 5. Stress Tests

```rust
#[test]
fn test_batch_factorization() {
    // Generate 1000 semi-primes
    let test_numbers: Vec<u64> = (0..1000)
        .map(|i| {
            let p1 = 100000 + i * 7;
            let p2 = 200000 + i * 11;
            p1 * p2
        })
        .collect();
    
    let start = Instant::now();
    let factorizer = CpuFactorizer::new(FactorizationAlgorithm::ParallelTrialDivision);
    
    for &n in &test_numbers {
        let result = factorizer.factorize(n).unwrap();
        assert_eq!(result.factors.len(), 2);
    }
    
    let duration = start.elapsed();
    println!("Batch factorization of 1000 numbers: {:?}", duration);
    assert!(duration.as_secs() < 60, "Batch processing too slow");
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_batch_processing() {
    let test_numbers: Vec<u64> = (0..1000)
        .map(|i| 100000007 + i * 1000)
        .collect();
    
    let mut cuda_factorizer = CudaFactorizer::new().unwrap();
    let start = Instant::now();
    
    let results = cuda_factorizer.batch_factorize(&test_numbers).unwrap();
    
    let duration = start.elapsed();
    assert_eq!(results.len(), test_numbers.len());
    println!("CUDA batch factorization: {:?}", duration);
    assert!(duration.as_millis() < 1000, "CUDA batch too slow");
}
```

### 6. Algorithm Comparison Tests

```rust
#[test]
fn test_algorithm_performance_comparison() {
    let target = 100822548703u64;
    let mut results = HashMap::new();
    
    // Trial Division
    let td_start = Instant::now();
    let td_factorizer = CpuFactorizer::new(FactorizationAlgorithm::TrialDivision);
    let td_result = td_factorizer.factorize(target).unwrap();
    results.insert("trial_division", td_start.elapsed());
    
    // Pollard's Rho
    let pr_start = Instant::now();
    let pr_factorizer = CpuFactorizer::new(FactorizationAlgorithm::PollardRho);
    let pr_result = pr_factorizer.factorize(target).unwrap();
    results.insert("pollard_rho", pr_start.elapsed());
    
    // Verify all produce same result
    assert_eq!(td_result.factors, pr_result.factors);
    
    // Pollard's Rho should be faster
    assert!(results["pollard_rho"] < results["trial_division"]);
}
```

### 7. Regression Test Suite

```rust
pub struct RegressionTestSuite {
    baseline_performances: HashMap<String, Duration>,
    tolerance_factor: f64, // Allow 20% performance variation
}

impl RegressionTestSuite {
    pub fn run_regression_tests(&self) -> Result<()> {
        let test_cases = vec![
            ("small_prime", 97u64),
            ("medium_prime", 1000003u64),
            ("large_semiprime", 100822548703u64),
        ];
        
        for (name, number) in test_cases {
            let start = Instant::now();
            let factorizer = CpuFactorizer::new(FactorizationAlgorithm::PollardRho);
            let _result = factorizer.factorize(number)?;
            let duration = start.elapsed();
            
            if let Some(baseline) = self.baseline_performances.get(name) {
                let ratio = duration.as_secs_f64() / baseline.as_secs_f64();
                assert!(ratio < 1.0 + self.tolerance_factor,
                        "Performance regression for {}: {:.2}x slower", name, ratio);
            }
        }
        Ok(())
    }
}
```

## Test Execution Plan

### Phase 1: Unit Tests
1. Run all correctness tests
2. Verify edge cases
3. Check memory bounds

### Phase 2: Performance Tests
1. Warm up (10 iterations)
2. Measure CPU algorithms (100 iterations each)
3. Measure GPU algorithms (100 iterations each)
4. Calculate statistics (mean, std dev, min, max)

### Phase 3: Comparison Analysis
1. Calculate speedup factors
2. Generate performance graphs
3. Identify bottlenecks
4. Document optimization opportunities

## Success Criteria

1. **Correctness**: All algorithms must correctly factorize 100822548703
2. **CPU Performance**: < 1 second for trial division
3. **GPU Performance**: > 10x speedup over CPU
4. **Memory Usage**: < 100MB GPU, < 10MB CPU
5. **Accuracy**: 100% accuracy on all test cases
6. **Stability**: < 5% variation across runs