/// Comprehensive tests for CUDA composite number factorizer
use pleiotropy::cuda::composite_factorizer::*;
use std::time::Instant;

#[test]
fn test_composite_type_classification() {
    if !pleiotropy::cuda::cuda_available() {
        println!("Skipping CUDA test - no GPU available");
        return;
    }
    
    let device = pleiotropy::cuda::CudaDevice::new(0).unwrap();
    let factorizer = CudaCompositeFactorizer::new(&device).unwrap();
    
    // Test cases with known classifications
    let test_cases = vec![
        (32, CompositeType::PowerOfPrime),      // 2^5
        (81, CompositeType::PowerOfPrime),      // 3^4
        (720, CompositeType::HighlyComposite),  // 2^4 * 3^2 * 5
        (1260, CompositeType::HighlyComposite), // 2^2 * 3^2 * 5 * 7
        (143, CompositeType::GeneralComposite), // 11 * 13
        (391, CompositeType::GeneralComposite), // 17 * 23
    ];
    
    for (n, expected_type) in test_cases {
        let classified = factorizer.classify_composite(n);
        assert_eq!(classified, expected_type, "Failed to classify {}", n);
    }
}

#[test]
fn test_fermat_factorization_perfect_squares() {
    if !pleiotropy::cuda::cuda_available() {
        println!("Skipping CUDA test - no GPU available");
        return;
    }
    
    let device = pleiotropy::cuda::CudaDevice::new(0).unwrap();
    let factorizer = CudaCompositeFactorizer::new(&device).unwrap();
    
    // Numbers that are products of primes close together
    let test_cases = vec![
        (403, 13, 31),     // Close factors
        (1517, 37, 41),    // Close factors
        (4189, 59, 71),    // Close factors
        (5767, 53, 109),   // Medium distance
    ];
    
    for (n, expected_f1, expected_f2) in test_cases {
        let start = Instant::now();
        let result = factorizer.fermat_factorize(n).unwrap();
        let elapsed = start.elapsed();
        
        assert!(result.is_some(), "Failed to factor {} with Fermat's method", n);
        let (f1, f2) = result.unwrap();
        
        // Verify factors
        assert_eq!(f1 * f2, n);
        assert!(f1 == expected_f1 || f1 == expected_f2);
        assert!(f2 == expected_f1 || f2 == expected_f2);
        
        println!("Fermat factored {} = {} × {} in {:?}", n, f1, f2, elapsed);
    }
}

#[test]
fn test_pollard_rho_various_composites() {
    if !pleiotropy::cuda::cuda_available() {
        println!("Skipping CUDA test - no GPU available");
        return;
    }
    
    let device = pleiotropy::cuda::CudaDevice::new(0).unwrap();
    let factorizer = CudaCompositeFactorizer::new(&device).unwrap();
    
    // Various composite numbers
    let test_cases = vec![
        8051,      // 83 × 97
        455459,    // 607 × 751
        1299709,   // 1021 × 1273
        2741311,   // 1327 × 2063
    ];
    
    for n in test_cases {
        let start = Instant::now();
        let result = factorizer.pollard_rho_factorize(n).unwrap();
        let elapsed = start.elapsed();
        
        if let Some((f1, f2)) = result {
            assert_eq!(f1 * f2, n, "Invalid factorization of {}", n);
            println!("Pollard's rho factored {} = {} × {} in {:?}", n, f1, f2, elapsed);
        } else {
            panic!("Failed to factor {} with Pollard's rho", n);
        }
    }
}

#[test]
fn test_full_factorization_various_types() {
    if !pleiotropy::cuda::cuda_available() {
        println!("Skipping CUDA test - no GPU available");
        return;
    }
    
    // Test comprehensive factorization
    let test_cases = vec![
        (24, vec![2, 2, 2, 3]),
        (100, vec![2, 2, 5, 5]),
        (128, vec![2, 2, 2, 2, 2, 2, 2]), // 2^7
        (243, vec![3, 3, 3, 3, 3]),       // 3^5
        (720, vec![2, 2, 2, 2, 3, 3, 5]), // Highly composite
        (1001, vec![7, 11, 13]),
        (2310, vec![2, 3, 5, 7, 11]),     // Product of first 5 primes
        (9797, vec![97, 101]),             // Semiprime
        (123456, vec![2, 2, 2, 2, 2, 2, 3, 643]),
    ];
    
    for (n, expected) in test_cases {
        let start = Instant::now();
        let mut factors = factorize_composite_cuda(n).unwrap();
        let elapsed = start.elapsed();
        
        factors.sort();
        assert_eq!(factors, expected, "Failed to factor {}", n);
        
        // Verify product
        let product: u64 = factors.iter().product();
        assert_eq!(product, n);
        
        println!("Factored {} = {:?} in {:?}", n, factors, elapsed);
    }
}

#[test]
fn test_large_composite_factorization() {
    if !pleiotropy::cuda::cuda_available() {
        println!("Skipping CUDA test - no GPU available");
        return;
    }
    
    // Test larger composite numbers
    let test_cases = vec![
        100822548703u64, // Large semiprime from original tests
        1234567890,      // General composite
        9876543210,      // General composite
    ];
    
    for n in test_cases {
        let start = Instant::now();
        let factors = factorize_composite_cuda(n).unwrap();
        let elapsed = start.elapsed();
        
        // Verify factorization
        let product: u64 = factors.iter().product();
        assert_eq!(product, n, "Invalid factorization of {}", n);
        
        println!("Large factorization: {} = {:?} in {:?}", n, factors, elapsed);
    }
}

#[test]
fn test_power_of_prime_factorization() {
    if !pleiotropy::cuda::cuda_available() {
        println!("Skipping CUDA test - no GPU available");
        return;
    }
    
    // Test powers of primes
    let test_cases = vec![
        (32, vec![2; 5]),       // 2^5
        (81, vec![3; 4]),       // 3^4
        (125, vec![5; 3]),      // 5^3
        (343, vec![7; 3]),      // 7^3
        (1024, vec![2; 10]),    // 2^10
        (2187, vec![3; 7]),     // 3^7
    ];
    
    for (n, expected) in test_cases {
        let factors = factorize_composite_cuda(n).unwrap();
        assert_eq!(factors, expected, "Failed to factor power of prime {}", n);
    }
}

#[test]
fn test_concurrent_factorizations() {
    if !pleiotropy::cuda::cuda_available() {
        println!("Skipping CUDA test - no GPU available");
        return;
    }
    
    use std::sync::Arc;
    use std::thread;
    
    // Test concurrent access to GPU
    let numbers = vec![
        1001, 1517, 4189, 5767, 8051, 9797,
        123456, 234567, 345678, 456789,
    ];
    
    let start = Instant::now();
    let handles: Vec<_> = numbers.into_iter()
        .map(|n| {
            thread::spawn(move || {
                let factors = factorize_composite_cuda(n).unwrap();
                let product: u64 = factors.iter().product();
                assert_eq!(product, n);
                (n, factors)
            })
        })
        .collect();
    
    for handle in handles {
        let (n, factors) = handle.join().unwrap();
        println!("Thread factored {} = {:?}", n, factors);
    }
    
    println!("Concurrent factorizations completed in {:?}", start.elapsed());
}

#[test]
fn test_edge_cases() {
    if !pleiotropy::cuda::cuda_available() {
        println!("Skipping CUDA test - no GPU available");
        return;
    }
    
    // Test edge cases
    assert_eq!(factorize_composite_cuda(0).unwrap(), vec![]);
    assert_eq!(factorize_composite_cuda(1).unwrap(), vec![]);
    assert_eq!(factorize_composite_cuda(2).unwrap(), vec![2]);
    assert_eq!(factorize_composite_cuda(3).unwrap(), vec![3]);
    assert_eq!(factorize_composite_cuda(4).unwrap(), vec![2, 2]);
}

#[test]
#[ignore] // Run with --ignored for performance benchmarks
fn benchmark_factorization_methods() {
    if !pleiotropy::cuda::cuda_available() {
        println!("Skipping CUDA benchmark - no GPU available");
        return;
    }
    
    let device = pleiotropy::cuda::CudaDevice::new(0).unwrap();
    let factorizer = CudaCompositeFactorizer::new(&device).unwrap();
    
    println!("\n=== CUDA Composite Factorization Benchmarks ===\n");
    
    // Benchmark different number types
    let benchmarks = vec![
        ("Small composite", 24u64),
        ("Medium composite", 123456),
        ("Large semiprime", 100822548703),
        ("Power of prime", 1024),
        ("Highly composite", 720720), // 2^4 * 3^2 * 5 * 7 * 11 * 13
    ];
    
    for (name, n) in benchmarks {
        let start = Instant::now();
        let factors = factorizer.factorize(n).unwrap();
        let elapsed = start.elapsed();
        
        println!("{}: {} = {:?}", name, n, factors);
        println!("  Time: {:?}", elapsed);
        println!("  Type: {:?}", factorizer.classify_composite(n));
        println!();
    }
}