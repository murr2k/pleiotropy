use pleiotropy_rust_impl::{
    prime_factorization::PrimeFactorizer,
    prime_compute_backend::PrimeComputeBackend,
};

#[test]
fn test_target_number_cpu() {
    let factorizer = PrimeFactorizer::new();
    let target = 100822548703u64;
    
    let result = factorizer.factorize(target);
    
    assert!(result.verify(), "Factorization verification failed");
    assert_eq!(result.factors.len(), 2, "Expected 2 factors");
    assert!(result.factors.contains(&316907), "Missing factor 316907");
    assert!(result.factors.contains(&318089), "Missing factor 318089");
    
    // Check the product
    let product: u64 = result.factors.iter().product();
    assert_eq!(product, target, "Product doesn't match original number");
}

#[test]
fn test_target_number_unified_backend() {
    let mut backend = PrimeComputeBackend::new().unwrap();
    let target = 100822548703u64;
    
    let result = backend.factorize(target).unwrap();
    
    assert!(result.verify(), "Factorization verification failed");
    assert_eq!(result.factors.len(), 2, "Expected 2 factors");
    assert!(result.factors.contains(&316907), "Missing factor 316907");
    assert!(result.factors.contains(&318089), "Missing factor 318089");
}

#[test]
fn test_batch_factorization() {
    let mut backend = PrimeComputeBackend::new().unwrap();
    
    let test_numbers = vec![
        12,              // 2² × 3
        35,              // 5 × 7
        77,              // 7 × 11
        100822548703,    // 316907 × 318089
    ];
    
    let results = backend.factorize_batch(&test_numbers).unwrap();
    
    assert_eq!(results.len(), 4);
    
    // Verify each result
    assert_eq!(results[0].factors, vec![2, 2, 3]);
    assert_eq!(results[1].factors, vec![5, 7]);
    assert_eq!(results[2].factors, vec![7, 11]);
    assert!(results[3].factors.contains(&316907));
    assert!(results[3].factors.contains(&318089));
    
    // Verify all results
    for (i, result) in results.iter().enumerate() {
        assert!(result.verify(), "Factorization {} failed verification", i);
        assert_eq!(result.number, test_numbers[i]);
    }
}

#[test]
fn test_large_prime() {
    let factorizer = PrimeFactorizer::new();
    
    // Test with a known large prime
    let large_prime = 1000000007u64;
    let result = factorizer.factorize(large_prime);
    
    assert_eq!(result.factors, vec![large_prime]);
    assert!(result.verify());
}

#[test]
fn test_performance_comparison() {
    let mut backend = PrimeComputeBackend::new().unwrap();
    
    // Generate test data
    let test_numbers: Vec<u64> = (0..100)
        .map(|i| 1000000 + i * 1337)
        .collect();
    
    // Test with forced CPU
    backend.set_force_cpu(true);
    let cpu_results = backend.factorize_batch(&test_numbers).unwrap();
    
    // Test with GPU (if available)
    backend.set_force_cpu(false);
    let gpu_results = backend.factorize_batch(&test_numbers).unwrap();
    
    // Results should be identical
    assert_eq!(cpu_results.len(), gpu_results.len());
    for (cpu_res, gpu_res) in cpu_results.iter().zip(gpu_results.iter()) {
        assert_eq!(cpu_res.number, gpu_res.number);
        assert_eq!(cpu_res.factors, gpu_res.factors);
    }
    
    // Print stats
    let stats = backend.get_stats();
    println!("Performance test stats: {:?}", stats);
}