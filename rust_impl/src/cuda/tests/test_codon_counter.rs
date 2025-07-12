use crate::cuda::{CudaDevice, kernels::CodonCounter};
use crate::types::{DnaSequence, CodonCounts};
use std::collections::HashMap;

fn create_test_sequence(id: &str, sequence: &str) -> DnaSequence {
    DnaSequence::new(id.to_string(), sequence.to_string())
}

#[test]
fn test_codon_counter_initialization() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let counter = CodonCounter::new(&device);
    assert!(counter.is_ok(), "Failed to initialize codon counter");
}

#[test]
fn test_single_sequence_counting() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let counter = CodonCounter::new(&device).expect("Failed to create counter");
    
    // Test sequence with known codon counts
    let sequence = create_test_sequence("test1", "ATGATGATG"); // 3x ATG
    let sequences = vec![sequence];
    
    let result = counter.count(&sequences).expect("Failed to count codons");
    
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].get("ATG"), Some(&3));
}

#[test]
fn test_multiple_sequences() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let counter = CodonCounter::new(&device).expect("Failed to create counter");
    
    let sequences = vec![
        create_test_sequence("seq1", "ATGATGATG"), // 3x ATG
        create_test_sequence("seq2", "GCGGCGGCG"), // 3x GCG
        create_test_sequence("seq3", "AAAAAAAAAA"), // 3x AAA + 1 A
    ];
    
    let results = counter.count(&sequences).expect("Failed to count codons");
    
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].get("ATG"), Some(&3));
    assert_eq!(results[1].get("GCG"), Some(&3));
    assert_eq!(results[2].get("AAA"), Some(&3));
}

#[test]
fn test_empty_sequence() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let counter = CodonCounter::new(&device).expect("Failed to create counter");
    
    let sequences = vec![];
    let results = counter.count(&sequences).expect("Failed to count codons");
    
    assert_eq!(results.len(), 0);
}

#[test]
fn test_mixed_case_sequences() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let counter = CodonCounter::new(&device).expect("Failed to create counter");
    
    let sequences = vec![
        create_test_sequence("mixed1", "ATGatgATG"), // Mixed case
        create_test_sequence("mixed2", "gcgGCGgcg"), // Mixed case
    ];
    
    let results = counter.count(&sequences).expect("Failed to count codons");
    
    assert_eq!(results[0].get("ATG"), Some(&3));
    assert_eq!(results[1].get("GCG"), Some(&3));
}

#[test]
fn test_performance_large_sequence() {
    use std::time::Instant;
    
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let counter = CodonCounter::new(&device).expect("Failed to create counter");
    
    // Create a large sequence (1MB)
    let large_seq = "ATGC".repeat(250_000);
    let sequences = vec![create_test_sequence("large", &large_seq)];
    
    let start = Instant::now();
    let results = counter.count(&sequences).expect("Failed to count codons");
    let duration = start.elapsed();
    
    println!("CUDA codon counting for 1MB sequence: {:?}", duration);
    assert!(!results.is_empty());
    assert!(duration.as_millis() < 100, "Performance requirement not met");
}

#[test]
fn test_correctness_all_codons() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let counter = CodonCounter::new(&device).expect("Failed to create counter");
    
    // Create sequence with all 64 codons
    let bases = ['A', 'C', 'G', 'T'];
    let mut all_codons = String::new();
    
    for &b1 in &bases {
        for &b2 in &bases {
            for &b3 in &bases {
                all_codons.push(b1);
                all_codons.push(b2);
                all_codons.push(b3);
            }
        }
    }
    
    let sequences = vec![create_test_sequence("all_codons", &all_codons)];
    let results = counter.count(&sequences).expect("Failed to count codons");
    
    // Each codon should appear exactly once
    let codon_count = results[0].values().filter(|&&count| count == 1).count();
    assert_eq!(codon_count, 64, "Should have all 64 codons");
}

#[test]
fn test_sliding_window_basic() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let counter = CodonCounter::new(&device).expect("Failed to create counter");
    
    // Test sequence: ATGATGATGATG (4x ATG)
    let sequence = create_test_sequence("test1", "ATGATGATGATG");
    let sequences = vec![sequence];
    
    // Window size 6, stride 3
    let results = counter.count_sliding_windows(&sequences, 6, 3)
        .expect("Failed to count with sliding windows");
    
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].len(), 3); // 3 windows
    
    // Each window should have 2 ATG codons
    for window in &results[0] {
        assert_eq!(window.get("ATG"), Some(&2));
    }
}

#[test]
fn test_sliding_window_overlap() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let counter = CodonCounter::new(&device).expect("Failed to create counter");
    
    // Test sequence with different codons
    let sequence = create_test_sequence("test1", "ATGCGCAAAAAA");
    let sequences = vec![sequence];
    
    // Window size 9, stride 3
    let results = counter.count_sliding_windows(&sequences, 9, 3)
        .expect("Failed to count with sliding windows");
    
    assert_eq!(results[0].len(), 2); // 2 windows
    
    // First window: ATGCGCAAA (ATG, CGC, AAA)
    assert_eq!(results[0][0].get("ATG"), Some(&1));
    assert_eq!(results[0][0].get("CGC"), Some(&1));
    assert_eq!(results[0][0].get("AAA"), Some(&1));
    
    // Second window: CGCAAAAAA (CGC, AAA, AAA)
    assert_eq!(results[0][1].get("CGC"), Some(&1));
    assert_eq!(results[0][1].get("AAA"), Some(&2));
}

#[test]
fn test_sliding_window_performance() {
    use std::time::Instant;
    
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let counter = CodonCounter::new(&device).expect("Failed to create counter");
    
    // Create 10 sequences of 100KB each
    let mut sequences = Vec::new();
    for i in 0..10 {
        let seq = "ATGC".repeat(25_000);
        sequences.push(create_test_sequence(&format!("seq{}", i), &seq));
    }
    
    let start = Instant::now();
    let results = counter.count_sliding_windows(&sequences, 10000, 5000)
        .expect("Failed to count with sliding windows");
    let duration = start.elapsed();
    
    println!("CUDA sliding window counting for 10x100KB sequences: {:?}", duration);
    assert_eq!(results.len(), 10);
    assert!(duration.as_millis() < 500, "Sliding window performance requirement not met");
}