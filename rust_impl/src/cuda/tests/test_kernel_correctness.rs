/// Comprehensive correctness tests for CUDA kernels
/// Ensures that GPU results exactly match CPU results

use crate::cuda::{CudaDevice, CudaAccelerator, kernels::*};
use crate::types::{DnaSequence, CodonCounts, TraitInfo};
use crate::FrequencyAnalyzer;
use std::collections::HashMap;
use approx::assert_relative_eq;

const EPSILON: f64 = 1e-6;

/// Helper to generate random DNA sequences
fn generate_random_sequence(id: &str, length: usize, seed: u64) -> DnaSequence {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    
    let bases = ['A', 'C', 'G', 'T'];
    let mut rng = StdRng::seed_from_u64(seed);
    
    let sequence: String = (0..length)
        .map(|_| bases[rng.gen_range(0..4)])
        .collect();
    
    DnaSequence::new(id.to_string(), sequence)
}

/// Test trait for deterministic testing
fn create_test_traits() -> Vec<TraitInfo> {
    vec![
        TraitInfo {
            name: "carbon_metabolism".to_string(),
            description: "Carbon metabolism pathways".to_string(),
            associated_genes: vec!["gene1".to_string(), "gene2".to_string()],
            known_sequences: vec!["ATGATG".to_string(), "GCGGCG".to_string()],
        },
        TraitInfo {
            name: "stress_response".to_string(),
            description: "Stress response mechanisms".to_string(),
            associated_genes: vec!["gene3".to_string()],
            known_sequences: vec!["AAATTT".to_string()],
        },
    ]
}

#[test]
fn test_codon_counting_exact_match() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let counter = CodonCounter::new(&device).expect("Failed to create counter");
    let analyzer = FrequencyAnalyzer::new();
    
    // Test various sequence sizes
    for (size, seed) in [(100, 42), (1000, 123), (10000, 456), (100000, 789)].iter() {
        let sequences = vec![generate_random_sequence("test", *size, *seed)];
        
        // CPU counting
        let cpu_counts = analyzer.count_codons_cpu(&sequences)
            .expect("CPU counting failed");
        
        // GPU counting
        let gpu_counts = counter.count(&sequences)
            .expect("GPU counting failed");
        
        // Verify exact match
        assert_eq!(cpu_counts.len(), gpu_counts.len(), "Count vectors length mismatch");
        
        for (cpu_map, gpu_map) in cpu_counts.iter().zip(gpu_counts.iter()) {
            assert_eq!(cpu_map.len(), gpu_map.len(), "Codon count size mismatch");
            
            for (codon, cpu_count) in cpu_map {
                let gpu_count = gpu_map.get(codon).unwrap_or(&0);
                assert_eq!(cpu_count, gpu_count, 
                    "Count mismatch for codon {} (size {}): CPU={}, GPU={}", 
                    codon, size, cpu_count, gpu_count);
            }
        }
    }
}

#[test]
fn test_frequency_calculation_exact_match() {
    let mut cuda_acc = CudaAccelerator::new().expect("Failed to create CUDA accelerator");
    let mut analyzer = FrequencyAnalyzer::new();
    
    // Generate test data
    let sequences = (0..20)
        .map(|i| generate_random_sequence(&format!("seq{}", i), 3000, i as u64))
        .collect::<Vec<_>>();
    
    let traits = create_test_traits();
    
    // Count codons on GPU first
    let gpu_codon_counts = cuda_acc.count_codons(&sequences)
        .expect("GPU codon counting failed");
    
    // CPU frequency calculation
    let cpu_codon_counts = analyzer.count_codons_cpu(&sequences)
        .expect("CPU codon counting failed");
    let cpu_freq_table = analyzer.calculate_frequencies_cpu(&cpu_codon_counts, &traits)
        .expect("CPU frequency calculation failed");
    
    // GPU frequency calculation
    let gpu_freq_table = cuda_acc.calculate_frequencies(&gpu_codon_counts, &traits)
        .expect("GPU frequency calculation failed");
    
    // Compare global frequencies
    for (cpu_codon, cpu_freq) in &cpu_freq_table.codon_frequencies {
        let gpu_freq = gpu_freq_table.codon_frequencies.iter()
            .find(|(codon, _)| codon == &cpu_codon.codon)
            .map(|(_, freq)| *freq)
            .unwrap_or(0.0);
        
        assert_relative_eq!(
            cpu_codon.global_frequency as f32,
            gpu_freq,
            epsilon = EPSILON as f32,
            "Frequency mismatch for codon {}: CPU={}, GPU={}",
            cpu_codon.codon,
            cpu_codon.global_frequency,
            gpu_freq
        );
    }
}

#[test]
fn test_sliding_window_correctness() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let counter = CodonCounter::new(&device).expect("Failed to create counter");
    
    // Create test sequence with known pattern
    let pattern = "ATGATGATGATGATGATGATGATGATGATG"; // 10x ATG
    let sequence = DnaSequence::new("test".to_string(), pattern.to_string());
    
    let window_size = 9;  // 3 codons
    let stride = 3;       // 1 codon stride
    
    // Count with sliding windows
    let windows = counter.count_sliding_windows(&[sequence], window_size, stride)
        .expect("Sliding window counting failed");
    
    // Verify window counts
    assert_eq!(windows.len(), 1);
    let seq_windows = &windows[0];
    
    // Expected: 8 windows, each with 3 ATG codons
    assert_eq!(seq_windows.len(), 8);
    for window_counts in seq_windows {
        assert_eq!(window_counts.get("ATG"), Some(&3), "Each window should have exactly 3 ATG codons");
    }
}

#[test]
fn test_edge_cases() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let counter = CodonCounter::new(&device).expect("Failed to create counter");
    
    // Test 1: Empty sequence
    let empty_seq = DnaSequence::new("empty".to_string(), "".to_string());
    let result = counter.count(&[empty_seq]).expect("Should handle empty sequence");
    assert_eq!(result[0].len(), 0);
    
    // Test 2: Sequence not divisible by 3
    let partial_seq = DnaSequence::new("partial".to_string(), "ATGATGA".to_string()); // 7 bases
    let result = counter.count(&[partial_seq]).expect("Should handle partial codon");
    assert_eq!(result[0].get("ATG"), Some(&2));
    
    // Test 3: Non-standard characters
    let invalid_seq = DnaSequence::new("invalid".to_string(), "ATGNNNXYZ".to_string());
    let result = counter.count(&[invalid_seq]).expect("Should handle invalid chars");
    assert_eq!(result[0].get("ATG"), Some(&1));
    assert_eq!(result[0].get("NNN"), None); // Invalid codons should be skipped
}

#[test]
fn test_large_batch_processing() {
    let mut cuda_acc = CudaAccelerator::new().expect("Failed to create CUDA accelerator");
    let analyzer = FrequencyAnalyzer::new();
    
    // Create large batch of sequences
    let sequences: Vec<_> = (0..1000)
        .map(|i| generate_random_sequence(&format!("seq{}", i), 300, i as u64))
        .collect();
    
    // Process on CPU
    let cpu_counts = analyzer.count_codons_cpu(&sequences)
        .expect("CPU batch processing failed");
    
    // Process on GPU
    let gpu_counts = cuda_acc.count_codons(&sequences)
        .expect("GPU batch processing failed");
    
    // Verify all sequences processed
    assert_eq!(cpu_counts.len(), gpu_counts.len(), "Batch size mismatch");
    
    // Spot check some sequences
    for i in (0..1000).step_by(100) {
        let cpu_total: u32 = cpu_counts[i].values().sum();
        let gpu_total: u32 = gpu_counts[i].values().sum();
        assert_eq!(cpu_total, gpu_total, "Total codon count mismatch for sequence {}", i);
    }
}

#[test]
fn test_pattern_matching_accuracy() {
    use crate::types::TraitPattern;
    
    let mut cuda_acc = CudaAccelerator::new().expect("Failed to create CUDA accelerator");
    
    // Create sequences with known patterns
    let sequences = vec![
        DnaSequence::new("pattern1".to_string(), 
            "ATGATGATGATGGCGGCGGCG".to_string()), // High ATG and GCG
        DnaSequence::new("pattern2".to_string(), 
            "AAAAAAAAATTTTTTTTT".to_string()), // High AAA and TTT
    ];
    
    // Define trait patterns
    let patterns = vec![
        TraitPattern {
            name: "atg_rich".to_string(),
            codon_preferences: vec![("ATG".to_string(), 2.0)],
            min_score: 0.6,
        },
        TraitPattern {
            name: "at_rich".to_string(),
            codon_preferences: vec![("AAA".to_string(), 1.5), ("TTT".to_string(), 1.5)],
            min_score: 0.5,
        },
    ];
    
    // Count codons and calculate frequencies
    let codon_counts = cuda_acc.count_codons(&sequences)
        .expect("Codon counting failed");
    let freq_table = cuda_acc.calculate_frequencies(&codon_counts, &create_test_traits())
        .expect("Frequency calculation failed");
    
    // Match patterns
    let matches = cuda_acc.match_patterns(&freq_table, &patterns)
        .expect("Pattern matching failed");
    
    // Verify pattern detection
    assert!(!matches.is_empty(), "Should find pattern matches");
    
    // Check that high-confidence matches are found
    let high_conf_matches: Vec<_> = matches.iter()
        .filter(|m| m.score > 0.7)
        .collect();
    assert!(!high_conf_matches.is_empty(), "Should find high-confidence matches");
}

#[test] 
fn test_matrix_operations_precision() {
    let mut cuda_acc = CudaAccelerator::new().expect("Failed to create CUDA accelerator");
    
    // Create symmetric positive definite matrix for eigenanalysis
    let size = 64;
    let mut matrix = vec![0.0f32; size * size];
    
    // Create a simple test matrix with known properties
    for i in 0..size {
        for j in 0..size {
            if i == j {
                matrix[i * size + j] = 2.0 + (i as f32) * 0.1; // Diagonal dominance
            } else {
                matrix[i * size + j] = 0.1 * ((i + j) as f32).sin().abs();
            }
        }
    }
    
    // GPU eigenanalysis
    let (gpu_eigenvalues, gpu_eigenvectors) = cuda_acc.eigenanalysis(&matrix, size)
        .expect("GPU eigenanalysis failed");
    
    // Basic validation
    assert_eq!(gpu_eigenvalues.len(), size);
    assert_eq!(gpu_eigenvectors.len(), size * size);
    
    // Check eigenvalues are positive (for positive definite matrix)
    for eigenval in &gpu_eigenvalues {
        assert!(*eigenval > 0.0, "Eigenvalue should be positive: {}", eigenval);
    }
    
    // Check eigenvalues are sorted in descending order
    for i in 1..gpu_eigenvalues.len() {
        assert!(gpu_eigenvalues[i-1] >= gpu_eigenvalues[i], 
            "Eigenvalues should be sorted: {} < {}", 
            gpu_eigenvalues[i-1], gpu_eigenvalues[i]);
    }
}

#[test]
fn test_memory_safety() {
    let mut cuda_acc = CudaAccelerator::new().expect("Failed to create CUDA accelerator");
    
    // Test with various challenging sizes
    let test_sizes = vec![
        1,        // Minimum
        63,       // Not aligned to warp size
        1024,     // Power of 2
        1000000,  // Large
        999999,   // Large and odd
    ];
    
    for size in test_sizes {
        let sequence = generate_random_sequence("mem_test", size, 42);
        let result = cuda_acc.count_codons(&[sequence]);
        assert!(result.is_ok(), "Failed to process sequence of size {}", size);
    }
}

#[test]
fn test_concurrent_kernel_execution() {
    use std::thread;
    use std::sync::Arc;
    
    // Test that multiple CUDA accelerators can work concurrently
    let handles: Vec<_> = (0..4)
        .map(|i| {
            thread::spawn(move || {
                let mut cuda_acc = CudaAccelerator::new()
                    .expect("Failed to create CUDA accelerator");
                
                let sequences = vec![generate_random_sequence(&format!("thread{}", i), 10000, i as u64)];
                let counts = cuda_acc.count_codons(&sequences)
                    .expect("Codon counting failed in thread");
                
                // Verify we got results
                assert!(!counts[0].is_empty());
            })
        })
        .collect();
    
    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}