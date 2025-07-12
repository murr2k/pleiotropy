use super::*;
use crate::cuda::device::CudaDevice;
use crate::cuda::kernels::{
    codon_counter::CodonCounter,
    frequency_calculator::FrequencyCalculator,
    pattern_matcher::PatternMatcher,
    matrix_processor::MatrixProcessor,
};
use crate::types::{Sequence, TraitInfo, PleiotropicGene};
use std::collections::HashMap;

#[test]
fn test_full_cuda_pipeline() {
    if !CudaDevice::is_available() {
        eprintln!("CUDA not available, skipping integration test");
        return;
    }
    
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    
    // Initialize all CUDA kernels
    let codon_counter = CodonCounter::new(&device).expect("Failed to create codon counter");
    let freq_calculator = FrequencyCalculator::new(&device).expect("Failed to create frequency calculator");
    let pattern_matcher = PatternMatcher::new(&device).expect("Failed to create pattern matcher");
    let matrix_processor = MatrixProcessor::new(&device).expect("Failed to create matrix processor");
    
    // Create synthetic test sequences
    let sequences = vec![
        Sequence {
            id: "seq1".to_string(),
            sequence: "ATGCTGGAAAAAACGTTTCGCTATTGGTCAA".to_string(), // Contains carbon metabolism codons
            metadata: HashMap::new(),
        },
        Sequence {
            id: "seq2".to_string(),
            sequence: "GAAGCTATTCTGAACTTCGGTCAACGTGAA".to_string(), // Contains stress response codons
            metadata: HashMap::new(),
        },
        Sequence {
            id: "seq3".to_string(),
            sequence: "CGTAACTTTGAAACCATTTCTTCTGAAATT".to_string(), // Contains regulatory codons
            metadata: HashMap::new(),
        },
    ];
    
    // Define trait information
    let traits = vec![
        TraitInfo {
            name: "carbon_metabolism".to_string(),
            codon_preferences: {
                let mut prefs = HashMap::new();
                prefs.insert("CTG".to_string(), 0.8);
                prefs.insert("GAA".to_string(), 0.8);
                prefs.insert("AAA".to_string(), 0.8);
                prefs.insert("CGT".to_string(), 0.8);
                prefs
            },
        },
        TraitInfo {
            name: "stress_response".to_string(),
            codon_preferences: {
                let mut prefs = HashMap::new();
                prefs.insert("GAA".to_string(), 0.7);
                prefs.insert("GCT".to_string(), 0.7);
                prefs.insert("ATT".to_string(), 0.7);
                prefs.insert("CTG".to_string(), 0.7);
                prefs
            },
        },
    ];
    
    // Step 1: Count codons
    println!("Step 1: Counting codons...");
    let codon_counts = codon_counter.count(&sequences)
        .expect("Codon counting failed");
    
    assert_eq!(codon_counts.len(), sequences.len());
    
    // Step 2: Calculate frequencies
    println!("Step 2: Calculating frequencies...");
    let frequency_table = freq_calculator.calculate(&codon_counts, &traits)
        .expect("Frequency calculation failed");
    
    assert!(!frequency_table.global_frequencies.is_empty());
    
    // Step 3: Pattern matching with NeuroDNA traits
    println!("Step 3: Pattern matching...");
    let mut neurodna_traits = HashMap::new();
    neurodna_traits.insert(
        "carbon_metabolism".to_string(),
        vec!["CTG".to_string(), "GAA".to_string(), "AAA".to_string(), "CGT".to_string()]
    );
    neurodna_traits.insert(
        "stress_response".to_string(),
        vec!["GAA".to_string(), "GCT".to_string(), "ATT".to_string(), "CTG".to_string()]
    );
    
    let pattern_matches = pattern_matcher.match_neurodna_patterns(&frequency_table, &neurodna_traits)
        .expect("Pattern matching failed");
    
    println!("Found {} pattern matches", pattern_matches.len());
    
    // Step 4: Matrix analysis for trait separation
    println!("Step 4: Eigenanalysis for trait separation...");
    
    // Create codon frequency matrix
    let num_sequences = sequences.len();
    let num_codons = 64;
    let mut freq_matrix = vec![0.0f32; num_sequences * num_codons];
    
    for (seq_idx, counts) in codon_counts.iter().enumerate() {
        let total: usize = counts.values().sum();
        for i in 0..64 {
            let codon = index_to_codon(i);
            let count = counts.get(&codon).copied().unwrap_or(0);
            freq_matrix[seq_idx * num_codons + i] = count as f32 / total as f32;
        }
    }
    
    // Identify trait components
    let components = matrix_processor.identify_trait_components(
        &freq_matrix,
        num_sequences,
        num_codons,
        0.95
    ).expect("Component identification failed");
    
    println!("Identified {} principal components explaining 95% variance", components.len());
    
    // Verify results
    assert!(!components.is_empty(), "Should identify at least one component");
    
    // Print summary
    println!("\n=== CUDA Pipeline Summary ===");
    println!("Processed {} sequences", sequences.len());
    println!("Analyzed {} traits", traits.len());
    println!("Found {} pattern matches", pattern_matches.len());
    println!("Identified {} principal components", components.len());
    
    for (idx, variance, _) in &components {
        println!("  Component {}: {:.2}% variance", idx, variance * 100.0);
    }
}

#[test]
fn test_sliding_window_integration() {
    if !CudaDevice::is_available() {
        eprintln!("CUDA not available, skipping integration test");
        return;
    }
    
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    
    let codon_counter = CodonCounter::new(&device).expect("Failed to create codon counter");
    let pattern_matcher = PatternMatcher::new(&device).expect("Failed to create pattern matcher");
    
    // Create a longer sequence with trait regions
    let mut long_sequence = String::new();
    
    // Random background
    for _ in 0..300 {
        long_sequence.push_str("ATG");
    }
    
    // Carbon metabolism region
    for _ in 0..50 {
        long_sequence.push_str("CTGGAAAAAACGT"); // High in carbon metabolism codons
    }
    
    // Random middle
    for _ in 0..200 {
        long_sequence.push_str("GCT");
    }
    
    // Stress response region
    for _ in 0..50 {
        long_sequence.push_str("GAAGCTATTCTG"); // High in stress response codons
    }
    
    // Random end
    for _ in 0..100 {
        long_sequence.push_str("TTC");
    }
    
    let sequence = Sequence {
        id: "long_seq".to_string(),
        sequence: long_sequence,
        metadata: HashMap::new(),
    };
    
    // Count codons using sliding windows
    let window_size = 300;
    let window_stride = 100;
    
    let window_counts = codon_counter.count_sliding_windows(
        &[sequence.clone()],
        window_size,
        window_stride
    ).expect("Sliding window counting failed");
    
    assert_eq!(window_counts.len(), 1); // One sequence
    let seq_windows = &window_counts[0];
    
    println!("Generated {} windows", seq_windows.len());
    
    // Convert to frequency matrix for pattern matching
    let num_windows = seq_windows.len();
    let num_codons = 64;
    let mut window_frequencies = vec![0.0f32; num_windows * num_codons];
    
    for (window_idx, counts) in seq_windows.iter().enumerate() {
        let total: usize = counts.values().sum();
        if total > 0 {
            for i in 0..64 {
                let codon = index_to_codon(i);
                let count = counts.get(&codon).copied().unwrap_or(0);
                window_frequencies[window_idx * num_codons + i] = count as f32 / total as f32;
            }
        }
    }
    
    // Create trait patterns
    let trait_patterns = vec![
        create_trait_pattern("carbon_metabolism", vec!["CTG", "GAA", "AAA", "CGT"]),
        create_trait_pattern("stress_response", vec!["GAA", "GCT", "ATT", "CTG"]),
    ];
    
    // Find trait regions
    let trait_regions = pattern_matcher.match_patterns_sliding_window(
        &window_frequencies,
        &trait_patterns,
        1, // Window already defined above
        1,
        num_windows,
    ).expect("Sliding window pattern matching failed");
    
    println!("\nDetected {} trait regions:", trait_regions.len());
    for (trait_name, position, score) in &trait_regions {
        println!("  {} at window {} (score: {:.3})", trait_name, position, score);
    }
    
    // Should detect both trait regions
    assert!(trait_regions.len() >= 2, "Expected to detect at least 2 trait regions");
}

// Helper functions
fn index_to_codon(index: usize) -> String {
    const BASES: [char; 4] = ['A', 'C', 'G', 'T'];
    let n1 = (index >> 4) & 0x3;
    let n2 = (index >> 2) & 0x3;
    let n3 = index & 0x3;
    
    format!("{}{}{}", BASES[n1], BASES[n2], BASES[n3])
}

fn create_trait_pattern(name: &str, preferred: Vec<&str>) -> crate::types::TraitPattern {
    let mut codon_preferences = HashMap::new();
    
    // Set high preference for specified codons
    for codon in &preferred {
        codon_preferences.insert(codon.to_string(), 1.0);
    }
    
    // Set low preference for others
    for i in 0..64 {
        let codon = index_to_codon(i);
        codon_preferences.entry(codon).or_insert(0.1);
    }
    
    crate::types::TraitPattern {
        trait_name: name.to_string(),
        preferred_codons: preferred.iter().map(|s| s.to_string()).collect(),
        avoided_codons: vec![],
        motifs: vec![],
        weight: 1.0,
        codon_preferences,
        regulatory_patterns: vec![],
    }
}