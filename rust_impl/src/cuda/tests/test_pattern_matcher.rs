use super::*;
use crate::cuda::kernels::pattern_matcher::PatternMatcher;
use crate::cuda::device::CudaDevice;
use crate::types::{CudaFrequencyTable, TraitPattern, PatternMatch};
use std::collections::HashMap;

#[test]
fn test_pattern_matching_basic() {
    // Check if CUDA is available
    if !CudaDevice::is_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }
    
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let matcher = PatternMatcher::new(&device).expect("Failed to create pattern matcher");
    
    // Create test frequency table
    let mut global_frequencies = HashMap::new();
    let mut trait_frequencies = HashMap::new();
    
    // Set some test frequencies
    for i in 0..64 {
        let codon = index_to_codon(i);
        global_frequencies.insert(codon.clone(), 0.015625); // 1/64
    }
    
    // Create trait-specific patterns
    let mut carbon_metabolism = HashMap::new();
    carbon_metabolism.insert("CTG".to_string(), 0.05);
    carbon_metabolism.insert("GAA".to_string(), 0.05);
    carbon_metabolism.insert("AAA".to_string(), 0.05);
    carbon_metabolism.insert("CGT".to_string(), 0.05);
    trait_frequencies.insert("carbon_metabolism".to_string(), carbon_metabolism);
    
    let freq_table = CudaFrequencyTable {
        global_frequencies,
        trait_frequencies,
    };
    
    // Create trait patterns
    let trait_pattern = TraitPattern {
        trait_name: "carbon_metabolism".to_string(),
        preferred_codons: vec!["CTG".to_string(), "GAA".to_string(), "AAA".to_string(), "CGT".to_string()],
        avoided_codons: vec![],
        motifs: vec![],
        weight: 1.0,
        codon_preferences: {
            let mut prefs = HashMap::new();
            prefs.insert("CTG".to_string(), 1.0);
            prefs.insert("GAA".to_string(), 1.0);
            prefs.insert("AAA".to_string(), 1.0);
            prefs.insert("CGT".to_string(), 1.0);
            prefs
        },
        regulatory_patterns: vec![],
    };
    
    // Run pattern matching
    let matches = matcher.match_patterns(&freq_table, &[trait_pattern])
        .expect("Pattern matching failed");
    
    // Verify we got some matches
    assert!(!matches.is_empty(), "Expected at least one pattern match");
}

#[test]
fn test_sliding_window_pattern_matching() {
    if !CudaDevice::is_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }
    
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let matcher = PatternMatcher::new(&device).expect("Failed to create pattern matcher");
    
    // Create sequence frequencies for sliding window
    let seq_length = 1000; // 1000 codons
    let num_codons = 64;
    let mut sequence_frequencies = vec![0.015625f32; seq_length * num_codons];
    
    // Create peak in specific region
    for i in 300..400 {
        sequence_frequencies[i * num_codons + 12] = 0.1; // CTG
        sequence_frequencies[i * num_codons + 24] = 0.1; // GAA
    }
    
    // Create trait pattern
    let trait_pattern = TraitPattern {
        trait_name: "test_trait".to_string(),
        preferred_codons: vec!["CTG".to_string(), "GAA".to_string()],
        avoided_codons: vec![],
        motifs: vec![],
        weight: 1.0,
        codon_preferences: {
            let mut prefs = HashMap::new();
            prefs.insert("CTG".to_string(), 1.0);
            prefs.insert("GAA".to_string(), 1.0);
            prefs
        },
        regulatory_patterns: vec![],
    };
    
    // Run sliding window matching
    let window_size = 100;
    let window_stride = 50;
    
    let results = matcher.match_patterns_sliding_window(
        &sequence_frequencies,
        &[trait_pattern],
        window_size,
        window_stride,
        seq_length,
    ).expect("Sliding window matching failed");
    
    // Verify we detected the region
    assert!(!results.is_empty(), "Expected to find trait region");
    
    // Check that the detected position is in the expected range
    if let Some((trait_name, position, score)) = results.first() {
        assert_eq!(trait_name, "test_trait");
        assert!(position >= &250 && position <= &350, 
                "Expected position near 300, got {}", position);
        assert!(score > &0.7, "Expected high confidence score, got {}", score);
    }
}

#[test]
fn test_neurodna_pattern_integration() {
    if !CudaDevice::is_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }
    
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let matcher = PatternMatcher::new(&device).expect("Failed to create pattern matcher");
    
    // Create frequency table
    let mut global_frequencies = HashMap::new();
    for i in 0..64 {
        let codon = index_to_codon(i);
        global_frequencies.insert(codon.clone(), 0.015625);
    }
    
    let freq_table = CudaFrequencyTable {
        global_frequencies,
        trait_frequencies: HashMap::new(),
    };
    
    // Create NeuroDNA trait patterns
    let mut neurodna_traits = HashMap::new();
    neurodna_traits.insert(
        "carbon_metabolism".to_string(),
        vec!["CTG".to_string(), "GAA".to_string(), "AAA".to_string(), "CGT".to_string()]
    );
    neurodna_traits.insert(
        "stress_response".to_string(),
        vec!["GAA".to_string(), "GCT".to_string(), "ATT".to_string(), "CTG".to_string()]
    );
    
    // Run NeuroDNA pattern matching
    let matches = matcher.match_neurodna_patterns(&freq_table, &neurodna_traits)
        .expect("NeuroDNA pattern matching failed");
    
    // Verify results
    assert!(matches.len() >= 2, "Expected matches for both traits");
}

// Helper function
fn index_to_codon(index: usize) -> String {
    const BASES: [char; 4] = ['A', 'C', 'G', 'T'];
    let n1 = (index >> 4) & 0x3;
    let n2 = (index >> 2) & 0x3;
    let n3 = index & 0x3;
    
    format!("{}{}{}", BASES[n1], BASES[n2], BASES[n3])
}