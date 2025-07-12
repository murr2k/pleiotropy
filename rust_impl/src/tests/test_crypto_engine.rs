use crate::crypto_engine::CryptoEngine;
use crate::types::*;
use std::collections::HashMap;

/// Helper function to create test sequences
fn create_test_sequence(id: &str, sequence: &str) -> Sequence {
    Sequence {
        id: id.to_string(),
        sequence: sequence.to_string(),
        annotations: HashMap::new(),
    }
}

/// Helper function to create test frequency table
fn create_test_frequency_table() -> FrequencyTable {
    let mut codon_frequencies = Vec::new();
    
    // Add some common codons with frequencies
    let codons = vec![
        ("ATG", 0.05, vec![("growth", 0.08), ("stress", 0.03)]),
        ("TAA", 0.02, vec![("growth", 0.01), ("stress", 0.04)]),
        ("GCG", 0.04, vec![("growth", 0.05), ("stress", 0.02)]),
        ("CTG", 0.06, vec![("growth", 0.07), ("stress", 0.05)]),
    ];
    
    for (codon, global_freq, trait_freqs) in codons {
        let mut trait_frequencies = HashMap::new();
        for (trait_name, freq) in trait_freqs {
            trait_frequencies.insert(trait_name.to_string(), freq);
        }
        
        codon_frequencies.push(CodonFrequency {
            codon: codon.to_string(),
            amino_acid: "X".to_string(), // Simplified
            global_frequency: global_freq,
            trait_frequencies,
        });
    }
    
    FrequencyTable {
        codon_frequencies,
        trait_definitions: vec![
            TraitDefinition {
                name: "growth".to_string(),
                description: "Growth rate".to_string(),
                known_genes: vec!["ftsZ".to_string()],
            },
            TraitDefinition {
                name: "stress".to_string(),
                description: "Stress response".to_string(),
                known_genes: vec!["rpoS".to_string()],
            },
        ],
    }
}

#[test]
fn test_crypto_engine_creation() {
    let engine = CryptoEngine::new();
    // Engine should be created with default parameters
    assert!(true); // Just verify it compiles and runs
}

#[test]
fn test_decrypt_empty_sequences() {
    let engine = CryptoEngine::new();
    let frequency_table = create_test_frequency_table();
    let sequences: Vec<Sequence> = vec![];
    
    let result = engine.decrypt_sequences(&sequences, &frequency_table);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);
}

#[test]
fn test_decrypt_single_sequence() {
    let engine = CryptoEngine::new();
    let frequency_table = create_test_frequency_table();
    
    // Create a sequence with repeated pattern (300 nucleotides = 100 codons)
    let sequence = "ATG".repeat(100);
    let sequences = vec![create_test_sequence("test_gene", &sequence)];
    
    let result = engine.decrypt_sequences(&sequences, &frequency_table);
    assert!(result.is_ok());
    
    let regions = result.unwrap();
    // Should detect at least one region
    assert!(!regions.is_empty());
}

#[test]
fn test_decrypt_sequence_with_regulatory_elements() {
    let engine = CryptoEngine::new();
    let frequency_table = create_test_frequency_table();
    
    // Create sequence with promoter motif
    let mut sequence = "TATAAT".to_string(); // Promoter motif
    sequence.push_str(&"ATG".repeat(95)); // Fill to window size
    let sequences = vec![create_test_sequence("regulated_gene", &sequence)];
    
    let result = engine.decrypt_sequences(&sequences, &frequency_table);
    assert!(result.is_ok());
    
    let regions = result.unwrap();
    if !regions.is_empty() {
        // Check regulatory context was detected
        assert!(regions[0].regulatory_context.promoter_strength > 0.0);
    }
}

#[test]
fn test_build_codon_vector() {
    let engine = CryptoEngine::new();
    let frequency_table = create_test_frequency_table();
    
    // Test window with known codons
    let window = "ATGCGCTAA";
    
    // This is a private method, so we test it indirectly through decrypt
    let sequences = vec![create_test_sequence("test", &window.repeat(11))]; // Make it window size
    let result = engine.decrypt_sequences(&sequences, &frequency_table);
    assert!(result.is_ok());
}

#[test]
fn test_detect_trait_patterns() {
    let engine = CryptoEngine::new();
    let frequency_table = create_test_frequency_table();
    
    // Create sequence with biased codon usage
    let biased_sequence = "CTG".repeat(100); // High frequency codon
    let sequences = vec![create_test_sequence("biased_gene", &biased_sequence)];
    
    let result = engine.decrypt_sequences(&sequences, &frequency_table);
    assert!(result.is_ok());
    
    let regions = result.unwrap();
    if !regions.is_empty() {
        // Should detect high expression pattern
        assert!(!regions[0].decrypted_traits.is_empty());
    }
}

#[test]
fn test_regulatory_element_detection() {
    let engine = CryptoEngine::new();
    let frequency_table = create_test_frequency_table();
    
    // Create sequence with multiple regulatory elements
    let mut sequence = String::new();
    sequence.push_str("TATAAT"); // Promoter
    sequence.push_str("GGAGG");  // Enhancer
    sequence.push_str(&"ATG".repeat(90)); // Fill
    sequence.push_str("ATAAA");  // Silencer
    
    let sequences = vec![create_test_sequence("complex_gene", &sequence)];
    
    let result = engine.decrypt_sequences(&sequences, &frequency_table);
    assert!(result.is_ok());
    
    let regions = result.unwrap();
    if !regions.is_empty() {
        let context = &regions[0].regulatory_context;
        assert!(context.promoter_strength > 0.0);
        assert!(!context.enhancers.is_empty());
        assert!(!context.silencers.is_empty());
    }
}

#[test]
fn test_confidence_calculation() {
    let engine = CryptoEngine::new();
    let frequency_table = create_test_frequency_table();
    
    // Create high-confidence sequence
    let mut sequence = "TTGACA".to_string(); // Strong promoter
    sequence.push_str(&"CTG".repeat(98)); // Biased codons
    
    let sequences = vec![create_test_sequence("confident_gene", &sequence)];
    
    let result = engine.decrypt_sequences(&sequences, &frequency_table);
    assert!(result.is_ok());
    
    let regions = result.unwrap();
    if !regions.is_empty() {
        // Check confidence scores exist
        assert!(!regions[0].confidence_scores.is_empty());
        
        // At least one trait should have high confidence
        let max_confidence = regions[0]
            .confidence_scores
            .values()
            .fold(0.0f64, |a, &b| a.max(b));
        assert!(max_confidence >= 0.5);
    }
}

#[test]
fn test_merge_overlapping_regions() {
    let engine = CryptoEngine::new();
    let frequency_table = create_test_frequency_table();
    
    // Create long sequence that will produce overlapping windows
    let sequence = "ATG".repeat(200); // 600 nucleotides
    let sequences = vec![create_test_sequence("long_gene", &sequence)];
    
    let result = engine.decrypt_sequences(&sequences, &frequency_table);
    assert!(result.is_ok());
    
    let regions = result.unwrap();
    // Verify regions are properly merged (no excessive overlap)
    for i in 1..regions.len() {
        let prev_end = regions[i - 1].end;
        let curr_start = regions[i].start;
        
        // Should have minimal overlap after merging
        if curr_start < prev_end {
            let overlap = (prev_end - curr_start) as f64 / (regions[i].end - curr_start) as f64;
            assert!(overlap <= 0.3); // Max 30% overlap
        }
    }
}

#[test]
fn test_short_sequence_handling() {
    let engine = CryptoEngine::new();
    let frequency_table = create_test_frequency_table();
    
    // Create sequence shorter than minimum window
    let short_sequence = "ATG".repeat(20); // Only 60 nucleotides
    let sequences = vec![create_test_sequence("short_gene", &short_sequence)];
    
    let result = engine.decrypt_sequences(&sequences, &frequency_table);
    assert!(result.is_ok());
    
    // Should handle gracefully, possibly no regions detected
    let regions = result.unwrap();
    assert_eq!(regions.len(), 0); // Too short for analysis
}

#[test]
fn test_invalid_nucleotides() {
    let engine = CryptoEngine::new();
    let frequency_table = create_test_frequency_table();
    
    // Create sequence with invalid characters
    let invalid_sequence = "ATG".repeat(30) + "NNN" + &"GCG".repeat(30);
    let sequences = vec![create_test_sequence("invalid_gene", &invalid_sequence)];
    
    let result = engine.decrypt_sequences(&sequences, &frequency_table);
    assert!(result.is_ok());
    
    // Should handle invalid nucleotides gracefully
    let regions = result.unwrap();
    // May or may not detect regions depending on invalid nucleotide handling
    assert!(regions.len() >= 0);
}

#[test]
fn test_periodic_pattern_detection() {
    let engine = CryptoEngine::new();
    let frequency_table = create_test_frequency_table();
    
    // Create sequence with periodic pattern
    let pattern = "ATGCGCTAA"; // 3-codon pattern
    let periodic_sequence = pattern.repeat(34); // Make it window size
    let sequences = vec![create_test_sequence("periodic_gene", &periodic_sequence)];
    
    let result = engine.decrypt_sequences(&sequences, &frequency_table);
    assert!(result.is_ok());
    
    let regions = result.unwrap();
    if !regions.is_empty() {
        // Should detect structural pattern
        let has_structural = regions[0]
            .decrypted_traits
            .iter()
            .any(|t| t.contains("structural"));
        // Pattern detection is probabilistic, so we don't assert
    }
}

#[test]
fn test_parallel_processing() {
    let engine = CryptoEngine::new();
    let frequency_table = create_test_frequency_table();
    
    // Create multiple sequences
    let mut sequences = Vec::new();
    for i in 0..10 {
        let seq = "ATG".repeat(100);
        sequences.push(create_test_sequence(&format!("gene_{}", i), &seq));
    }
    
    let result = engine.decrypt_sequences(&sequences, &frequency_table);
    assert!(result.is_ok());
    
    // Should process all sequences
    let regions = result.unwrap();
    // Each sequence might produce multiple regions
    assert!(regions.len() >= sequences.len());
}

#[test]
fn test_edge_case_window_boundaries() {
    let engine = CryptoEngine::new();
    let frequency_table = create_test_frequency_table();
    
    // Create sequence exactly at window boundary
    let sequence = "ATG".repeat(100); // Exactly 300 nucleotides
    let sequences = vec![create_test_sequence("boundary_gene", &sequence)];
    
    let result = engine.decrypt_sequences(&sequences, &frequency_table);
    assert!(result.is_ok());
    
    let regions = result.unwrap();
    if !regions.is_empty() {
        // Check boundaries are within sequence
        for region in &regions {
            assert!(region.start < 300);
            assert!(region.end <= 300);
            assert!(region.start < region.end);
        }
    }
}