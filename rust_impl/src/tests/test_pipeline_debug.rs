// Test cases for debugging the trait detection pipeline
use crate::*;
use std::collections::HashMap;

#[test]
fn test_trait_pattern_detection() {
    let engine = CryptoEngine::new();
    
    // Create a synthetic sequence with known patterns
    let test_sequence = Sequence {
        id: "test_gene".to_string(),
        name: "Test Gene".to_string(),
        sequence: "ATGAAAGAAGAAGAACTGCTGCTGCTGGATGATGATGATCGCCGCCGCAAAAAAGGG".to_string(),
        annotations: HashMap::new(),
    };
    
    // Build frequency table
    let analyzer = FrequencyAnalyzer::new();
    let sequences = vec![test_sequence.clone()];
    let freq_table = analyzer.build_frequency_table(&sequences).unwrap();
    
    // Test decryption
    let decrypted = engine.decrypt_sequences(&sequences, &freq_table).unwrap();
    
    println!("Decrypted regions: {:?}", decrypted);
    assert!(!decrypted.is_empty(), "Should detect at least one region");
    
    // Check that traits were detected
    let has_traits = decrypted.iter().any(|r| !r.decrypted_traits.is_empty());
    assert!(has_traits, "Should detect at least one trait");
}

#[test]
fn test_pleiotropic_gene_detection() {
    let mut analyzer = GenomicCryptanalysis::new();
    
    // Create test sequences with known pleiotropic patterns
    let sequences = vec![
        Sequence {
            id: "crp_gene".to_string(),
            name: "CRP regulatory protein".to_string(),
            sequence: "ATGGTGCTTGGCAAACCGCAAACAGACCCGACTCTCGAACTGCACGCTGAAAAAGGG".to_string(),
            annotations: HashMap::new(),
        },
        Sequence {
            id: "fis_gene".to_string(),
            name: "FIS nucleoid protein".to_string(),
            sequence: "ATGAAAGAAGAAGAACTGAAAAAAGCGCGCGATGATGATCGCCGCCGCGGGTTTTAA".to_string(),
            annotations: HashMap::new(),
        },
    ];
    
    // Define traits matching the default E. coli traits
    let traits = vec![
        TraitInfo {
            name: "carbon_metabolism".to_string(),
            description: "Carbon source utilization".to_string(),
            associated_genes: vec!["crp".to_string()],
            known_sequences: vec![],
        },
        TraitInfo {
            name: "regulatory".to_string(),
            description: "Gene expression regulation".to_string(),
            associated_genes: vec!["crp".to_string(), "fis".to_string()],
            known_sequences: vec![],
        },
    ];
    
    // Create a test file
    let test_file = "/tmp/test_sequences.fasta";
    let mut fasta_content = String::new();
    for seq in &sequences {
        fasta_content.push_str(&format!(">{} {}\n{}\n", seq.id, seq.name, seq.sequence));
    }
    std::fs::write(test_file, fasta_content).unwrap();
    
    // Analyze
    let analysis = analyzer.analyze_genome(test_file, traits).unwrap();
    
    println!("Analysis results:");
    println!("  Sequences: {}", analysis.sequences);
    println!("  Identified traits: {}", analysis.identified_traits.len());
    for trait in &analysis.identified_traits {
        println!("    Gene: {}, Traits: {:?}, Confidence: {:.3}", 
                 trait.gene_id, trait.trait_names, trait.confidence_score);
    }
    
    // Find pleiotropic genes
    let pleiotropic = analyzer.find_pleiotropic_genes(&analysis, 2);
    
    println!("\nPleiotropic genes:");
    for gene in &pleiotropic {
        println!("  Gene: {}, Traits: {:?}, Confidence: {:.3}",
                 gene.gene_id, gene.traits, gene.confidence);
    }
    
    // Clean up
    std::fs::remove_file(test_file).ok();
    
    assert!(!analysis.identified_traits.is_empty(), "Should identify some traits");
    assert!(!pleiotropic.is_empty(), "Should find at least one pleiotropic gene");
}

#[test]
fn test_confidence_threshold_adjustment() {
    let engine = CryptoEngine::new();
    
    // Verify the confidence threshold was reduced
    assert_eq!(engine.min_confidence, 0.4, "Confidence threshold should be 0.4");
}

#[test]
fn test_trait_specific_frequencies() {
    let analyzer = FrequencyAnalyzer::new();
    
    // Create test sequences
    let sequences = vec![
        Sequence {
            id: "high_expr_gene".to_string(),
            name: "Highly expressed gene".to_string(),
            sequence: "ATGAAAAAAGAAGAACTGCTGCTGCTGGATGATGATGATCGCCGCCGCAAAAAAGGG".to_string(),
            annotations: HashMap::new(),
        },
    ];
    
    let freq_table = analyzer.build_frequency_table(&sequences).unwrap();
    
    // Check that synthetic trait patterns were added
    let has_trait_specific = freq_table.codon_frequencies.iter()
        .any(|cf| !cf.trait_specific_frequency.is_empty());
    
    assert!(has_trait_specific, "Should have trait-specific frequencies");
    
    // Check for specific traits
    let has_high_expr = freq_table.codon_frequencies.iter()
        .any(|cf| cf.trait_specific_frequency.contains_key("high_expression"));
    
    assert!(has_high_expr, "Should have high_expression trait patterns");
}