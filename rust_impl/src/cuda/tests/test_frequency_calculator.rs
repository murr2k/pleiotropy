use crate::cuda::{CudaDevice, kernels::FrequencyCalculator};
use crate::types::{CodonCounts, TraitInfo, CudaFrequencyTable};
use std::collections::HashMap;

fn create_test_codon_counts() -> Vec<CodonCounts> {
    let mut counts1 = CodonCounts::new();
    counts1.insert("ATG".to_string(), 10);
    counts1.insert("GCG".to_string(), 5);
    counts1.insert("AAA".to_string(), 15);
    
    let mut counts2 = CodonCounts::new();
    counts2.insert("ATG".to_string(), 20);
    counts2.insert("GCG".to_string(), 10);
    counts2.insert("TTT".to_string(), 5);
    
    vec![counts1, counts2]
}

fn create_test_traits() -> Vec<TraitInfo> {
    vec![
        TraitInfo {
            name: "growth".to_string(),
            description: "Growth trait".to_string(),
            associated_genes: vec!["gene1".to_string()],
            known_sequences: vec![],
        },
        TraitInfo {
            name: "resistance".to_string(),
            description: "Resistance trait".to_string(),
            associated_genes: vec!["gene2".to_string()],
            known_sequences: vec![],
        },
    ]
}

#[test]
fn test_frequency_calculator_initialization() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let calculator = FrequencyCalculator::new(&device);
    assert!(calculator.is_ok(), "Failed to initialize frequency calculator");
}

#[test]
fn test_basic_frequency_calculation() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let calculator = FrequencyCalculator::new(&device).expect("Failed to create calculator");
    
    let codon_counts = create_test_codon_counts();
    let traits = create_test_traits();
    
    let result = calculator.calculate(&codon_counts, &traits)
        .expect("Failed to calculate frequencies");
    
    // Check global frequencies exist
    assert!(!result.global_frequencies.is_empty());
    
    // Check ATG frequency (should be average of 10 and 20)
    let atg_freq = result.global_frequencies.get("ATG");
    assert!(atg_freq.is_some());
}

#[test]
fn test_batch_frequency_calculation() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let calculator = FrequencyCalculator::new(&device).expect("Failed to create calculator");
    
    let codon_counts = create_test_codon_counts();
    // Trait assignments: sequence 0 has trait 0, sequence 1 has trait 1
    let trait_assignments = vec![0b01, 0b10];
    
    let result = calculator.calculate_batch(&codon_counts, &trait_assignments, 2)
        .expect("Failed to calculate batch frequencies");
    
    // Check that we have trait-specific frequencies
    assert_eq!(result.trait_frequencies.len(), 2);
    assert!(result.trait_frequencies.contains_key("trait_0"));
    assert!(result.trait_frequencies.contains_key("trait_1"));
}

#[test]
fn test_frequency_normalization() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let calculator = FrequencyCalculator::new(&device).expect("Failed to create calculator");
    
    let mut counts = CodonCounts::new();
    // Total of 100 codons
    counts.insert("ATG".to_string(), 25);
    counts.insert("GCG".to_string(), 25);
    counts.insert("AAA".to_string(), 25);
    counts.insert("TTT".to_string(), 25);
    
    let codon_counts = vec![counts];
    let traits = create_test_traits();
    
    let result = calculator.calculate(&codon_counts, &traits)
        .expect("Failed to calculate frequencies");
    
    // Check that frequencies sum to approximately 1.0
    let sum: f64 = result.global_frequencies.values()
        .filter(|&&v| v > 0.0)
        .sum();
    
    // Allow some floating point error
    assert!((sum - 1.0).abs() < 0.01, "Frequencies should sum to ~1.0, got {}", sum);
}

#[test]
fn test_empty_codon_counts() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let calculator = FrequencyCalculator::new(&device).expect("Failed to create calculator");
    
    let codon_counts = vec![];
    let traits = create_test_traits();
    
    let result = calculator.calculate(&codon_counts, &traits);
    assert!(result.is_ok(), "Should handle empty input gracefully");
}

#[test]
fn test_performance_large_dataset() {
    use std::time::Instant;
    
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let calculator = FrequencyCalculator::new(&device).expect("Failed to create calculator");
    
    // Create 1000 sequences with random codon counts
    let mut codon_counts = Vec::new();
    for _ in 0..1000 {
        let mut counts = CodonCounts::new();
        for codon in ["ATG", "GCG", "AAA", "TTT", "CCC", "GGG"].iter() {
            counts.insert(codon.to_string(), (rand::random::<u8>() as usize) + 1);
        }
        codon_counts.push(counts);
    }
    
    let traits = create_test_traits();
    
    let start = Instant::now();
    let result = calculator.calculate(&codon_counts, &traits)
        .expect("Failed to calculate frequencies");
    let duration = start.elapsed();
    
    println!("CUDA frequency calculation for 1000 sequences: {:?}", duration);
    assert!(!result.global_frequencies.is_empty());
    assert!(duration.as_millis() < 100, "Performance requirement not met");
}

#[test]
fn test_trait_specific_bias() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let calculator = FrequencyCalculator::new(&device).expect("Failed to create calculator");
    
    // Create sequences with different codon preferences
    let mut growth_counts = CodonCounts::new();
    growth_counts.insert("ATG".to_string(), 100); // High ATG for growth
    growth_counts.insert("GCG".to_string(), 10);
    
    let mut resistance_counts = CodonCounts::new();
    resistance_counts.insert("ATG".to_string(), 10);
    resistance_counts.insert("GCG".to_string(), 100); // High GCG for resistance
    
    let codon_counts = vec![growth_counts, resistance_counts];
    let trait_assignments = vec![0b01, 0b10]; // First has growth, second has resistance
    
    let result = calculator.calculate_batch(&codon_counts, &trait_assignments, 2)
        .expect("Failed to calculate batch frequencies");
    
    // Check that trait 0 (growth) prefers ATG
    let trait0_atg = result.trait_frequencies["trait_0"]["ATG"];
    let trait0_gcg = result.trait_frequencies["trait_0"]["GCG"];
    assert!(trait0_atg > trait0_gcg, "Growth trait should prefer ATG");
    
    // Check that trait 1 (resistance) prefers GCG
    let trait1_atg = result.trait_frequencies["trait_1"]["ATG"];
    let trait1_gcg = result.trait_frequencies["trait_1"]["GCG"];
    assert!(trait1_gcg > trait1_atg, "Resistance trait should prefer GCG");
}

// Add rand dependency for testing
use rand;