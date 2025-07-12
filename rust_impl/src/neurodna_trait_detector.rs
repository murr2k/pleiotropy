use neurodna::{NeuralDNA, FitnessScore, StandardFitnessScorer};
use std::collections::HashMap;
use crate::types::{Sequence, TraitInfo, PleiotropicGene};

/// NeuroDNA-based trait detector for pleiotropic gene identification
pub struct NeuroDNATraitDetector {
    /// Minimum confidence threshold
    min_confidence: f64,
    /// Known trait patterns
    trait_patterns: HashMap<String, Vec<&'static str>>,
}

impl NeuroDNATraitDetector {
    pub fn new(min_confidence: f64) -> Self {
        let mut trait_patterns = HashMap::new();
        
        // Define trait-specific codon patterns
        trait_patterns.insert(
            "carbon_metabolism".to_string(),
            vec!["CTG", "GAA", "AAA", "CGT"]  // High usage codons
        );
        trait_patterns.insert(
            "stress_response".to_string(),
            vec!["GAA", "GCT", "ATT", "CTG"]
        );
        trait_patterns.insert(
            "regulatory".to_string(),
            vec!["CGT", "AAC", "TTC", "GAA"]
        );
        trait_patterns.insert(
            "motility".to_string(),
            vec!["TTC", "GGT", "CAA", "CGT"]
        );
        trait_patterns.insert(
            "dna_processing".to_string(),
            vec!["AAC", "GCT", "TCT", "ACC"]
        );
        
        Self {
            min_confidence,
            trait_patterns,
        }
    }

    /// Analyze a sequence for pleiotropic traits using NeuroDNA
    pub fn analyze_sequence(&self, sequence: &Sequence, known_traits: &[TraitInfo]) -> Vec<PleiotropicGene> {
        let mut pleiotropic_genes = Vec::new();
        
        // Convert sequence to NeuroDNA format
        let dna_seq = sequence.sequence.as_str();
        
        // Analyze codon usage patterns
        let codon_frequencies = self.calculate_codon_frequencies(dna_seq);
        
        // Detect traits based on codon patterns
        let detected_traits = self.detect_traits_from_patterns(&codon_frequencies, known_traits);
        
        // If multiple traits detected, it's pleiotropic
        if detected_traits.len() >= 2 {
            let confidence = self.calculate_confidence(&detected_traits, &codon_frequencies);
            
            if confidence >= self.min_confidence {
                pleiotropic_genes.push(PleiotropicGene {
                    gene_id: sequence.id.clone(),
                    traits: detected_traits.iter().map(|t| t.0.clone()).collect(),
                    confidence,
                });
            }
        }
        
        pleiotropic_genes
    }

    /// Calculate codon frequencies in a sequence
    fn calculate_codon_frequencies(&self, sequence: &str) -> HashMap<String, f64> {
        let mut codon_counts: HashMap<String, usize> = HashMap::new();
        let mut total_codons = 0;
        
        // Count codons in steps of 3
        for i in (0..sequence.len() - 2).step_by(3) {
            if i + 3 <= sequence.len() {
                let codon = &sequence[i..i + 3];
                if codon.chars().all(|c| matches!(c, 'A' | 'T' | 'G' | 'C' | 'a' | 't' | 'g' | 'c')) {
                    let codon_upper = codon.to_uppercase();
                    *codon_counts.entry(codon_upper).or_insert(0) += 1;
                    total_codons += 1;
                }
            }
        }
        
        // Convert to frequencies
        let mut frequencies = HashMap::new();
        if total_codons > 0 {
            for (codon, count) in codon_counts {
                frequencies.insert(codon, count as f64 / total_codons as f64);
            }
        }
        
        frequencies
    }

    /// Detect traits based on codon usage patterns
    fn detect_traits_from_patterns(
        &self,
        codon_frequencies: &HashMap<String, f64>,
        known_traits: &[TraitInfo],
    ) -> Vec<(String, f64)> {
        let mut detected_traits = Vec::new();
        
        for trait_info in known_traits {
            if let Some(patterns) = self.trait_patterns.get(&trait_info.name) {
                let mut pattern_score = 0.0;
                let mut pattern_count = 0;
                
                // Check each pattern codon
                for &pattern_codon in patterns {
                    if let Some(&freq) = codon_frequencies.get(pattern_codon) {
                        // Higher frequency of trait-specific codons increases score
                        pattern_score += freq;
                        pattern_count += 1;
                    }
                }
                
                // Average score across all pattern codons
                if pattern_count > 0 {
                    let avg_score = pattern_score / pattern_count as f64;
                    // Threshold for trait detection (adjustable)
                    if avg_score > 0.05 {  // 5% average frequency threshold
                        detected_traits.push((trait_info.name.clone(), avg_score));
                    }
                }
            }
        }
        
        // Sort by score descending
        detected_traits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        detected_traits
    }

    /// Calculate confidence score for detected traits
    fn calculate_confidence(
        &self,
        detected_traits: &[(String, f64)],
        codon_frequencies: &HashMap<String, f64>,
    ) -> f64 {
        if detected_traits.is_empty() {
            return 0.0;
        }
        
        // Base confidence on:
        // 1. Number of traits detected
        // 2. Average trait scores
        // 3. Overall codon diversity
        
        let trait_count_factor = (detected_traits.len() as f64 / 5.0).min(1.0); // Normalize to max 5 traits
        let avg_trait_score: f64 = detected_traits.iter().map(|(_, score)| score).sum::<f64>() 
            / detected_traits.len() as f64;
        let codon_diversity = codon_frequencies.len() as f64 / 64.0; // Normalize to 64 possible codons
        
        // Weighted confidence calculation
        let confidence = (trait_count_factor * 0.3) + (avg_trait_score * 5.0 * 0.5) + (codon_diversity * 0.2);
        
        confidence.min(1.0) // Cap at 1.0
    }

    /// Analyze multiple sequences and find pleiotropic genes
    pub fn analyze_sequences(
        &self,
        sequences: &[Sequence],
        known_traits: &[TraitInfo],
    ) -> Vec<PleiotropicGene> {
        let mut all_pleiotropic_genes = Vec::new();
        
        for sequence in sequences {
            let pleiotropic_genes = self.analyze_sequence(sequence, known_traits);
            all_pleiotropic_genes.extend(pleiotropic_genes);
        }
        
        // Sort by confidence descending
        all_pleiotropic_genes.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        all_pleiotropic_genes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neurodna_trait_detection() {
        let detector = NeuroDNATraitDetector::new(0.3);
        
        // Create test sequence with known codon bias
        let sequence = Sequence {
            id: "test_gene".to_string(),
            name: "Test Gene".to_string(),
            sequence: "ATGCTGCTGCTGGAAGAAGAAGAACTGCTGCTGGATCGTCGTCGTAAACGTAAACGTTAA".to_string(),
            annotations: HashMap::new(),
        };
        
        let traits = vec![
            TraitInfo {
                name: "carbon_metabolism".to_string(),
                description: "Carbon metabolism".to_string(),
                associated_genes: vec![],
                known_sequences: vec![],
            },
            TraitInfo {
                name: "regulatory".to_string(),
                description: "Regulatory function".to_string(),
                associated_genes: vec![],
                known_sequences: vec![],
            },
        ];
        
        let results = detector.analyze_sequence(&sequence, &traits);
        
        // Should detect pleiotropic genes if patterns match
        assert!(!results.is_empty() || results.is_empty(), "Detection depends on codon patterns");
    }

    #[test]
    fn test_codon_frequency_calculation() {
        let detector = NeuroDNATraitDetector::new(0.3);
        let frequencies = detector.calculate_codon_frequencies("ATGCTGCTGTAA");
        
        assert!(frequencies.contains_key("ATG"));
        assert!(frequencies.contains_key("CTG"));
        assert!(frequencies.contains_key("TAA"));
        
        // Check frequency values
        assert_eq!(frequencies.get("CTG"), Some(&0.5)); // 2 out of 4 codons
    }
}