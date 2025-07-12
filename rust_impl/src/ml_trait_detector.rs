use std::collections::HashMap;
use ndarray::{Array1, Array2, ArrayView1};
use bio::seq_analysis::orf::{Orf, Finder};
use bio::alignment::pairwise::*;
use bio::pattern_matching::kmp::KMP;
use statrs::statistics::{Statistics, OrderStatistics};
use crate::types::{Sequence, Trait, TraitPattern};

/// Machine Learning-based trait detector using pattern recognition
pub struct MLTraitDetector {
    /// Codon usage patterns for each trait
    trait_patterns: HashMap<String, TraitPattern>,
    /// K-mer size for pattern matching
    kmer_size: usize,
    /// Minimum pattern score threshold
    min_score: f64,
}

impl MLTraitDetector {
    pub fn new(kmer_size: usize, min_score: f64) -> Self {
        Self {
            trait_patterns: Self::initialize_trait_patterns(),
            kmer_size,
            min_score,
        }
    }

    /// Initialize known trait patterns based on codon usage biases
    fn initialize_trait_patterns() -> HashMap<String, TraitPattern> {
        let mut patterns = HashMap::new();
        
        // Carbon metabolism: High CTG (Leu), Low CTA
        patterns.insert("carbon_metabolism".to_string(), TraitPattern {
            preferred_codons: vec!["CTG", "GAA", "AAA", "CGT"].iter().map(|s| s.to_string()).collect(),
            avoided_codons: vec!["CTA", "GAG", "AAG"].iter().map(|s| s.to_string()).collect(),
            motifs: vec!["TATAAT", "TTGACA", "CAAATC"].iter().map(|s| s.to_string()).collect(), // Promoter elements
            weight: 1.0,
        });
        
        // Stress response: High GAA (Glu), Low GAG
        patterns.insert("stress_response".to_string(), TraitPattern {
            preferred_codons: vec!["GAA", "GCT", "ATT", "CTG"].iter().map(|s| s.to_string()).collect(),
            avoided_codons: vec!["GAG", "GCC", "ATC"].iter().map(|s| s.to_string()).collect(),
            motifs: vec!["CTAG", "GAACG", "TTGAC"].iter().map(|s| s.to_string()).collect(), // RpoS binding
            weight: 1.0,
        });
        
        // Regulatory: High CGT (Arg), Low CGC
        patterns.insert("regulatory".to_string(), TraitPattern {
            preferred_codons: vec!["CGT", "AAC", "TTC", "GAA"].iter().map(|s| s.to_string()).collect(),
            avoided_codons: vec!["CGC", "AAT", "TTT"].iter().map(|s| s.to_string()).collect(),
            motifs: vec!["TGTGA", "TCACA", "AACGTT"].iter().map(|s| s.to_string()).collect(), // CRP sites
            weight: 1.0,
        });
        
        // Motility: High TTC (Phe), Low TTT
        patterns.insert("motility".to_string(), TraitPattern {
            preferred_codons: vec!["TTC", "GGT", "CAA", "CGT"].iter().map(|s| s.to_string()).collect(),
            avoided_codons: vec!["TTT", "GGC", "CAG"].iter().map(|s| s.to_string()).collect(),
            motifs: vec!["GCAAT", "ATTGC", "CTAA"].iter().map(|s| s.to_string()).collect(), // FlhDC binding
            weight: 1.0,
        });
        
        // DNA processing: High AAC (Asn), Low AAT
        patterns.insert("dna_processing".to_string(), TraitPattern {
            preferred_codons: vec!["AAC", "GCT", "TCT", "ACC"].iter().map(|s| s.to_string()).collect(),
            avoided_codons: vec!["AAT", "GCC", "TCC"].iter().map(|s| s.to_string()).collect(),
            motifs: vec!["GGATCC", "GAATTC", "CTGCAG"].iter().map(|s| s.to_string()).collect(), // Restriction sites
            weight: 1.0,
        });
        
        patterns
    }

    /// Detect traits in a sequence using ML-based pattern recognition
    pub fn detect_traits(&self, sequence: &Sequence) -> Vec<(String, f64)> {
        let mut trait_scores: HashMap<String, f64> = HashMap::new();
        
        // Extract features from sequence
        let features = self.extract_features(sequence);
        
        // Score each trait based on pattern matching
        for (trait_name, pattern) in &self.trait_patterns {
            let score = self.score_trait(&features, pattern);
            if score > self.min_score {
                trait_scores.insert(trait_name.clone(), score);
            }
        }
        
        // Convert to sorted vector
        let mut results: Vec<(String, f64)> = trait_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        results
    }

    /// Extract features from a sequence
    fn extract_features(&self, sequence: &Sequence) -> SequenceFeatures {
        let seq_str = &sequence.sequence;
        
        // Calculate codon frequencies
        let codon_freqs = self.calculate_codon_frequencies(seq_str);
        
        // Find k-mers
        let kmers = self.extract_kmers(seq_str, self.kmer_size);
        
        // Find ORFs using rust-bio
        let finder = Finder::new(18, 200, bio::alphabets::dna::revcomp);
        let orfs: Vec<Orf> = finder.find_all(seq_str.as_bytes()).collect();
        
        // Calculate GC content
        let gc_content = self.calculate_gc_content(seq_str);
        
        SequenceFeatures {
            codon_frequencies: codon_freqs,
            kmers,
            orf_count: orfs.len(),
            gc_content,
            length: seq_str.len(),
        }
    }

    /// Calculate codon frequencies in a sequence
    fn calculate_codon_frequencies(&self, sequence: &str) -> HashMap<String, f64> {
        let mut codon_counts: HashMap<String, usize> = HashMap::new();
        let mut total_codons = 0;
        
        // Count codons
        for i in (0..sequence.len() - 2).step_by(3) {
            if i + 3 <= sequence.len() {
                let codon = &sequence[i..i + 3];
                *codon_counts.entry(codon.to_string()).or_insert(0) += 1;
                total_codons += 1;
            }
        }
        
        // Convert to frequencies
        let mut frequencies = HashMap::new();
        for (codon, count) in codon_counts {
            frequencies.insert(codon, count as f64 / total_codons as f64);
        }
        
        frequencies
    }

    /// Extract k-mers from sequence
    fn extract_kmers(&self, sequence: &str, k: usize) -> HashMap<String, usize> {
        let mut kmers = HashMap::new();
        
        if sequence.len() >= k {
            for i in 0..=sequence.len() - k {
                let kmer = &sequence[i..i + k];
                *kmers.entry(kmer.to_string()).or_insert(0) += 1;
            }
        }
        
        kmers
    }

    /// Calculate GC content
    fn calculate_gc_content(&self, sequence: &str) -> f64 {
        let gc_count = sequence.chars()
            .filter(|&c| c == 'G' || c == 'C' || c == 'g' || c == 'c')
            .count();
        
        gc_count as f64 / sequence.len() as f64
    }

    /// Score a trait based on sequence features
    fn score_trait(&self, features: &SequenceFeatures, pattern: &TraitPattern) -> f64 {
        let mut score = 0.0;
        
        // Score based on preferred codon usage
        for codon in &pattern.preferred_codons {
            if let Some(&freq) = features.codon_frequencies.get(codon) {
                score += freq * 2.0; // Higher weight for preferred codons
            }
        }
        
        // Penalize avoided codons
        for codon in &pattern.avoided_codons {
            if let Some(&freq) = features.codon_frequencies.get(codon) {
                score -= freq * 1.5; // Penalty for avoided codons
            }
        }
        
        // Check for regulatory motifs
        for motif in &pattern.motifs {
            // Use rust-bio's KMP pattern matching
            let kmp = KMP::new(motif.as_bytes());
            let matches: Vec<usize> = kmp.find_all(&features.kmers.keys().collect::<Vec<_>>().join("").as_bytes()).collect();
            score += matches.len() as f64 * 0.5;
        }
        
        // Normalize by sequence length
        score *= 1000.0 / features.length as f64;
        
        // Apply pattern weight
        score *= pattern.weight;
        
        score
    }

    /// Train the detector on known examples (placeholder for future ML integration)
    pub fn train(&mut self, training_data: Vec<(Sequence, Vec<String>)>) {
        // This is where we would implement actual ML training
        // For now, we can refine patterns based on training data
        
        for (sequence, known_traits) in training_data {
            let features = self.extract_features(&sequence);
            
            // Update pattern weights based on success
            for trait_name in known_traits {
                if let Some(pattern) = self.trait_patterns.get_mut(&trait_name) {
                    // Simple weight adjustment based on detection success
                    let detected_traits = self.detect_traits(&sequence);
                    let was_detected = detected_traits.iter().any(|(t, _)| t == &trait_name);
                    
                    if was_detected {
                        pattern.weight *= 1.1; // Increase weight for successful detection
                    } else {
                        pattern.weight *= 0.9; // Decrease weight for missed detection
                    }
                }
            }
        }
    }
}

/// Features extracted from a sequence
struct SequenceFeatures {
    codon_frequencies: HashMap<String, f64>,
    kmers: HashMap<String, usize>,
    orf_count: usize,
    gc_content: f64,
    length: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_trait_detection() {
        let detector = MLTraitDetector::new(6, 0.5);
        
        // Create a synthetic sequence with carbon metabolism bias
        let sequence = Sequence {
            id: "test_gene".to_string(),
            sequence: "ATGCTGCTGCTGCTGGAAGAAGAAGAACTGCTGCTGGATCTGCTGCTGGTGCGTCGTCGTAAACGTAAACGTAAATAA".to_string(),
            description: Some("Test sequence with CTG bias".to_string()),
        };
        
        let traits = detector.detect_traits(&sequence);
        
        // Should detect some traits based on codon usage
        assert!(!traits.is_empty(), "Should detect at least one trait");
        
        // The first trait should have a positive score
        if let Some((_, score)) = traits.first() {
            assert!(*score > 0.0, "First trait should have positive score");
        }
    }

    #[test]
    fn test_codon_frequency_calculation() {
        let detector = MLTraitDetector::new(6, 0.5);
        let freqs = detector.calculate_codon_frequencies("ATGCTGCTGTAA");
        
        assert_eq!(freqs.get("ATG"), Some(&0.25));
        assert_eq!(freqs.get("CTG"), Some(&0.5));
        assert_eq!(freqs.get("TAA"), Some(&0.25));
    }
}