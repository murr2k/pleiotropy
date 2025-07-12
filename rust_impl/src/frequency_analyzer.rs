use crate::types::{CodonFrequency, FrequencyTable, Sequence, genetic_code};
use anyhow::Result;
use rayon::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::collections::HashMap;

pub struct FrequencyAnalyzer {
    codon_size: usize,
    min_occurrences: usize,
}

impl FrequencyAnalyzer {
    pub fn new() -> Self {
        Self {
            codon_size: 3,
            min_occurrences: 5,
        }
    }

    pub fn build_frequency_table(&self, sequences: &[Sequence]) -> Result<FrequencyTable> {
        // Count codons in parallel
        let codon_counts: HashMap<String, usize> = sequences
            .par_iter()
            .map(|seq| self.count_codons(&seq.sequence))
            .reduce(HashMap::new, |mut acc, counts| {
                for (codon, count) in counts {
                    *acc.entry(codon).or_insert(0) += count;
                }
                acc
            });

        let total_codons: usize = codon_counts.values().sum();
        let genetic_code_map = genetic_code();

        // Calculate frequencies
        let mut codon_frequencies = Vec::new();
        for (codon, count) in &codon_counts {
            if *count < self.min_occurrences {
                continue;
            }

            let rna_codon = codon.replace('T', "U");
            let amino_acid = genetic_code_map
                .get(&rna_codon)
                .copied()
                .unwrap_or('X');

            codon_frequencies.push(CodonFrequency {
                codon: codon.clone(),
                amino_acid,
                global_frequency: *count as f64 / total_codons as f64,
                trait_specific_frequency: HashMap::new(),
            });
        }

        Ok(FrequencyTable {
            codon_frequencies,
            total_codons,
            trait_codon_counts: HashMap::new(),
        })
    }

    fn count_codons(&self, sequence: &str) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        let seq = sequence.to_uppercase();
        
        for i in (0..seq.len()).step_by(self.codon_size) {
            if i + self.codon_size <= seq.len() {
                let codon = &seq[i..i + self.codon_size];
                if codon.chars().all(|c| "ATCG".contains(c)) {
                    *counts.entry(codon.to_string()).or_insert(0) += 1;
                }
            }
        }
        
        counts
    }

    /// Calculate codon usage bias for specific traits
    pub fn calculate_trait_bias(
        &self,
        sequences: &[Sequence],
        trait_sequences: &[String],
        frequency_table: &mut FrequencyTable,
    ) -> Result<()> {
        // Count codons in trait-specific sequences
        let trait_counts = trait_sequences
            .par_iter()
            .map(|seq| self.count_codons(seq))
            .reduce(HashMap::new, |mut acc, counts| {
                for (codon, count) in counts {
                    *acc.entry(codon).or_insert(0) += count;
                }
                acc
            });

        let trait_total: usize = trait_counts.values().sum();

        // Update frequency table with trait-specific frequencies
        for codon_freq in &mut frequency_table.codon_frequencies {
            let trait_count = trait_counts.get(&codon_freq.codon).copied().unwrap_or(0);
            let trait_frequency = trait_count as f64 / trait_total as f64;
            
            codon_freq.trait_specific_frequency.insert(
                "current_trait".to_string(),
                trait_frequency,
            );
        }

        Ok(())
    }

    /// Perform chi-squared test for codon bias significance
    pub fn test_codon_bias_significance(
        &self,
        observed_freq: f64,
        expected_freq: f64,
        total_observations: usize,
    ) -> f64 {
        let observed = (observed_freq * total_observations as f64) as f64;
        let expected = (expected_freq * total_observations as f64) as f64;
        
        if expected == 0.0 {
            return 0.0;
        }

        let chi_squared = ((observed - expected).powi(2)) / expected;
        let df = 1.0; // degrees of freedom
        let dist = ChiSquared::new(df).unwrap();
        
        1.0 - dist.cdf(chi_squared)
    }

    /// Calculate mutual information between codons and traits
    pub fn mutual_information(
        &self,
        codon_frequencies: &[CodonFrequency],
        trait_name: &str,
    ) -> f64 {
        let mut mi = 0.0;
        
        for codon_freq in codon_frequencies {
            let p_codon = codon_freq.global_frequency;
            if let Some(p_codon_trait) = codon_freq.trait_specific_frequency.get(trait_name) {
                if p_codon > 0.0 && *p_codon_trait > 0.0 {
                    mi += p_codon_trait * (p_codon_trait / p_codon).ln();
                }
            }
        }
        
        mi
    }

    /// Identify synonymous codon usage patterns
    pub fn analyze_synonymous_usage(&self, frequency_table: &FrequencyTable) -> HashMap<char, Vec<(String, f64)>> {
        let mut synonymous_groups: HashMap<char, Vec<(String, f64)>> = HashMap::new();
        
        for codon_freq in &frequency_table.codon_frequencies {
            synonymous_groups
                .entry(codon_freq.amino_acid)
                .or_insert_with(Vec::new)
                .push((codon_freq.codon.clone(), codon_freq.global_frequency));
        }
        
        // Sort by frequency within each group
        for group in synonymous_groups.values_mut() {
            group.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        }
        
        synonymous_groups
    }
}