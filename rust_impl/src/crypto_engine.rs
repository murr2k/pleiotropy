use crate::types::{DecryptedRegion, FrequencyTable, RegulatoryContext, Sequence};
use anyhow::Result;
use nalgebra::DVector;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

pub struct CryptoEngine {
    window_size: usize,
    overlap_threshold: f64,
    min_confidence: f64,
}

impl CryptoEngine {
    pub fn new() -> Self {
        Self {
            window_size: 300, // 100 codons
            overlap_threshold: 0.3,
            min_confidence: 0.7,
        }
    }

    /// Main decryption function using cryptanalysis techniques
    pub fn decrypt_sequences(
        &self,
        sequences: &[Sequence],
        frequency_table: &FrequencyTable,
    ) -> Result<Vec<DecryptedRegion>> {
        let results: Vec<Vec<DecryptedRegion>> = sequences
            .par_iter()
            .map(|seq| self.decrypt_single_sequence(seq, frequency_table))
            .collect();
        
        Ok(results.into_iter().flatten().collect())
    }

    fn decrypt_single_sequence(
        &self,
        sequence: &Sequence,
        frequency_table: &FrequencyTable,
    ) -> Vec<DecryptedRegion> {
        let mut regions = Vec::new();
        let seq_len = sequence.sequence.len();
        
        // Slide window across sequence
        for start in (0..seq_len).step_by(self.window_size / 2) {
            let end = (start + self.window_size).min(seq_len);
            if end - start < 90 { // Minimum 30 codons
                continue;
            }
            
            let window = &sequence.sequence[start..end];
            
            // Apply cryptanalysis to window
            if let Some(decrypted) = self.analyze_window(window, start, end, &sequence.id, frequency_table) {
                regions.push(decrypted);
            }
        }
        
        // Merge overlapping regions
        self.merge_overlapping_regions(regions)
    }

    fn analyze_window(
        &self,
        window: &str,
        start: usize,
        end: usize,
        gene_id: &str,
        frequency_table: &FrequencyTable,
    ) -> Option<DecryptedRegion> {
        // 1. Frequency analysis
        let codon_vector = self.build_codon_vector(window, frequency_table);
        
        // 2. Pattern detection using eigenanalysis
        let trait_patterns = self.detect_trait_patterns(&codon_vector);
        
        // 3. Regulatory context detection
        let regulatory_context = self.detect_regulatory_elements(window);
        
        // 4. Calculate confidence scores
        let confidence_scores = self.calculate_confidence(&trait_patterns, &regulatory_context);
        
        // Filter by minimum confidence
        let high_confidence_traits: Vec<String> = confidence_scores
            .iter()
            .filter(|(_, score)| **score >= self.min_confidence)
            .map(|(trait_name, _)| trait_name.clone())
            .collect();
        
        if high_confidence_traits.is_empty() {
            return None;
        }
        
        Some(DecryptedRegion {
            start,
            end,
            gene_id: gene_id.to_string(),
            decrypted_traits: high_confidence_traits,
            confidence_scores,
            regulatory_context,
        })
    }

    fn build_codon_vector(&self, window: &str, frequency_table: &FrequencyTable) -> DVector<f64> {
        let mut codon_counts: HashMap<String, usize> = HashMap::new();
        
        // Count codons in window
        for i in (0..window.len()).step_by(3) {
            if i + 3 <= window.len() {
                let codon = &window[i..i + 3];
                if codon.chars().all(|c| "ATCG".contains(c)) {
                    *codon_counts.entry(codon.to_string()).or_insert(0) += 1;
                }
            }
        }
        
        // Build normalized vector
        let total_codons: usize = codon_counts.values().sum();
        let mut vector = Vec::new();
        
        for codon_freq in &frequency_table.codon_frequencies {
            let count = codon_counts.get(&codon_freq.codon).copied().unwrap_or(0);
            let observed_freq = count as f64 / total_codons.max(1) as f64;
            let expected_freq = codon_freq.global_frequency;
            
            // Use log-odds ratio as feature
            let log_odds = if observed_freq > 0.0 && expected_freq > 0.0 {
                (observed_freq / expected_freq).ln()
            } else {
                0.0
            };
            
            vector.push(log_odds);
        }
        
        DVector::from_vec(vector)
    }

    fn detect_trait_patterns(&self, codon_vector: &DVector<f64>) -> Vec<String> {
        // Simplified pattern detection - in reality would use trained models
        let mut patterns = Vec::new();
        
        // Calculate vector magnitude as complexity measure
        let magnitude = codon_vector.norm();
        
        // High complexity suggests regulatory function
        if magnitude > 2.0 {
            patterns.push("regulatory".to_string());
        }
        
        // Analyze specific codon biases
        let max_bias = codon_vector.max();
        let min_bias = codon_vector.min();
        
        if max_bias > 1.5 {
            patterns.push("high_expression".to_string());
        }
        if min_bias < -1.5 {
            patterns.push("low_expression".to_string());
        }
        
        // Check for periodic patterns (simplified)
        if self.has_periodic_pattern(codon_vector) {
            patterns.push("structural".to_string());
        }
        
        patterns
    }

    fn has_periodic_pattern(&self, vector: &DVector<f64>) -> bool {
        // Simple autocorrelation check
        if vector.len() < 6 {
            return false;
        }
        
        let mut correlation = 0.0;
        let period = 3; // Check for 3-codon periodicity
        
        for i in 0..vector.len() - period {
            correlation += vector[i] * vector[i + period];
        }
        
        correlation /= (vector.len() - period) as f64;
        correlation.abs() > 0.5
    }

    fn detect_regulatory_elements(&self, window: &str) -> RegulatoryContext {
        let mut context = RegulatoryContext {
            promoter_strength: 0.0,
            enhancers: Vec::new(),
            silencers: Vec::new(),
            expression_conditions: Vec::new(),
        };
        
        // Simple motif detection
        let promoter_motifs = ["TATAAT", "TTGACA", "CAAT", "GCGC"];
        let enhancer_motifs = ["GGAGG", "CCACC"];
        let silencer_motifs = ["ATAAA", "TTTTT"];
        
        // Check for promoter elements
        for motif in &promoter_motifs {
            if window.contains(motif) {
                context.promoter_strength += 0.25;
            }
        }
        
        // Find enhancers
        for motif in &enhancer_motifs {
            for (idx, _) in window.match_indices(motif) {
                context.enhancers.push((idx, idx + motif.len()));
            }
        }
        
        // Find silencers
        for motif in &silencer_motifs {
            for (idx, _) in window.match_indices(motif) {
                context.silencers.push((idx, idx + motif.len()));
            }
        }
        
        // Infer expression conditions
        if context.promoter_strength > 0.5 {
            context.expression_conditions.push("constitutive".to_string());
        }
        if !context.enhancers.is_empty() {
            context.expression_conditions.push("inducible".to_string());
        }
        if !context.silencers.is_empty() {
            context.expression_conditions.push("repressible".to_string());
        }
        
        context
    }

    fn calculate_confidence(
        &self,
        trait_patterns: &[String],
        regulatory_context: &RegulatoryContext,
    ) -> HashMap<String, f64> {
        let mut scores = HashMap::new();
        
        for pattern in trait_patterns {
            let mut score = 0.5; // Base score
            
            // Adjust based on regulatory context
            match pattern.as_str() {
                "regulatory" => {
                    score += regulatory_context.promoter_strength * 0.3;
                    score += (regulatory_context.enhancers.len() as f64) * 0.1;
                }
                "high_expression" => {
                    score += regulatory_context.promoter_strength * 0.4;
                    score -= (regulatory_context.silencers.len() as f64) * 0.1;
                }
                "structural" => {
                    score += 0.2; // Structural genes often have distinct patterns
                }
                _ => {}
            }
            
            scores.insert(pattern.clone(), score.min(1.0));
        }
        
        scores
    }

    fn merge_overlapping_regions(&self, mut regions: Vec<DecryptedRegion>) -> Vec<DecryptedRegion> {
        if regions.is_empty() {
            return regions;
        }
        
        // Sort by start position
        regions.sort_by_key(|r| r.start);
        
        let mut merged = Vec::new();
        let mut current = regions.remove(0);
        
        for region in regions {
            let overlap = (current.end.min(region.end) as f64 - region.start.max(current.start) as f64)
                / (region.end - region.start) as f64;
            
            if overlap > self.overlap_threshold {
                // Merge regions
                current.end = current.end.max(region.end);
                
                // Combine traits
                let mut all_traits: HashSet<String> = current.decrypted_traits.iter().cloned().collect();
                all_traits.extend(region.decrypted_traits);
                current.decrypted_traits = all_traits.into_iter().collect();
                
                // Merge confidence scores
                for (trait_name, score) in region.confidence_scores {
                    current.confidence_scores
                        .entry(trait_name)
                        .and_modify(|s| *s = s.max(score))
                        .or_insert(score);
                }
            } else {
                merged.push(current);
                current = region;
            }
        }
        
        merged.push(current);
        merged
    }
}