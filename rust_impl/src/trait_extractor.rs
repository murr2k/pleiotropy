use crate::types::{DecryptedRegion, TraitInfo, TraitSignature};
use anyhow::Result;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

pub struct TraitExtractor {
    min_region_overlap: f64,
    codon_pattern_length: usize,
}

impl TraitExtractor {
    pub fn new() -> Self {
        Self {
            min_region_overlap: 0.5,
            codon_pattern_length: 9, // 3 codons
        }
    }

    /// Extract individual traits from decrypted regions
    pub fn extract_traits(
        &self,
        decrypted_regions: &[DecryptedRegion],
        known_traits: &[TraitInfo],
    ) -> Result<Vec<TraitSignature>> {
        // Group regions by gene
        let gene_regions = self.group_by_gene(decrypted_regions);
        
        // Process each gene in parallel
        let trait_signatures: Vec<TraitSignature> = gene_regions
            .par_iter()
            .flat_map(|(gene_id, regions)| {
                self.extract_gene_traits(gene_id, regions, known_traits)
            })
            .collect();
        
        Ok(trait_signatures)
    }

    fn group_by_gene<'a>(&self, regions: &'a [DecryptedRegion]) -> HashMap<String, Vec<&'a DecryptedRegion>> {
        let mut gene_map = HashMap::new();
        
        for region in regions {
            gene_map
                .entry(region.gene_id.clone())
                .or_insert_with(Vec::new)
                .push(region);
        }
        
        gene_map
    }

    fn extract_gene_traits(
        &self,
        gene_id: &str,
        regions: &[&DecryptedRegion],
        known_traits: &[TraitInfo],
    ) -> Vec<TraitSignature> {
        let mut signatures = Vec::new();
        
        // Collect all traits mentioned in this gene's regions
        let mut trait_regions: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
        let mut trait_confidences: HashMap<String, Vec<f64>> = HashMap::new();
        
        for region in regions {
            for trait_name in &region.decrypted_traits {
                trait_regions
                    .entry(trait_name.clone())
                    .or_insert_with(Vec::new)
                    .push((region.start, region.end));
                
                if let Some(confidence) = region.confidence_scores.get(trait_name) {
                    trait_confidences
                        .entry(trait_name.clone())
                        .or_insert_with(Vec::new)
                        .push(*confidence);
                }
            }
        }
        
        // Create signature for each trait
        for (trait_name, regions) in trait_regions {
            let avg_confidence = trait_confidences
                .get(&trait_name)
                .map(|scores| scores.iter().sum::<f64>() / scores.len() as f64)
                .unwrap_or(0.0);
            
            // Find associated genes from known traits
            let associated_genes = known_traits
                .iter()
                .find(|t| t.name == trait_name)
                .map(|t| t.associated_genes.clone())
                .unwrap_or_default();
            
            // Extract codon patterns (simplified)
            let codon_patterns = self.extract_codon_patterns(&regions);
            
            signatures.push(TraitSignature {
                gene_id: gene_id.to_string(),
                trait_names: vec![trait_name],
                contributing_regions: regions,
                confidence_score: avg_confidence,
                codon_patterns,
                associated_genes,
            });
        }
        
        // Identify pleiotropic signatures (multiple traits from same regions)
        let pleiotropic_sig = self.find_pleiotropic_patterns(&signatures);
        signatures.extend(pleiotropic_sig);
        
        signatures
    }

    fn extract_codon_patterns(&self, regions: &[(usize, usize)]) -> Vec<String> {
        // In a real implementation, would extract actual sequence patterns
        let mut patterns = Vec::new();
        
        for (start, end) in regions {
            // Simulate pattern extraction
            patterns.push(format!("PATTERN_{}_{}", start, end));
        }
        
        patterns
    }

    fn find_pleiotropic_patterns(&self, signatures: &[TraitSignature]) -> Vec<TraitSignature> {
        let mut pleiotropic = Vec::new();
        
        // Find signatures with overlapping regions
        for i in 0..signatures.len() {
            for j in (i + 1)..signatures.len() {
                if let Some(combined) = self.check_pleiotropy(&signatures[i], &signatures[j]) {
                    pleiotropic.push(combined);
                }
            }
        }
        
        pleiotropic
    }

    fn check_pleiotropy(&self, sig1: &TraitSignature, sig2: &TraitSignature) -> Option<TraitSignature> {
        // Calculate region overlap
        let overlap = self.calculate_region_overlap(&sig1.contributing_regions, &sig2.contributing_regions);
        
        if overlap < self.min_region_overlap {
            return None;
        }
        
        // Combine traits
        let mut combined_traits = sig1.trait_names.clone();
        combined_traits.extend(sig2.trait_names.clone());
        combined_traits.sort();
        combined_traits.dedup();
        
        // Combine regions
        let mut all_regions = sig1.contributing_regions.clone();
        all_regions.extend(sig2.contributing_regions.clone());
        all_regions.sort_by_key(|r| r.0);
        all_regions.dedup();
        
        // Combine associated genes
        let mut all_genes: HashSet<String> = sig1.associated_genes.iter().cloned().collect();
        all_genes.extend(sig2.associated_genes.iter().cloned());
        
        // Average confidence
        let combined_confidence = (sig1.confidence_score + sig2.confidence_score) / 2.0;
        
        Some(TraitSignature {
            gene_id: sig1.gene_id.clone(),
            trait_names: combined_traits,
            contributing_regions: all_regions,
            confidence_score: combined_confidence,
            codon_patterns: vec!["PLEIOTROPIC_PATTERN".to_string()],
            associated_genes: all_genes.into_iter().collect(),
        })
    }

    fn calculate_region_overlap(&self, regions1: &[(usize, usize)], regions2: &[(usize, usize)]) -> f64 {
        let total_overlap = regions1
            .iter()
            .flat_map(|r1| {
                regions2.iter().map(move |r2| {
                    let overlap_start = r1.0.max(r2.0);
                    let overlap_end = r1.1.min(r2.1);
                    if overlap_start < overlap_end {
                        (overlap_end - overlap_start) as f64
                    } else {
                        0.0
                    }
                })
            })
            .sum::<f64>();
        
        let total_length1: usize = regions1.iter().map(|(s, e)| e - s).sum();
        let total_length2: usize = regions2.iter().map(|(s, e)| e - s).sum();
        
        if total_length1 == 0 || total_length2 == 0 {
            return 0.0;
        }
        
        total_overlap / total_length1.min(total_length2) as f64
    }

    /// Rank traits by their pleiotropic importance
    pub fn rank_pleiotropic_traits(&self, signatures: &[TraitSignature]) -> Vec<(String, f64)> {
        let mut gene_scores: HashMap<String, f64> = HashMap::new();
        
        for sig in signatures {
            let pleiotropy_score = sig.trait_names.len() as f64 * sig.confidence_score;
            *gene_scores.entry(sig.gene_id.clone()).or_insert(0.0) += pleiotropy_score;
        }
        
        let mut ranked: Vec<_> = gene_scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        ranked
    }
}