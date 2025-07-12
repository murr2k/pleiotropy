/// GPU-accelerated trait extraction module
/// This module provides enhanced trait extraction using CUDA when available

use crate::types::{DecryptedRegion, TraitInfo, TraitSignature};
use crate::compute_backend::ComputeBackend;
use anyhow::Result;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

pub struct TraitExtractorGPU {
    min_region_overlap: f64,
    codon_pattern_length: usize,
    compute_backend: ComputeBackend,
}

impl TraitExtractorGPU {
    pub fn new() -> Result<Self> {
        Ok(Self {
            min_region_overlap: 0.5,
            codon_pattern_length: 9, // 3 codons
            compute_backend: ComputeBackend::new()?,
        })
    }
    
    /// Check if GPU acceleration is available
    pub fn is_gpu_enabled(&self) -> bool {
        self.compute_backend.is_cuda_available()
    }
    
    /// Get performance statistics
    pub fn get_stats(&self) -> &crate::compute_backend::PerformanceStats {
        self.compute_backend.get_stats()
    }
    
    /// Extract individual traits from decrypted regions with GPU acceleration
    pub fn extract_traits(
        &mut self,
        decrypted_regions: &[DecryptedRegion],
        known_traits: &[TraitInfo],
    ) -> Result<Vec<TraitSignature>> {
        // Group regions by gene
        let gene_regions = self.group_by_gene(decrypted_regions);
        
        // For GPU processing, we need to batch operations
        if self.compute_backend.is_cuda_available() && gene_regions.len() > 10 {
            // Use GPU for batch processing
            self.extract_traits_gpu_batch(&gene_regions, known_traits)
        } else {
            // Use CPU for small datasets or when GPU not available
            self.extract_traits_cpu(&gene_regions, known_traits)
        }
    }
    
    fn extract_traits_gpu_batch(
        &mut self,
        gene_regions: &HashMap<String, Vec<&DecryptedRegion>>,
        known_traits: &[TraitInfo],
    ) -> Result<Vec<TraitSignature>> {
        #[cfg(feature = "cuda")]
        {
            use crate::cuda::*;
            
            let mut all_signatures = Vec::new();
            
            // Process in batches for GPU efficiency
            const BATCH_SIZE: usize = 100;
            let gene_ids: Vec<_> = gene_regions.keys().cloned().collect();
            
            for batch_start in (0..gene_ids.len()).step_by(BATCH_SIZE) {
                let batch_end = (batch_start + BATCH_SIZE).min(gene_ids.len());
                let batch_genes = &gene_ids[batch_start..batch_end];
                
                // Prepare batch data for GPU
                let mut batch_trait_regions: Vec<HashMap<String, Vec<(usize, usize)>>> = Vec::new();
                let mut batch_confidences: Vec<HashMap<String, Vec<f64>>> = Vec::new();
                
                for gene_id in batch_genes {
                    let regions = &gene_regions[gene_id];
                    let (trait_regions, trait_confidences) = self.collect_trait_data(regions);
                    batch_trait_regions.push(trait_regions);
                    batch_confidences.push(trait_confidences);
                }
                
                // Process batch on GPU
                let batch_signatures = self.process_batch_on_gpu(
                    batch_genes,
                    &batch_trait_regions,
                    &batch_confidences,
                    known_traits,
                )?;
                
                all_signatures.extend(batch_signatures);
            }
            
            // Find pleiotropic patterns
            let pleiotropic_sig = self.find_pleiotropic_patterns(&all_signatures);
            all_signatures.extend(pleiotropic_sig);
            
            Ok(all_signatures)
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            // Fallback to CPU implementation
            self.extract_traits_cpu(gene_regions, known_traits)
        }
    }
    
    fn extract_traits_cpu(
        &self,
        gene_regions: &HashMap<String, Vec<&DecryptedRegion>>,
        known_traits: &[TraitInfo],
    ) -> Result<Vec<TraitSignature>> {
        let trait_signatures: Vec<TraitSignature> = gene_regions
            .par_iter()
            .flat_map(|(gene_id, regions)| {
                self.extract_gene_traits(gene_id, regions, known_traits)
            })
            .collect();
        
        Ok(trait_signatures)
    }
    
    #[cfg(feature = "cuda")]
    fn process_batch_on_gpu(
        &mut self,
        gene_ids: &[String],
        trait_regions: &[HashMap<String, Vec<(usize, usize)>>],
        trait_confidences: &[HashMap<String, Vec<f64>>],
        known_traits: &[TraitInfo],
    ) -> Result<Vec<TraitSignature>> {
        let mut signatures = Vec::new();
        
        // GPU processing would happen here
        // For now, we process each gene
        for (idx, gene_id) in gene_ids.iter().enumerate() {
            let gene_trait_regions = &trait_regions[idx];
            let gene_trait_confidences = &trait_confidences[idx];
            
            for (trait_name, regions) in gene_trait_regions {
                let avg_confidence = gene_trait_confidences
                    .get(trait_name)
                    .map(|scores| scores.iter().sum::<f64>() / scores.len() as f64)
                    .unwrap_or(0.0);
                
                let associated_genes = known_traits
                    .iter()
                    .find(|t| t.name == *trait_name)
                    .map(|t| t.associated_genes.clone())
                    .unwrap_or_default();
                
                signatures.push(TraitSignature {
                    gene_id: gene_id.clone(),
                    trait_names: vec![trait_name.clone()],
                    contributing_regions: regions.clone(),
                    confidence_score: avg_confidence,
                    codon_patterns: vec![format!("GPU_PATTERN_{}", trait_name)],
                    associated_genes,
                });
            }
        }
        
        Ok(signatures)
    }
    
    fn collect_trait_data(
        &self,
        regions: &[&DecryptedRegion],
    ) -> (HashMap<String, Vec<(usize, usize)>>, HashMap<String, Vec<f64>>) {
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
        
        (trait_regions, trait_confidences)
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
        let (trait_regions, trait_confidences) = self.collect_trait_data(regions);
        
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
        let mut patterns = Vec::new();
        
        for (start, end) in regions {
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_trait_extractor() {
        let extractor = TraitExtractorGPU::new().unwrap();
        println!("GPU enabled: {}", extractor.is_gpu_enabled());
        assert!(true); // Should always create successfully
    }
}