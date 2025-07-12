pub mod sequence_parser;
pub mod frequency_analyzer;
pub mod trait_extractor;
pub mod trait_extractor_gpu;
pub mod crypto_engine;
pub mod types;
pub mod neurodna_trait_detector;
pub mod compute_backend;
#[cfg(feature = "cuda")]
pub mod cuda;

pub use sequence_parser::SequenceParser;
pub use frequency_analyzer::FrequencyAnalyzer;
pub use trait_extractor::TraitExtractor;
pub use crypto_engine::CryptoEngine;
pub use types::*;
pub use neurodna_trait_detector::NeuroDNATraitDetector;
pub use compute_backend::ComputeBackend;

use anyhow::Result;
use std::path::Path;
use std::collections::{HashMap, HashSet};

/// Main API for genomic cryptanalysis
pub struct GenomicCryptanalysis {
    parser: SequenceParser,
    analyzer: FrequencyAnalyzer,
    extractor: TraitExtractor,
    engine: CryptoEngine,
    neurodna: NeuroDNATraitDetector,
    compute_backend: ComputeBackend,
}

impl GenomicCryptanalysis {
    pub fn new() -> Self {
        let compute_backend = match ComputeBackend::new() {
            Ok(backend) => {
                if backend.is_cuda_available() {
                    log::info!("CUDA acceleration enabled");
                } else {
                    log::info!("Using CPU backend");
                }
                backend
            }
            Err(e) => {
                log::warn!("Failed to create compute backend: {}, using CPU fallback", e);
                ComputeBackend::new().unwrap() // Should always succeed with CPU
            }
        };
        
        Self {
            parser: SequenceParser::new(),
            analyzer: FrequencyAnalyzer::new(),
            extractor: TraitExtractor::new(),
            engine: CryptoEngine::new(),
            neurodna: NeuroDNATraitDetector::new(0.4),
            compute_backend,
        }
    }

    /// Load and analyze a genome file
    pub fn analyze_genome<P: AsRef<Path>>(
        &mut self,
        genome_path: P,
        known_traits: Vec<TraitInfo>,
    ) -> Result<PleiotropyAnalysis> {
        // Parse sequences
        let sequences = self.parser.parse_file(genome_path)?;
        
        // Build frequency tables
        let mut freq_table = self.analyzer.build_frequency_table(&sequences)?;
        
        // Calculate trait-specific frequencies
        self.analyzer.calculate_trait_bias(&sequences, &known_traits, &mut freq_table)?;
        
        // First try NeuroDNA-based detection
        let pleiotropic_genes = self.neurodna.analyze_sequences(&sequences, &known_traits);
        
        // Convert NeuroDNA results to TraitSignatures if found
        let traits = if !pleiotropic_genes.is_empty() {
            pleiotropic_genes.iter().map(|gene| {
                TraitSignature {
                    gene_id: gene.gene_id.clone(),
                    trait_names: gene.traits.clone(),
                    contributing_regions: vec![],
                    confidence_score: gene.confidence,
                    codon_patterns: vec![],
                    associated_genes: vec![],
                }
            }).collect()
        } else {
            // Use compute backend for GPU-accelerated cryptanalysis
            let decrypted = self.compute_backend.decrypt_sequences(&sequences, &freq_table)?;
            self.extractor.extract_traits(&decrypted, &known_traits)?
        };
        
        Ok(PleiotropyAnalysis {
            sequences: sequences.len(),
            identified_traits: traits,
            frequency_table: freq_table,
        })
    }
    
    /// Identify pleiotropic genes
    pub fn find_pleiotropic_genes(
        &self,
        analysis: &PleiotropyAnalysis,
        min_traits: usize,
    ) -> Vec<PleiotropicGene> {
        // Group traits by gene to properly detect pleiotropy
        let mut gene_traits: HashMap<String, Vec<TraitSignature>> = HashMap::new();
        
        for trait_sig in &analysis.identified_traits {
            gene_traits
                .entry(trait_sig.gene_id.clone())
                .or_insert_with(Vec::new)
                .push(trait_sig.clone());
        }
        
        // Find genes affecting multiple traits
        let mut pleiotropic_genes = Vec::new();
        
        for (gene_id, signatures) in gene_traits {
            // Collect all unique traits for this gene
            let mut all_traits: HashSet<String> = HashSet::new();
            let mut total_confidence = 0.0;
            let mut count = 0;
            
            for sig in &signatures {
                all_traits.extend(sig.trait_names.iter().cloned());
                total_confidence += sig.confidence_score;
                count += 1;
            }
            
            // Check if gene affects minimum number of traits
            if all_traits.len() >= min_traits {
                pleiotropic_genes.push(PleiotropicGene {
                    gene_id,
                    traits: all_traits.into_iter().collect(),
                    confidence: total_confidence / count.max(1) as f64,
                });
            }
        }
        
        // Sort by number of traits (descending) then by confidence
        pleiotropic_genes.sort_by(|a, b| {
            b.traits.len().cmp(&a.traits.len())
                .then(b.confidence.partial_cmp(&a.confidence).unwrap())
        });
        
        pleiotropic_genes
    }
    
    /// Get performance statistics from the compute backend
    pub fn get_performance_stats(&self) -> &compute_backend::PerformanceStats {
        self.compute_backend.get_stats()
    }
    
    /// Check if CUDA is available and being used
    pub fn is_cuda_enabled(&self) -> bool {
        self.compute_backend.is_cuda_available()
    }
    
    /// Force CPU usage even if CUDA is available
    pub fn set_force_cpu(&mut self, force: bool) {
        self.compute_backend.set_force_cpu(force);
    }
}