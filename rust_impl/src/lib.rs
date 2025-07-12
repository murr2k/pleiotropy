pub mod sequence_parser;
pub mod frequency_analyzer;
pub mod trait_extractor;
pub mod crypto_engine;
pub mod types;
pub mod neurodna_trait_detector;

pub use sequence_parser::SequenceParser;
pub use frequency_analyzer::FrequencyAnalyzer;
pub use trait_extractor::TraitExtractor;
pub use crypto_engine::CryptoEngine;
pub use types::*;
pub use neurodna_trait_detector::NeuroDNATraitDetector;

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
}

impl GenomicCryptanalysis {
    pub fn new() -> Self {
        Self {
            parser: SequenceParser::new(),
            analyzer: FrequencyAnalyzer::new(),
            extractor: TraitExtractor::new(),
            engine: CryptoEngine::new(),
            neurodna: NeuroDNATraitDetector::new(0.4),
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
            // Fallback to original cryptanalysis if NeuroDNA doesn't find anything
            let decrypted = self.engine.decrypt_sequences(&sequences, &freq_table)?;
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
}