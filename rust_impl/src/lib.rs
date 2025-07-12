pub mod sequence_parser;
pub mod frequency_analyzer;
pub mod trait_extractor;
pub mod crypto_engine;
pub mod types;

pub use sequence_parser::SequenceParser;
pub use frequency_analyzer::FrequencyAnalyzer;
pub use trait_extractor::TraitExtractor;
pub use crypto_engine::CryptoEngine;
pub use types::*;

use anyhow::Result;
use std::path::Path;

/// Main API for genomic cryptanalysis
pub struct GenomicCryptanalysis {
    parser: SequenceParser,
    analyzer: FrequencyAnalyzer,
    extractor: TraitExtractor,
    engine: CryptoEngine,
}

impl GenomicCryptanalysis {
    pub fn new() -> Self {
        Self {
            parser: SequenceParser::new(),
            analyzer: FrequencyAnalyzer::new(),
            extractor: TraitExtractor::new(),
            engine: CryptoEngine::new(),
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
        let freq_table = self.analyzer.build_frequency_table(&sequences)?;
        
        // Decrypt sequences using cryptanalysis
        let decrypted = self.engine.decrypt_sequences(&sequences, &freq_table)?;
        
        // Extract individual traits
        let traits = self.extractor.extract_traits(&decrypted, &known_traits)?;
        
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
        analysis
            .identified_traits
            .iter()
            .filter(|t| t.associated_genes.len() >= min_traits)
            .map(|t| PleiotropicGene {
                gene_id: t.gene_id.clone(),
                traits: t.trait_names.clone(),
                confidence: t.confidence_score,
            })
            .collect()
    }
}