use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sequence {
    pub id: String,
    pub name: String,
    pub sequence: String,
    pub annotations: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodonFrequency {
    pub codon: String,
    pub amino_acid: char,
    pub global_frequency: f64,
    pub trait_specific_frequency: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyTable {
    pub codon_frequencies: Vec<CodonFrequency>,
    pub total_codons: usize,
    pub trait_codon_counts: HashMap<String, HashMap<String, usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitInfo {
    pub name: String,
    pub description: String,
    pub associated_genes: Vec<String>,
    pub known_sequences: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecryptedRegion {
    pub start: usize,
    pub end: usize,
    pub gene_id: String,
    pub decrypted_traits: Vec<String>,
    pub confidence_scores: HashMap<String, f64>,
    pub regulatory_context: RegulatoryContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryContext {
    pub promoter_strength: f64,
    pub enhancers: Vec<(usize, usize)>,
    pub silencers: Vec<(usize, usize)>,
    pub expression_conditions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitSignature {
    pub gene_id: String,
    pub trait_names: Vec<String>,
    pub contributing_regions: Vec<(usize, usize)>,
    pub confidence_score: f64,
    pub codon_patterns: Vec<String>,
    pub associated_genes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PleiotropyAnalysis {
    pub sequences: usize,
    pub identified_traits: Vec<TraitSignature>,
    pub frequency_table: FrequencyTable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PleiotropicGene {
    pub gene_id: String,
    pub traits: Vec<String>,
    pub confidence: f64,
}

/// Genetic code mapping
pub fn genetic_code() -> HashMap<String, char> {
    let mut code = HashMap::new();
    
    // Standard genetic code
    code.insert("UUU".to_string(), 'F'); code.insert("UUC".to_string(), 'F');
    code.insert("UUA".to_string(), 'L'); code.insert("UUG".to_string(), 'L');
    code.insert("UCU".to_string(), 'S'); code.insert("UCC".to_string(), 'S');
    code.insert("UCA".to_string(), 'S'); code.insert("UCG".to_string(), 'S');
    code.insert("UAU".to_string(), 'Y'); code.insert("UAC".to_string(), 'Y');
    code.insert("UAA".to_string(), '*'); code.insert("UAG".to_string(), '*');
    code.insert("UGU".to_string(), 'C'); code.insert("UGC".to_string(), 'C');
    code.insert("UGA".to_string(), '*'); code.insert("UGG".to_string(), 'W');
    
    code.insert("CUU".to_string(), 'L'); code.insert("CUC".to_string(), 'L');
    code.insert("CUA".to_string(), 'L'); code.insert("CUG".to_string(), 'L');
    code.insert("CCU".to_string(), 'P'); code.insert("CCC".to_string(), 'P');
    code.insert("CCA".to_string(), 'P'); code.insert("CCG".to_string(), 'P');
    code.insert("CAU".to_string(), 'H'); code.insert("CAC".to_string(), 'H');
    code.insert("CAA".to_string(), 'Q'); code.insert("CAG".to_string(), 'Q');
    code.insert("CGU".to_string(), 'R'); code.insert("CGC".to_string(), 'R');
    code.insert("CGA".to_string(), 'R'); code.insert("CGG".to_string(), 'R');
    
    code.insert("AUU".to_string(), 'I'); code.insert("AUC".to_string(), 'I');
    code.insert("AUA".to_string(), 'I'); code.insert("AUG".to_string(), 'M');
    code.insert("ACU".to_string(), 'T'); code.insert("ACC".to_string(), 'T');
    code.insert("ACA".to_string(), 'T'); code.insert("ACG".to_string(), 'T');
    code.insert("AAU".to_string(), 'N'); code.insert("AAC".to_string(), 'N');
    code.insert("AAA".to_string(), 'K'); code.insert("AAG".to_string(), 'K');
    code.insert("AGU".to_string(), 'S'); code.insert("AGC".to_string(), 'S');
    code.insert("AGA".to_string(), 'R'); code.insert("AGG".to_string(), 'R');
    
    code.insert("GUU".to_string(), 'V'); code.insert("GUC".to_string(), 'V');
    code.insert("GUA".to_string(), 'V'); code.insert("GUG".to_string(), 'V');
    code.insert("GCU".to_string(), 'A'); code.insert("GCC".to_string(), 'A');
    code.insert("GCA".to_string(), 'A'); code.insert("GCG".to_string(), 'A');
    code.insert("GAU".to_string(), 'D'); code.insert("GAC".to_string(), 'D');
    code.insert("GAA".to_string(), 'E'); code.insert("GAG".to_string(), 'E');
    code.insert("GGU".to_string(), 'G'); code.insert("GGC".to_string(), 'G');
    code.insert("GGA".to_string(), 'G'); code.insert("GGG".to_string(), 'G');
    
    code
}