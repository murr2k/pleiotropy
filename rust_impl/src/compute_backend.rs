/// Unified compute backend for CPU/GPU switching
/// This module provides a unified interface that seamlessly switches between
/// CPU and CUDA implementations based on availability and user preferences.

use crate::types::*;
use crate::crypto_engine::CryptoEngine;
use crate::large_prime_factorization::{LargePrimeFactorizer, FactorizationResult, ProgressCallback};
use anyhow::Result;
use std::collections::HashMap;
use nalgebra::DVector;

#[cfg(feature = "cuda")]
use crate::cuda::{CudaAccelerator, cuda_available, should_use_cuda};

/// Unified compute backend that abstracts CPU/GPU execution
pub struct ComputeBackend {
    /// CPU implementation
    cpu_engine: CryptoEngine,
    
    /// Prime factorizer with CUDA support
    prime_factorizer: LargePrimeFactorizer,
    
    /// CUDA accelerator (if available)
    #[cfg(feature = "cuda")]
    cuda_accelerator: Option<CudaAccelerator>,
    
    /// Force CPU usage even if CUDA is available
    force_cpu: bool,
    
    /// Performance statistics
    stats: PerformanceStats,
}

#[derive(Default, Debug, Clone)]
pub struct PerformanceStats {
    pub cpu_calls: usize,
    pub cuda_calls: usize,
    pub cuda_failures: usize,
    pub total_sequences_processed: usize,
    pub avg_cpu_time_ms: f64,
    pub avg_cuda_time_ms: f64,
}

impl ComputeBackend {
    /// Create a new compute backend with automatic GPU detection
    pub fn new() -> Result<Self> {
        let cpu_engine = CryptoEngine::new();
        let prime_factorizer = LargePrimeFactorizer::new()?;
        
        #[cfg(feature = "cuda")]
        let cuda_accelerator = if should_use_cuda() {
            match CudaAccelerator::new() {
                Ok(acc) => {
                    log::info!("CUDA backend initialized successfully");
                    Some(acc)
                }
                Err(e) => {
                    log::warn!("Failed to initialize CUDA backend: {}", e);
                    None
                }
            }
        } else {
            log::info!("CUDA disabled by user preference or not available");
            None
        };
        
        Ok(Self {
            cpu_engine,
            prime_factorizer,
            #[cfg(feature = "cuda")]
            cuda_accelerator,
            force_cpu: false,
            stats: PerformanceStats::default(),
        })
    }
    
    /// Force CPU usage for testing or debugging
    pub fn set_force_cpu(&mut self, force: bool) {
        self.force_cpu = force;
        self.prime_factorizer.set_force_cpu(force);
        if force {
            log::info!("Forcing CPU backend usage");
        }
    }
    
    /// Check if CUDA is available and will be used
    pub fn is_cuda_available(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            !self.force_cpu && self.cuda_accelerator.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
    
    /// Get performance statistics
    pub fn get_stats(&self) -> &PerformanceStats {
        &self.stats
    }
    
    /// Build codon frequency vectors with GPU acceleration if available
    pub fn build_codon_vectors(
        &mut self,
        sequences: &[Sequence],
        frequency_table: &FrequencyTable,
    ) -> Result<Vec<DVector<f64>>> {
        let start = std::time::Instant::now();
        
        #[cfg(feature = "cuda")]
        {
            if self.is_cuda_available() {
                match self.build_codon_vectors_cuda(sequences, frequency_table) {
                    Ok(vectors) => {
                        self.stats.cuda_calls += 1;
                        self.stats.total_sequences_processed += sequences.len();
                        let elapsed = start.elapsed().as_millis() as f64;
                        self.update_cuda_time(elapsed);
                        log::debug!("CUDA codon vector build completed in {}ms", elapsed);
                        return Ok(vectors);
                    }
                    Err(e) => {
                        log::warn!("CUDA codon vector build failed, falling back to CPU: {}", e);
                        self.stats.cuda_failures += 1;
                    }
                }
            }
        }
        
        // CPU fallback
        let vectors = self.build_codon_vectors_cpu(sequences, frequency_table)?;
        self.stats.cpu_calls += 1;
        self.stats.total_sequences_processed += sequences.len();
        let elapsed = start.elapsed().as_millis() as f64;
        self.update_cpu_time(elapsed);
        log::debug!("CPU codon vector build completed in {}ms", elapsed);
        Ok(vectors)
    }
    
    /// Decrypt sequences using cryptanalysis with GPU acceleration if available
    pub fn decrypt_sequences(
        &mut self,
        sequences: &[Sequence],
        frequency_table: &FrequencyTable,
    ) -> Result<Vec<DecryptedRegion>> {
        let start = std::time::Instant::now();
        
        #[cfg(feature = "cuda")]
        {
            if self.is_cuda_available() {
                match self.decrypt_sequences_cuda(sequences, frequency_table) {
                    Ok(regions) => {
                        self.stats.cuda_calls += 1;
                        let elapsed = start.elapsed().as_millis() as f64;
                        self.update_cuda_time(elapsed);
                        log::info!("CUDA decryption completed in {}ms for {} sequences", elapsed, sequences.len());
                        return Ok(regions);
                    }
                    Err(e) => {
                        log::warn!("CUDA decryption failed, falling back to CPU: {}", e);
                        self.stats.cuda_failures += 1;
                    }
                }
            }
        }
        
        // CPU fallback
        let regions = self.cpu_engine.decrypt_sequences(sequences, frequency_table)?;
        self.stats.cpu_calls += 1;
        let elapsed = start.elapsed().as_millis() as f64;
        self.update_cpu_time(elapsed);
        log::info!("CPU decryption completed in {}ms for {} sequences", elapsed, sequences.len());
        Ok(regions)
    }
    
    /// Calculate codon usage bias with GPU acceleration
    pub fn calculate_codon_bias(
        &mut self,
        sequences: &[Sequence],
        traits: &[TraitInfo],
    ) -> Result<HashMap<String, Vec<f64>>> {
        #[cfg(feature = "cuda")]
        {
            if self.is_cuda_available() {
                match self.calculate_codon_bias_cuda(sequences, traits) {
                    Ok(bias_map) => {
                        self.stats.cuda_calls += 1;
                        return Ok(bias_map);
                    }
                    Err(e) => {
                        log::warn!("CUDA bias calculation failed: {}", e);
                        self.stats.cuda_failures += 1;
                    }
                }
            }
        }
        
        // CPU fallback - simplified version
        self.stats.cpu_calls += 1;
        self.calculate_codon_bias_cpu(sequences, traits)
    }
    
    // ===== CPU implementations =====
    
    fn build_codon_vectors_cpu(
        &self,
        sequences: &[Sequence],
        frequency_table: &FrequencyTable,
    ) -> Result<Vec<DVector<f64>>> {
        sequences.iter()
            .map(|seq| {
                let window = &seq.sequence;
                Ok(self.cpu_engine.build_codon_vector(window, frequency_table))
            })
            .collect()
    }
    
    fn calculate_codon_bias_cpu(
        &self,
        sequences: &[Sequence],
        traits: &[TraitInfo],
    ) -> Result<HashMap<String, Vec<f64>>> {
        let mut bias_map = HashMap::new();
        
        for trait_info in traits {
            let mut biases = Vec::new();
            for seq in sequences {
                // Simple bias calculation based on codon frequency
                let bias = self.calculate_single_bias(&seq.sequence, &trait_info.name);
                biases.push(bias);
            }
            bias_map.insert(trait_info.name.clone(), biases);
        }
        
        Ok(bias_map)
    }
    
    fn calculate_single_bias(&self, sequence: &str, _trait_name: &str) -> f64 {
        // Simplified bias calculation
        let gc_content = sequence.chars()
            .filter(|&c| c == 'G' || c == 'C')
            .count() as f64 / sequence.len() as f64;
        gc_content
    }
    
    // ===== CUDA implementations =====
    
    #[cfg(feature = "cuda")]
    fn build_codon_vectors_cuda(
        &mut self,
        sequences: &[Sequence],
        frequency_table: &FrequencyTable,
    ) -> Result<Vec<DVector<f64>>> {
        use crate::cuda::*;
        
        let cuda = self.cuda_accelerator.as_mut()
            .ok_or_else(|| anyhow::anyhow!("CUDA accelerator not initialized"))?;
        
        // Convert sequences to DnaSequence format for CUDA
        let dna_sequences: Vec<DnaSequence> = sequences.iter()
            .map(|seq| DnaSequence {
                id: seq.id.clone(),
                sequence: seq.sequence.clone(),
            })
            .collect();
        
        // Count codons on GPU
        let codon_counts = cuda.count_codons(&dna_sequences)
            .map_err(|e| anyhow::anyhow!("CUDA codon counting failed: {:?}", e))?;
        
        // Build vectors from counts
        let vectors = codon_counts.iter()
            .map(|counts| {
                let mut vector = Vec::new();
                for codon_freq in &frequency_table.codon_frequencies {
                    let count = counts.get(&codon_freq.codon).copied().unwrap_or(0) as f64;
                    let total = counts.values().sum::<u32>() as f64;
                    let observed_freq = count / total.max(1.0);
                    let expected_freq = codon_freq.global_frequency;
                    
                    let log_odds = if observed_freq > 0.0 && expected_freq > 0.0 {
                        (observed_freq / expected_freq).ln()
                    } else {
                        0.0
                    };
                    
                    vector.push(log_odds);
                }
                DVector::from_vec(vector)
            })
            .collect();
        
        Ok(vectors)
    }
    
    #[cfg(feature = "cuda")]
    fn decrypt_sequences_cuda(
        &mut self,
        sequences: &[Sequence],
        frequency_table: &FrequencyTable,
    ) -> Result<Vec<DecryptedRegion>> {
        use crate::cuda::*;
        
        let cuda = self.cuda_accelerator.as_mut()
            .ok_or_else(|| anyhow::anyhow!("CUDA accelerator not initialized"))?;
        
        // Convert to CUDA types
        let dna_sequences: Vec<DnaSequence> = sequences.iter()
            .map(|seq| DnaSequence {
                id: seq.id.clone(),
                sequence: seq.sequence.clone(),
            })
            .collect();
        
        // Build CUDA frequency table
        let cuda_freq_table = self.convert_to_cuda_frequency_table(frequency_table)?;
        
        // Process with sliding windows on GPU
        let window_size = 300; // 100 codons
        let window_stride = 150; // 50% overlap
        
        let mut all_regions = Vec::new();
        
        for (seq_idx, seq) in sequences.iter().enumerate() {
            // Calculate frequencies for this sequence
            let seq_frequencies = self.calculate_sequence_frequencies(&seq.sequence, frequency_table)?;
            
            // Match patterns using sliding windows
            let matches = cuda.match_patterns_sliding_window(
                &seq_frequencies,
                &self.get_trait_patterns(),
                window_size,
                window_stride,
                seq.sequence.len(),
            ).map_err(|e| anyhow::anyhow!("CUDA pattern matching failed: {:?}", e))?;
            
            // Convert matches to DecryptedRegions
            for (trait_name, position, confidence) in matches {
                if confidence >= self.cpu_engine.min_confidence {
                    let mut confidence_scores = HashMap::new();
                    confidence_scores.insert(trait_name.clone(), confidence);
                    
                    all_regions.push(DecryptedRegion {
                        start: position,
                        end: (position + window_size).min(seq.sequence.len()),
                        gene_id: seq.id.clone(),
                        decrypted_traits: vec![trait_name],
                        confidence_scores,
                        regulatory_context: RegulatoryContext::default(),
                    });
                }
            }
        }
        
        Ok(all_regions)
    }
    
    #[cfg(feature = "cuda")]
    fn calculate_codon_bias_cuda(
        &mut self,
        sequences: &[Sequence],
        traits: &[TraitInfo],
    ) -> Result<HashMap<String, Vec<f64>>> {
        use crate::cuda::*;
        
        let cuda = self.cuda_accelerator.as_mut()
            .ok_or_else(|| anyhow::anyhow!("CUDA accelerator not initialized"))?;
        
        // Convert sequences
        let dna_sequences: Vec<DnaSequence> = sequences.iter()
            .map(|seq| DnaSequence {
                id: seq.id.clone(),
                sequence: seq.sequence.clone(),
            })
            .collect();
        
        // Count codons
        let codon_counts = cuda.count_codons(&dna_sequences)
            .map_err(|e| anyhow::anyhow!("CUDA counting failed: {:?}", e))?;
        
        // Calculate frequencies
        let cuda_freq_table = cuda.calculate_frequencies(&codon_counts, traits)
            .map_err(|e| anyhow::anyhow!("CUDA frequency calculation failed: {:?}", e))?;
        
        // Convert back to bias map
        self.convert_cuda_frequencies_to_bias_map(&cuda_freq_table, traits)
    }
    
    // ===== Helper methods =====
    
    #[cfg(feature = "cuda")]
    fn convert_to_cuda_frequency_table(&self, freq_table: &FrequencyTable) -> Result<CudaFrequencyTable> {
        // This is a placeholder - actual implementation would convert the frequency table
        // to the format expected by CUDA kernels
        Ok(CudaFrequencyTable {
            codon_frequencies: freq_table.codon_frequencies.iter()
                .map(|cf| (cf.codon.clone(), cf.global_frequency as f32))
                .collect(),
            trait_frequencies: HashMap::new(),
        })
    }
    
    #[cfg(feature = "cuda")]
    fn calculate_sequence_frequencies(&self, sequence: &str, freq_table: &FrequencyTable) -> Result<Vec<f32>> {
        // Calculate frequency vector for the sequence
        let mut frequencies = Vec::new();
        let codon_counts = self.count_codons_in_sequence(sequence);
        let total = codon_counts.values().sum::<usize>() as f32;
        
        for codon_freq in &freq_table.codon_frequencies {
            let count = codon_counts.get(&codon_freq.codon).copied().unwrap_or(0) as f32;
            frequencies.push(count / total.max(1.0));
        }
        
        Ok(frequencies)
    }
    
    #[cfg(feature = "cuda")]
    fn count_codons_in_sequence(&self, sequence: &str) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for i in (0..sequence.len()).step_by(3) {
            if i + 3 <= sequence.len() {
                let codon = &sequence[i..i + 3];
                if codon.chars().all(|c| "ATCG".contains(c)) {
                    *counts.entry(codon.to_string()).or_insert(0) += 1;
                }
            }
        }
        counts
    }
    
    #[cfg(feature = "cuda")]
    fn get_trait_patterns(&self) -> Vec<TraitPattern> {
        // Define trait patterns for CUDA matching
        vec![
            TraitPattern {
                name: "carbon_metabolism".to_string(),
                codon_preferences: vec![("ATG".to_string(), 1.2)],
                min_score: 0.7,
            },
            TraitPattern {
                name: "stress_response".to_string(),
                codon_preferences: vec![("GCG".to_string(), 1.1)],
                min_score: 0.6,
            },
            TraitPattern {
                name: "motility".to_string(),
                codon_preferences: vec![("TTT".to_string(), 1.0)],
                min_score: 0.65,
            },
            TraitPattern {
                name: "regulatory".to_string(),
                codon_preferences: vec![("GAA".to_string(), 1.15)],
                min_score: 0.5,
            },
        ]
    }
    
    #[cfg(feature = "cuda")]
    fn convert_cuda_frequencies_to_bias_map(
        &self,
        cuda_freq_table: &CudaFrequencyTable,
        traits: &[TraitInfo],
    ) -> Result<HashMap<String, Vec<f64>>> {
        let mut bias_map = HashMap::new();
        
        // This is a simplified conversion - actual implementation would properly
        // extract trait-specific biases from the CUDA frequency table
        for trait_info in traits {
            let biases = vec![0.5; cuda_freq_table.codon_frequencies.len()];
            bias_map.insert(trait_info.name.clone(), biases);
        }
        
        Ok(bias_map)
    }
    
    fn update_cpu_time(&mut self, elapsed_ms: f64) {
        let n = self.stats.cpu_calls as f64;
        self.stats.avg_cpu_time_ms = 
            (self.stats.avg_cpu_time_ms * (n - 1.0) + elapsed_ms) / n;
    }
    
    fn update_cuda_time(&mut self, elapsed_ms: f64) {
        let n = self.stats.cuda_calls as f64;
        self.stats.avg_cuda_time_ms = 
            (self.stats.avg_cuda_time_ms * (n - 1.0) + elapsed_ms) / n;
    }
    
    // ===== Prime factorization methods =====
    
    /// Factorize a 64-bit integer using GPU acceleration if available
    pub fn factorize_u64(&mut self, number: u64) -> Result<FactorizationResult> {
        let start = std::time::Instant::now();
        let result = self.prime_factorizer.factorize_u64(number)?;
        
        if result.used_cuda {
            self.stats.cuda_calls += 1;
            self.update_cuda_time(result.elapsed_ms);
        } else {
            self.stats.cpu_calls += 1;
            self.update_cpu_time(result.elapsed_ms);
        }
        
        Ok(result)
    }
    
    /// Factorize a 128-bit integer
    pub fn factorize_u128(&mut self, number: u128) -> Result<FactorizationResult> {
        let start = std::time::Instant::now();
        let result = self.prime_factorizer.factorize_u128(number)?;
        
        if result.used_cuda {
            self.stats.cuda_calls += 1;
            self.update_cuda_time(result.elapsed_ms);
        } else {
            self.stats.cpu_calls += 1;
            self.update_cpu_time(result.elapsed_ms);
        }
        
        Ok(result)
    }
    
    /// Batch factorization with progress reporting
    pub fn factorize_batch(&mut self, numbers: &[u64]) -> Result<Vec<FactorizationResult>> {
        let results = self.prime_factorizer.factorize_batch(numbers)?;
        
        // Update statistics
        for result in &results {
            if result.used_cuda {
                self.stats.cuda_calls += 1;
                self.update_cuda_time(result.elapsed_ms);
            } else {
                self.stats.cpu_calls += 1;
                self.update_cpu_time(result.elapsed_ms);
            }
        }
        
        Ok(results)
    }
    
    /// Add a progress callback for factorization operations
    pub fn add_factorization_progress_callback(&mut self, callback: ProgressCallback) {
        self.prime_factorizer.add_progress_callback(callback);
    }
    
    /// Async factorization for non-blocking operations
    pub async fn factorize_u64_async(&self, number: u64) -> Result<FactorizationResult> {
        self.prime_factorizer.factorize_u64_async(number).await
    }
    
    /// Async factorization for 128-bit numbers
    pub async fn factorize_u128_async(&self, number: u128) -> Result<FactorizationResult> {
        self.prime_factorizer.factorize_u128_async(number).await
    }
}

// CUDA type definitions that are used in the compute backend
#[cfg(feature = "cuda")]
use crate::cuda::{DnaSequence, CodonCounts, CudaFrequencyTable, TraitPattern, PatternMatch};

// Define these types for non-CUDA builds to avoid compilation errors
#[cfg(not(feature = "cuda"))]
mod cuda_types {
    use std::collections::HashMap;
    
    pub struct DnaSequence {
        pub id: String,
        pub sequence: String,
    }
    
    pub type CodonCounts = HashMap<String, u32>;
    
    pub struct CudaFrequencyTable {
        pub codon_frequencies: Vec<(String, f32)>,
        pub trait_frequencies: HashMap<String, Vec<f32>>,
    }
    
    pub struct TraitPattern {
        pub name: String,
        pub codon_preferences: Vec<(String, f32)>,
        pub min_score: f32,
    }
    
    pub struct PatternMatch {
        pub trait_name: String,
        pub position: usize,
        pub score: f32,
    }
}

#[cfg(not(feature = "cuda"))]
use cuda_types::*;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_compute_backend_creation() {
        let backend = ComputeBackend::new().unwrap();
        println!("CUDA available: {}", backend.is_cuda_available());
        assert!(true); // Backend should always create successfully
    }
    
    #[test]
    fn test_force_cpu() {
        let mut backend = ComputeBackend::new().unwrap();
        backend.set_force_cpu(true);
        assert!(!backend.is_cuda_available());
    }
}