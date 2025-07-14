// CUDA acceleration module for genomic cryptanalysis

pub mod features;

#[cfg(feature = "cuda")]
pub mod device;
#[cfg(feature = "cuda")]
pub mod prime_factorization;
#[cfg(feature = "cuda")]
pub mod composite_factorizer;
#[cfg(feature = "cuda")]
pub mod semiprime_cuda;
#[cfg(feature = "cuda")]
pub mod kernels;
#[cfg(feature = "cuda")]
pub mod memory;
#[cfg(feature = "cuda")]
pub mod error;

#[cfg(feature = "cuda")]
pub use device::CudaDevice;
#[cfg(feature = "cuda")]
pub use error::{CudaError, CudaResult};
#[cfg(feature = "cuda")]
pub use memory::{CudaBuffer, PinnedBuffer};

use crate::types::*;
use anyhow::Result;

/// Main CUDA accelerator interface
#[cfg(feature = "cuda")]
pub struct CudaAccelerator {
    device: CudaDevice,
    codon_counter: kernels::CodonCounter,
    frequency_calculator: kernels::FrequencyCalculator,
    pattern_matcher: kernels::PatternMatcher,
    matrix_processor: kernels::MatrixProcessor,
}

#[cfg(feature = "cuda")]
impl CudaAccelerator {
    pub fn new() -> CudaResult<Self> {
        let device = CudaDevice::new(0)?; // Use first GPU
        
        Ok(Self {
            codon_counter: kernels::CodonCounter::new(&device)?,
            frequency_calculator: kernels::FrequencyCalculator::new(&device)?,
            pattern_matcher: kernels::PatternMatcher::new(&device)?,
            matrix_processor: kernels::MatrixProcessor::new(&device)?,
            device,
        })
    }
    
    /// Count codons in parallel on GPU
    pub fn count_codons(&mut self, sequences: &[DnaSequence]) -> CudaResult<Vec<CodonCounts>> {
        self.codon_counter.count(sequences)
    }
    
    /// Calculate frequency tables on GPU
    pub fn calculate_frequencies(
        &mut self,
        codon_counts: &[CodonCounts],
        traits: &[TraitInfo],
    ) -> CudaResult<CudaFrequencyTable> {
        self.frequency_calculator.calculate(codon_counts, traits)
    }
    
    /// Match patterns in parallel
    pub fn match_patterns(
        &mut self,
        frequency_table: &CudaFrequencyTable,
        trait_patterns: &[TraitPattern],
    ) -> CudaResult<Vec<PatternMatch>> {
        self.pattern_matcher.match_patterns(frequency_table, trait_patterns)
    }
    
    /// Perform eigenanalysis on correlation matrix
    pub fn eigenanalysis(
        &mut self,
        correlation_matrix: &[f32],
        size: usize,
    ) -> CudaResult<(Vec<f32>, Vec<f32>)> {
        self.matrix_processor.eigendecompose(correlation_matrix, size)
    }
    
    /// Match patterns using sliding windows
    pub fn match_patterns_sliding_window(
        &mut self,
        sequence_frequencies: &[f32],
        trait_patterns: &[TraitPattern],
        window_size: usize,
        window_stride: usize,
        seq_length: usize,
    ) -> CudaResult<Vec<(String, usize, f64)>> {
        self.pattern_matcher.match_patterns_sliding_window(
            sequence_frequencies,
            trait_patterns,
            window_size,
            window_stride,
            seq_length,
        )
    }
    
    /// Match NeuroDNA trait patterns
    pub fn match_neurodna_patterns(
        &mut self,
        frequency_table: &CudaFrequencyTable,
        neurodna_traits: &std::collections::HashMap<String, Vec<String>>,
    ) -> CudaResult<Vec<PatternMatch>> {
        self.pattern_matcher.match_neurodna_patterns(frequency_table, neurodna_traits)
    }
    
    /// Perform PCA-based trait separation
    pub fn pca_trait_separation(
        &mut self,
        codon_frequencies: &[f32],
        num_sequences: usize,
        num_codons: usize,
        num_components: usize,
    ) -> CudaResult<(Vec<f32>, Vec<f32>)> {
        self.matrix_processor.pca_trait_separation(
            codon_frequencies,
            num_sequences,
            num_codons,
            num_components,
        )
    }
    
    /// Identify trait components using eigenanalysis
    pub fn identify_trait_components(
        &mut self,
        codon_frequencies: &[f32],
        num_sequences: usize,
        num_codons: usize,
        variance_threshold: f32,
    ) -> CudaResult<Vec<(usize, f32, Vec<f32>)>> {
        self.matrix_processor.identify_trait_components(
            codon_frequencies,
            num_sequences,
            num_codons,
            variance_threshold,
        )
    }
    
    /// Count codons using sliding windows
    pub fn count_codons_sliding_windows(
        &mut self,
        sequences: &[DnaSequence],
        window_size: usize,
        window_stride: usize,
    ) -> CudaResult<Vec<Vec<CodonCounts>>> {
        self.codon_counter.count_sliding_windows(sequences, window_size, window_stride)
    }
}

/// CPU fallback implementation
#[cfg(not(feature = "cuda"))]
pub struct CudaAccelerator;

#[cfg(not(feature = "cuda"))]
impl CudaAccelerator {
    pub fn new() -> Result<Self> {
        log::warn!("CUDA support not enabled, using CPU fallback");
        Ok(Self)
    }
    
    pub fn count_codons(&mut self, _sequences: &[DnaSequence]) -> Result<Vec<CodonCounts>> {
        Err(anyhow::anyhow!("CUDA not available - use CPU implementation"))
    }
    
    pub fn calculate_frequencies(
        &mut self,
        _codon_counts: &[CodonCounts],
        _traits: &[TraitInfo],
    ) -> Result<CudaFrequencyTable> {
        Err(anyhow::anyhow!("CUDA not available - use CPU implementation"))
    }
    
    pub fn match_patterns(
        &mut self,
        _frequency_table: &CudaFrequencyTable,
        _trait_patterns: &[TraitPattern],
    ) -> Result<Vec<PatternMatch>> {
        Err(anyhow::anyhow!("CUDA not available - use CPU implementation"))
    }
    
    pub fn eigenanalysis(
        &mut self,
        _correlation_matrix: &[f32],
        _size: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        Err(anyhow::anyhow!("CUDA not available - use CPU implementation"))
    }
}

/// Check if CUDA is available on the system
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        CudaDevice::count() > 0
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get CUDA device information
pub fn cuda_info() -> Option<String> {
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = CudaDevice::new(0) {
            Some(device.info())
        } else {
            None
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}