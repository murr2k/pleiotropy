use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use std::sync::Arc;
use std::collections::HashMap;

use crate::cuda::{CudaBuffer, CudaError, CudaResult};
use crate::types::{CudaFrequencyTable, TraitPattern, PatternMatch};

// CUDA kernel for pattern matching - optimized for GTX 2070
const PATTERN_MATCH_KERNEL: &str = r#"
// Enhanced pattern matching kernel with NeuroDNA trait pattern support
extern "C" __global__ void pattern_match_kernel(
    const float* frequency_tables,
    const float* trait_patterns,
    float* match_scores,
    const unsigned int num_sequences,
    const unsigned int num_patterns,
    const unsigned int pattern_length,
    const unsigned int num_codons
) {
    // Thread configuration optimized for GTX 2070
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid & 31;
    const int global_tid = bid * blockDim.x + tid;
    const int total_threads = gridDim.x * blockDim.x;
    
    // Shared memory for pattern caching and reduction
    extern __shared__ float shared_mem[];
    float* shared_patterns = shared_mem;
    float* warp_scores = &shared_mem[blockDim.x * pattern_length];
    
    // Cooperatively load patterns into shared memory
    const int patterns_per_block = min(num_patterns, blockDim.x / 32); // One pattern per warp
    const int pattern_elements_per_thread = (pattern_length + blockDim.x - 1) / blockDim.x;
    
    if (warp_id < patterns_per_block) {
        for (int i = 0; i < pattern_elements_per_thread; i++) {
            int elem_idx = tid + i * blockDim.x;
            if (elem_idx < pattern_length) {
                shared_patterns[warp_id * pattern_length + elem_idx] = 
                    trait_patterns[warp_id * pattern_length + elem_idx];
            }
        }
    }
    __syncthreads();
    
    // Grid-stride loop for processing sequence-pattern pairs
    for (int work_idx = global_tid; work_idx < num_sequences * num_patterns; work_idx += total_threads) {
        const int seq_idx = work_idx / num_patterns;
        const int pattern_idx = work_idx % num_patterns;
        
        if (seq_idx >= num_sequences || pattern_idx >= num_patterns) continue;
        
        const float* seq_freq = &frequency_tables[seq_idx * num_codons];
        const float* pattern = (pattern_idx < patterns_per_block) ?
            &shared_patterns[pattern_idx * pattern_length] :
            &trait_patterns[pattern_idx * pattern_length];
        
        // Enhanced similarity calculation with weighted components
        float cosine_similarity = 0.0f;
        float chi_squared = 0.0f;
        float kl_divergence = 0.0f;
        
        // Warp-level parallel reduction for dot product and magnitudes
        float local_dot = 0.0f;
        float local_seq_mag = 0.0f;
        float local_pat_mag = 0.0f;
        float local_chi = 0.0f;
        float local_kl = 0.0f;
        
        // Each thread processes multiple codons
        for (int i = lane_id; i < pattern_length && i < num_codons; i += 32) {
            float s = seq_freq[i];
            float p = pattern[i];
            
            local_dot += s * p;
            local_seq_mag += s * s;
            local_pat_mag += p * p;
            
            // Chi-squared component
            if (p > 1e-6f) {
                float diff = s - p;
                local_chi += (diff * diff) / p;
            }
            
            // KL divergence component
            if (s > 1e-6f && p > 1e-6f) {
                local_kl += s * __logf(s / p);
            }
        }
        
        // Warp-level reduction using shuffle operations
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_dot += __shfl_down_sync(0xFFFFFFFF, local_dot, offset);
            local_seq_mag += __shfl_down_sync(0xFFFFFFFF, local_seq_mag, offset);
            local_pat_mag += __shfl_down_sync(0xFFFFFFFF, local_pat_mag, offset);
            local_chi += __shfl_down_sync(0xFFFFFFFF, local_chi, offset);
            local_kl += __shfl_down_sync(0xFFFFFFFF, local_kl, offset);
        }
        
        // First thread in warp computes final score
        if (lane_id == 0) {
            // Cosine similarity component
            float seq_magnitude = sqrtf(local_seq_mag);
            float pattern_magnitude = sqrtf(local_pat_mag);
            
            if (seq_magnitude > 1e-6f && pattern_magnitude > 1e-6f) {
                cosine_similarity = local_dot / (seq_magnitude * pattern_magnitude);
            }
            
            // Normalize chi-squared (convert to similarity)
            chi_squared = expf(-local_chi / 64.0f); // 64 codons
            
            // Normalize KL divergence (convert to similarity)
            kl_divergence = expf(-local_kl);
            
            // Weighted combination of metrics
            float weight_cosine = 0.5f;
            float weight_chi = 0.3f;
            float weight_kl = 0.2f;
            
            float combined_score = weight_cosine * cosine_similarity + 
                                 weight_chi * chi_squared + 
                                 weight_kl * kl_divergence;
            
            // Apply sigmoid for final score normalization
            combined_score = 1.0f / (1.0f + expf(-10.0f * (combined_score - 0.5f)));
            
            // Store match score
            match_scores[seq_idx * num_patterns + pattern_idx] = combined_score;
        }
    }
}

// Sliding window pattern matching for detecting trait regions
extern "C" __global__ void sliding_window_pattern_kernel(
    const float* sequence_frequencies,
    const float* trait_patterns,
    float* window_scores,
    unsigned int* best_positions,
    const unsigned int seq_length,
    const unsigned int num_patterns,
    const unsigned int window_size,
    const unsigned int window_stride,
    const unsigned int num_codons
) {
    const int pattern_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (pattern_idx >= num_patterns) return;
    
    // Shared memory for pattern and reduction
    extern __shared__ float shared_data[];
    float* pattern_cache = shared_data;
    float* window_cache = &shared_data[num_codons];
    float* score_cache = &shared_data[2 * num_codons];
    
    // Load pattern into shared memory
    for (int i = tid; i < num_codons; i += blockDim.x) {
        pattern_cache[i] = trait_patterns[pattern_idx * num_codons + i];
    }
    __syncthreads();
    
    // Initialize best score tracking
    __shared__ float best_score;
    __shared__ unsigned int best_pos;
    
    if (tid == 0) {
        best_score = -1.0f;
        best_pos = 0;
    }
    __syncthreads();
    
    // Process windows
    const int num_windows = (seq_length > window_size) ? 
        ((seq_length - window_size) / window_stride + 1) : 1;
    
    for (int window_idx = 0; window_idx < num_windows; window_idx++) {
        const int window_start = window_idx * window_stride;
        
        // Load window frequencies
        for (int i = tid; i < num_codons; i += blockDim.x) {
            int pos = window_start * num_codons + i;
            window_cache[i] = (pos < seq_length * num_codons) ? 
                sequence_frequencies[pos] : 0.0f;
        }
        __syncthreads();
        
        // Calculate pattern match score for this window
        float local_score = 0.0f;
        for (int i = tid; i < num_codons; i += blockDim.x) {
            float diff = window_cache[i] - pattern_cache[i];
            local_score += expf(-diff * diff);
        }
        
        // Reduce to get window score
        score_cache[tid] = local_score;
        __syncthreads();
        
        // Parallel reduction
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                score_cache[tid] += score_cache[tid + s];
            }
            __syncthreads();
        }
        
        // Update best score
        if (tid == 0) {
            float window_score = score_cache[0] / num_codons;
            window_scores[pattern_idx * num_windows + window_idx] = window_score;
            
            if (window_score > best_score) {
                best_score = window_score;
                best_pos = window_start;
            }
        }
        __syncthreads();
    }
    
    // Store best position
    if (tid == 0) {
        best_positions[pattern_idx] = best_pos;
    }
}

extern "C" __global__ void find_best_matches_kernel(
    const float* match_scores,
    unsigned int* best_pattern_indices,
    float* best_scores,
    const unsigned int num_sequences,
    const unsigned int num_patterns,
    const float threshold
) {
    const int seq_idx = blockIdx.x;
    if (seq_idx >= num_sequences) return;
    
    // Each block finds the best match for one sequence
    __shared__ float max_score;
    __shared__ unsigned int max_index;
    
    if (threadIdx.x == 0) {
        max_score = -1.0f;
        max_index = 0;
    }
    __syncthreads();
    
    // Find maximum score for this sequence
    for (int pattern_idx = threadIdx.x; pattern_idx < num_patterns; pattern_idx += blockDim.x) {
        float score = match_scores[seq_idx * num_patterns + pattern_idx];
        
        // Atomic max operation
        if (score > threshold) {
            atomicMax((int*)&max_score, __float_as_int(score));
            if (__float_as_int(score) == *(int*)&max_score) {
                max_index = pattern_idx;
            }
        }
    }
    __syncthreads();
    
    // Write result
    if (threadIdx.x == 0) {
        best_pattern_indices[seq_idx] = max_index;
        best_scores[seq_idx] = max_score;
    }
}
"#;

pub struct PatternMatcher {
    device: Arc<CudaDevice>,
    match_kernel: String,
    best_kernel: String,
    sliding_window_kernel: String,
}

impl PatternMatcher {
    pub fn new(device: &super::super::device::CudaDevice) -> CudaResult<Self> {
        let match_kernel = "pattern_match_kernel".to_string();
        let best_kernel = "find_best_matches_kernel".to_string();
        let sliding_window_kernel = "sliding_window_pattern_kernel".to_string();
        
        // Compile kernels
        let ptx = cudarc::nvrtc::compile_ptx(PATTERN_MATCH_KERNEL)
            .map_err(|e| CudaError::kernel(format!("Failed to compile pattern matcher: {}", e)))?;
        
        // Load module
        device.inner()
            .load_ptx(ptx, "pattern_module", &[&match_kernel, &best_kernel, &sliding_window_kernel])
            .map_err(|e| CudaError::kernel(format!("Failed to load kernels: {}", e)))?;
        
        Ok(Self {
            device: device.inner().clone(),
            match_kernel,
            best_kernel,
            sliding_window_kernel,
        })
    }
    
    pub fn match_patterns(
        &self,
        frequency_table: &CudaFrequencyTable,
        trait_patterns: &[TraitPattern],
    ) -> CudaResult<Vec<PatternMatch>> {
        if trait_patterns.is_empty() {
            return Ok(vec![]);
        }
        
        // Prepare frequency data
        let (freq_array, num_sequences) = Self::prepare_frequencies(frequency_table);
        let (pattern_array, pattern_info) = Self::prepare_patterns(trait_patterns);
        
        let num_patterns = trait_patterns.len();
        let pattern_length = 64; // All 64 codons
        let num_codons = 64;
        
        // Allocate device memory
        let d_frequencies = CudaBuffer::from_slice(self.device.clone(), &freq_array)?;
        let d_patterns = CudaBuffer::from_slice(self.device.clone(), &pattern_array)?;
        let mut d_scores = CudaBuffer::<f32>::new(
            self.device.clone(),
            num_sequences * num_patterns
        )?;
        
        // Launch pattern matching kernel
        let block_size = 128;
        let grid_size = ((num_sequences * num_patterns + block_size - 1) / block_size) as u32;
        let shared_mem = (block_size * pattern_length * std::mem::size_of::<f32>()) as u32;
        
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: shared_mem,
        };
        
        let kernel = self.device
            .get_func("pattern_module", &self.match_kernel)
            .map_err(|e| CudaError::kernel(format!("Failed to get match kernel: {}", e)))?;
        
        unsafe {
            kernel.launch(
                config,
                (
                    d_frequencies.as_device_ptr(),
                    d_patterns.as_device_ptr(),
                    d_scores.as_device_ptr_mut(),
                    num_sequences as u32,
                    num_patterns as u32,
                    pattern_length as u32,
                    num_codons as u32,
                ),
            ).map_err(|e| CudaError::kernel(format!("Pattern matching failed: {}", e)))?;
        }
        
        // Find best matches
        let mut d_best_indices = CudaBuffer::<u32>::new(self.device.clone(), num_sequences)?;
        let mut d_best_scores = CudaBuffer::<f32>::new(self.device.clone(), num_sequences)?;
        
        let best_config = LaunchConfig {
            grid_dim: (num_sequences as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        
        let best_kernel = self.device
            .get_func("pattern_module", &self.best_kernel)
            .map_err(|e| CudaError::kernel(format!("Failed to get best kernel: {}", e)))?;
        
        unsafe {
            best_kernel.launch(
                best_config,
                (
                    d_scores.as_device_ptr(),
                    d_best_indices.as_device_ptr_mut(),
                    d_best_scores.as_device_ptr_mut(),
                    num_sequences as u32,
                    num_patterns as u32,
                    0.7f32, // Threshold
                ),
            ).map_err(|e| CudaError::kernel(format!("Best match finding failed: {}", e)))?;
        }
        
        // Synchronize and get results
        self.device.synchronize()
            .map_err(|e| CudaError::sync("Synchronization failed"))?;
        
        let best_indices = d_best_indices.to_vec()?;
        let best_scores = d_best_scores.to_vec()?;
        
        // Create pattern matches
        let mut matches = Vec::new();
        for (seq_idx, (pattern_idx, score)) in best_indices.iter()
            .zip(best_scores.iter())
            .enumerate()
        {
            if *score > 0.0 {
                matches.push(PatternMatch {
                    sequence_id: format!("seq_{}", seq_idx),
                    pattern_id: trait_patterns[*pattern_idx as usize].trait_name.clone(),
                    score: *score as f64,
                    position: 0, // Full sequence match
                    confidence: *score as f64,
                });
            }
        }
        
        Ok(matches)
    }
    
    fn prepare_frequencies(freq_table: &CudaFrequencyTable) -> (Vec<f32>, usize) {
        // For simplicity, use global frequencies repeated
        // In production, would use sequence-specific frequencies
        let mut freq_array = Vec::new();
        let num_sequences = 10; // Example: 10 sequences
        
        for _ in 0..num_sequences {
            for i in 0..64 {
                let codon = Self::index_to_codon(i);
                let freq = freq_table.global_frequencies
                    .get(&codon)
                    .copied()
                    .unwrap_or(0.0) as f32;
                freq_array.push(freq);
            }
        }
        
        (freq_array, num_sequences)
    }
    
    fn prepare_patterns(trait_patterns: &[TraitPattern]) -> (Vec<f32>, Vec<(String, usize)>) {
        let mut pattern_array = Vec::new();
        let mut pattern_info = Vec::new();
        
        for pattern in trait_patterns {
            pattern_info.push((pattern.trait_name.clone(), pattern.codon_preferences.len()));
            
            // Convert pattern to fixed-size array
            for i in 0..64 {
                let codon = Self::index_to_codon(i);
                let pref = pattern.codon_preferences
                    .get(&codon)
                    .copied()
                    .unwrap_or(0.0) as f32;
                pattern_array.push(pref);
            }
        }
        
        (pattern_array, pattern_info)
    }
    
    fn index_to_codon(index: usize) -> String {
        const BASES: [char; 4] = ['A', 'C', 'G', 'T'];
        let n1 = (index >> 4) & 0x3;
        let n2 = (index >> 2) & 0x3;
        let n3 = index & 0x3;
        
        format!("{}{}{}", BASES[n1], BASES[n2], BASES[n3])
    }
    
    /// Perform sliding window pattern matching for fine-grained trait detection
    pub fn match_patterns_sliding_window(
        &self,
        sequence_frequencies: &[f32], // Flattened sequence frequencies
        trait_patterns: &[TraitPattern],
        window_size: usize,
        window_stride: usize,
        seq_length: usize,
    ) -> CudaResult<Vec<(String, usize, f64)>> { // (trait_name, position, score)
        if trait_patterns.is_empty() || sequence_frequencies.is_empty() {
            return Ok(vec![]);
        }
        
        let num_patterns = trait_patterns.len();
        let num_codons = 64;
        let num_windows = if seq_length > window_size {
            (seq_length - window_size) / window_stride + 1
        } else {
            1
        };
        
        // Prepare pattern data
        let (pattern_array, _) = Self::prepare_patterns(trait_patterns);
        
        // Allocate device memory
        let d_seq_frequencies = CudaBuffer::from_slice(self.device.clone(), sequence_frequencies)?;
        let d_patterns = CudaBuffer::from_slice(self.device.clone(), &pattern_array)?;
        let mut d_window_scores = CudaBuffer::<f32>::new(
            self.device.clone(),
            num_patterns * num_windows
        )?;
        let mut d_best_positions = CudaBuffer::<u32>::new(
            self.device.clone(),
            num_patterns
        )?;
        
        // Launch sliding window kernel
        let block_size = 256;
        let shared_mem = (3 * num_codons * std::mem::size_of::<f32>()) as u32;
        
        let config = LaunchConfig {
            grid_dim: (num_patterns as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: shared_mem,
        };
        
        let kernel = self.device
            .get_func("pattern_module", &self.sliding_window_kernel)
            .map_err(|e| CudaError::kernel(format!("Failed to get sliding window kernel: {}", e)))?;
        
        unsafe {
            kernel.launch(
                config,
                (
                    d_seq_frequencies.as_device_ptr(),
                    d_patterns.as_device_ptr(),
                    d_window_scores.as_device_ptr_mut(),
                    d_best_positions.as_device_ptr_mut(),
                    seq_length as u32,
                    num_patterns as u32,
                    window_size as u32,
                    window_stride as u32,
                    num_codons as u32,
                ),
            ).map_err(|e| CudaError::kernel(format!("Sliding window kernel failed: {}", e)))?;
        }
        
        // Synchronize and get results
        self.device.synchronize()
            .map_err(|e| CudaError::sync("Synchronization failed"))?;
        
        let window_scores = d_window_scores.to_vec()?;
        let best_positions = d_best_positions.to_vec()?;
        
        // Extract best matches
        let mut results = Vec::new();
        for (pattern_idx, &best_pos) in best_positions.iter().enumerate() {
            // Find best score for this pattern
            let mut best_score = 0.0f32;
            for window_idx in 0..num_windows {
                let score = window_scores[pattern_idx * num_windows + window_idx];
                if score > best_score {
                    best_score = score;
                }
            }
            
            if best_score > 0.7 { // Threshold
                results.push((
                    trait_patterns[pattern_idx].trait_name.clone(),
                    best_pos as usize,
                    best_score as f64,
                ));
            }
        }
        
        Ok(results)
    }
    
    /// Enhanced pattern matching with NeuroDNA integration
    pub fn match_neurodna_patterns(
        &self,
        frequency_table: &CudaFrequencyTable,
        neurodna_traits: &HashMap<String, Vec<String>>, // trait_name -> preferred_codons
    ) -> CudaResult<Vec<PatternMatch>> {
        // Convert NeuroDNA trait patterns to TraitPattern format
        let mut trait_patterns = Vec::new();
        
        for (trait_name, preferred_codons) in neurodna_traits {
            let mut codon_preferences = HashMap::new();
            
            // Set high preference for NeuroDNA-identified codons
            for codon in preferred_codons {
                codon_preferences.insert(codon.clone(), 1.0);
            }
            
            // Set lower preference for other codons
            for i in 0..64 {
                let codon = Self::index_to_codon(i);
                if !codon_preferences.contains_key(&codon) {
                    codon_preferences.insert(codon, 0.1);
                }
            }
            
            trait_patterns.push(TraitPattern {
                trait_name: trait_name.clone(),
                preferred_codons: preferred_codons.clone(),
                avoided_codons: vec![],
                motifs: vec![],
                weight: 1.0,
                codon_preferences,
                regulatory_patterns: vec![],
            });
        }
        
        // Use enhanced pattern matching
        self.match_patterns(frequency_table, &trait_patterns)
    }
}