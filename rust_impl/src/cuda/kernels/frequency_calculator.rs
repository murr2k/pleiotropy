use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use std::sync::Arc;
use std::collections::HashMap;

use crate::cuda::{CudaBuffer, CudaError, CudaResult};
use crate::types::{CodonCounts, TraitInfo, CudaFrequencyTable};

// CUDA kernel for frequency calculation - optimized for GTX 2070
const FREQUENCY_CALC_KERNEL: &str = r#"
// Optimized frequency calculation using warp-level operations
extern "C" __global__ void frequency_calc_kernel(
    const unsigned int* codon_counts,
    float* frequency_tables,
    const unsigned int* normalization_factors,
    const unsigned int num_sequences,
    const unsigned int num_traits,
    const unsigned int num_codons
) {
    // Thread configuration for coalesced access
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid & 31;
    
    // Each block processes one or more sequences
    const int seqs_per_block = blockDim.x / 64; // Each sequence needs 64 threads
    const int seq_in_block = tid / 64;
    const int codon_in_seq = tid % 64;
    
    // Grid stride loop for processing multiple sequences
    for (int seq_idx = bid * seqs_per_block + seq_in_block; 
         seq_idx < num_sequences; 
         seq_idx += gridDim.x * seqs_per_block) {
        
        if (seq_idx >= num_sequences) return;
        
        // Shared memory for normalization factors
        __shared__ float norm_factors[4]; // Up to 4 sequences per block
        
        // Load normalization factor
        if (tid < seqs_per_block && seq_idx < num_sequences) {
            norm_factors[tid] = (float)normalization_factors[seq_idx];
        }
        __syncthreads();
        
        // Process codons with coalesced memory access
        if (codon_in_seq < num_codons && seq_in_block < seqs_per_block) {
            const int idx = seq_idx * num_codons + codon_in_seq;
            unsigned int count = codon_counts[idx];
            float norm = norm_factors[seq_in_block];
            
            // Calculate frequency with fast division
            float frequency = (norm > 0.0f) ? __fdividef((float)count, norm) : 0.0f;
            
            // Store with coalesced access
            frequency_tables[idx] = frequency;
        }
    }
}

// Optimized trait bias calculation with warp shuffle operations
extern "C" __global__ void trait_bias_kernel(
    const float* base_frequencies,
    float* trait_frequencies,
    const unsigned char* trait_mask,
    const unsigned int num_sequences,
    const unsigned int num_codons,
    const unsigned int trait_id
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid & 31;
    
    // Shared memory for block-wide reduction
    __shared__ float block_sums[64];
    __shared__ unsigned int block_counts[64];
    
    // Initialize shared memory
    if (tid < 64) {
        block_sums[tid] = 0.0f;
        block_counts[tid] = 0;
    }
    __syncthreads();
    
    // Each warp processes different sequences
    const int seqs_per_warp = 32;
    const int start_seq = bid * blockDim.x + warp_id * seqs_per_warp;
    
    // Warp-level accumulation
    float warp_sums[2] = {0.0f, 0.0f}; // Process 2 codons per thread
    unsigned int warp_counts[2] = {0, 0};
    
    for (int seq_offset = 0; seq_offset < seqs_per_warp; seq_offset++) {
        int seq_idx = start_seq + seq_offset;
        if (seq_idx >= num_sequences) break;
        
        // Check trait mask
        unsigned char has_trait = trait_mask[seq_idx] & (1 << trait_id);
        if (!has_trait) continue;
        
        // Each thread in warp handles 2 codons
        for (int i = 0; i < 2; i++) {
            int codon_idx = lane_id * 2 + i;
            if (codon_idx < num_codons) {
                float freq = base_frequencies[seq_idx * num_codons + codon_idx];
                warp_sums[i] += freq;
                warp_counts[i]++;
            }
        }
    }
    
    // Warp-level reduction using shuffle operations
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        warp_sums[0] += __shfl_down_sync(0xFFFFFFFF, warp_sums[0], offset);
        warp_sums[1] += __shfl_down_sync(0xFFFFFFFF, warp_sums[1], offset);
        warp_counts[0] += __shfl_down_sync(0xFFFFFFFF, warp_counts[0], offset);
        warp_counts[1] += __shfl_down_sync(0xFFFFFFFF, warp_counts[1], offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        for (int i = 0; i < 2; i++) {
            int codon_idx = warp_id * 64 + i;
            if (codon_idx < 64) {
                atomicAdd(&block_sums[codon_idx], warp_sums[i]);
                atomicAdd(&block_counts[codon_idx], warp_counts[i]);
            }
        }
    }
    
    __syncthreads();
    
    // Final calculation - one thread per codon
    if (tid < num_codons && block_counts[tid] > 0) {
        float avg_freq = __fdividef(block_sums[tid], (float)block_counts[tid]);
        atomicAdd(&trait_frequencies[trait_id * num_codons + tid], avg_freq);
    }
}

// Batch frequency calculation kernel for multiple traits
extern "C" __global__ void batch_trait_frequency_kernel(
    const float* base_frequencies,
    float* trait_frequencies,
    const unsigned int* trait_assignments, // Which sequences have which traits
    const unsigned int num_sequences,
    const unsigned int num_codons,
    const unsigned int num_traits
) {
    // Cooperative groups for efficient reduction
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int trait_id = bid % num_traits;
    const int codon_group = bid / num_traits;
    
    // Process 16 codons per block
    const int codons_per_block = 16;
    const int start_codon = codon_group * codons_per_block;
    
    // Shared memory for accumulation
    __shared__ float codon_sums[16];
    __shared__ unsigned int codon_counts[16];
    
    if (tid < codons_per_block) {
        codon_sums[tid] = 0.0f;
        codon_counts[tid] = 0;
    }
    __syncthreads();
    
    // Grid-stride loop through sequences
    for (int seq_idx = tid; seq_idx < num_sequences; seq_idx += blockDim.x) {
        // Check if sequence has this trait
        unsigned int traits = trait_assignments[seq_idx];
        if (!(traits & (1 << trait_id))) continue;
        
        // Accumulate frequencies for this block's codons
        for (int i = 0; i < codons_per_block && start_codon + i < num_codons; i++) {
            int codon_idx = start_codon + i;
            float freq = base_frequencies[seq_idx * num_codons + codon_idx];
            atomicAdd(&codon_sums[i], freq);
            atomicAdd(&codon_counts[i], 1);
        }
    }
    
    __syncthreads();
    
    // Write results
    if (tid < codons_per_block && start_codon + tid < num_codons) {
        if (codon_counts[tid] > 0) {
            float avg = __fdividef(codon_sums[tid], (float)codon_counts[tid]);
            trait_frequencies[trait_id * num_codons + start_codon + tid] = avg;
        }
    }
}
"#;

pub struct FrequencyCalculator {
    device: Arc<CudaDevice>,
    calc_kernel: String,
    bias_kernel: String,
    batch_kernel: String,
}

impl FrequencyCalculator {
    pub fn new(device: &super::super::device::CudaDevice) -> CudaResult<Self> {
        let calc_kernel = "frequency_calc_kernel".to_string();
        let bias_kernel = "trait_bias_kernel".to_string();
        let batch_kernel = "batch_trait_frequency_kernel".to_string();
        
        // Compile kernels
        let ptx = cudarc::nvrtc::compile_ptx(FREQUENCY_CALC_KERNEL)
            .map_err(|e| CudaError::kernel(format!("Failed to compile frequency calculator: {}", e)))?;
        
        // Load module with all kernels
        device.inner()
            .load_ptx(ptx, "frequency_module", &[&calc_kernel, &bias_kernel, &batch_kernel])
            .map_err(|e| CudaError::kernel(format!("Failed to load kernels: {}", e)))?;
        
        Ok(Self {
            device: device.inner().clone(),
            calc_kernel,
            bias_kernel,
            batch_kernel,
        })
    }
    
    pub fn calculate(
        &self,
        codon_counts: &[CodonCounts],
        traits: &[TraitInfo],
    ) -> CudaResult<CudaFrequencyTable> {
        let num_sequences = codon_counts.len();
        let num_codons = 64;
        let num_traits = traits.len();
        
        // Flatten codon counts to array
        let (counts_array, norm_factors) = Self::flatten_counts(codon_counts);
        
        // Allocate device memory
        let d_counts = CudaBuffer::from_slice(self.device.clone(), &counts_array)?;
        let d_norm_factors = CudaBuffer::from_slice(self.device.clone(), &norm_factors)?;
        let mut d_frequencies = CudaBuffer::<f32>::new(
            self.device.clone(),
            num_sequences * num_codons
        )?;
        
        // Launch frequency calculation kernel
        // Optimized for GTX 2070: process multiple sequences per block
        let block_size = 256;
        let seqs_per_block = block_size / 64; // 4 sequences per block
        let grid_size = ((num_sequences + seqs_per_block - 1) / seqs_per_block) as u32;
        
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: seqs_per_block as u32 * std::mem::size_of::<f32>() as u32,
        };
        
        let kernel = self.device
            .get_func("frequency_module", &self.calc_kernel)
            .map_err(|e| CudaError::kernel(format!("Failed to get calc kernel: {}", e)))?;
        
        unsafe {
            kernel.launch(
                config,
                (
                    d_counts.as_device_ptr(),
                    d_frequencies.as_device_ptr_mut(),
                    d_norm_factors.as_device_ptr(),
                    num_sequences as u32,
                    num_traits as u32,
                    num_codons as u32,
                ),
            ).map_err(|e| CudaError::kernel(format!("Frequency calc failed: {}", e)))?;
        }
        
        // Calculate trait-specific biases
        let trait_mask = Self::create_trait_mask(codon_counts, traits);
        let d_trait_mask = CudaBuffer::from_slice(self.device.clone(), &trait_mask)?;
        let mut d_trait_frequencies = CudaBuffer::<f32>::new(
            self.device.clone(),
            num_traits * num_codons
        )?;
        
        // Launch trait bias kernel for each trait
        let bias_kernel = self.device
            .get_func("frequency_module", &self.bias_kernel)
            .map_err(|e| CudaError::kernel(format!("Failed to get bias kernel: {}", e)))?;
        
        for trait_id in 0..num_traits {
            let config = LaunchConfig {
                grid_dim: ((num_sequences + 255) / 256) as u32,
                block_dim: (256, 1, 1),
                shared_mem_bytes: (64 * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>())) as u32,
            };
            
            unsafe {
                bias_kernel.launch(
                    config,
                    (
                        d_frequencies.as_device_ptr(),
                        d_trait_frequencies.as_device_ptr_mut(),
                        d_trait_mask.as_device_ptr(),
                        num_sequences as u32,
                        num_codons as u32,
                        trait_id as u32,
                    ),
                ).map_err(|e| CudaError::kernel(format!("Trait bias calc failed: {}", e)))?;
            }
        }
        
        // Synchronize and copy results
        self.device.synchronize()
            .map_err(|e| CudaError::sync("Synchronization failed"))?;
        
        let base_frequencies = d_frequencies.to_vec()?;
        let trait_frequencies = d_trait_frequencies.to_vec()?;
        
        // Build frequency table
        let mut freq_table = CudaFrequencyTable {
            global_frequencies: HashMap::new(),
            trait_frequencies: HashMap::new(),
        };
        
        // Calculate global frequencies (average across all sequences)
        for codon_idx in 0..num_codons {
            let mut sum = 0.0;
            for seq_idx in 0..num_sequences {
                sum += base_frequencies[seq_idx * num_codons + codon_idx];
            }
            let codon = Self::index_to_codon(codon_idx);
            freq_table.global_frequencies.insert(codon, sum / num_sequences as f64);
        }
        
        // Extract trait-specific frequencies
        for (trait_idx, trait_info) in traits.iter().enumerate() {
            let mut trait_freq = HashMap::new();
            for codon_idx in 0..num_codons {
                let freq = trait_frequencies[trait_idx * num_codons + codon_idx];
                let codon = Self::index_to_codon(codon_idx);
                trait_freq.insert(codon, freq as f64);
            }
            freq_table.trait_frequencies.insert(trait_info.name.clone(), trait_freq);
        }
        
        Ok(freq_table)
    }
    
    fn flatten_counts(codon_counts: &[CodonCounts]) -> (Vec<u32>, Vec<u32>) {
        let mut counts_array = Vec::new();
        let mut norm_factors = Vec::new();
        
        for counts in codon_counts {
            let mut total = 0u32;
            
            // Extract counts in consistent order
            for i in 0..64 {
                let codon = Self::index_to_codon(i);
                let count = counts.get(&codon).copied().unwrap_or(0) as u32;
                counts_array.push(count);
                total += count;
            }
            
            norm_factors.push(total);
        }
        
        (counts_array, norm_factors)
    }
    
    fn create_trait_mask(codon_counts: &[CodonCounts], traits: &[TraitInfo]) -> Vec<u8> {
        // Simple mask: assume each sequence has associated traits
        // In production, this would come from metadata
        vec![0xFF; codon_counts.len()] // All traits for now
    }
    
    fn index_to_codon(index: usize) -> String {
        const BASES: [char; 4] = ['A', 'C', 'G', 'T'];
        let n1 = (index >> 4) & 0x3;
        let n2 = (index >> 2) & 0x3;
        let n3 = index & 0x3;
        
        format!("{}{}{}", BASES[n1], BASES[n2], BASES[n3])
    }
    
    /// Calculate frequencies using batch processing for better performance
    pub fn calculate_batch(
        &self,
        codon_counts: &[CodonCounts],
        trait_assignments: &[u32], // Bit mask of traits per sequence
        num_traits: usize,
    ) -> CudaResult<CudaFrequencyTable> {
        let num_sequences = codon_counts.len();
        let num_codons = 64;
        
        // Flatten codon counts
        let (counts_array, norm_factors) = Self::flatten_counts(codon_counts);
        
        // Allocate device memory
        let d_counts = CudaBuffer::from_slice(self.device.clone(), &counts_array)?;
        let d_norm_factors = CudaBuffer::from_slice(self.device.clone(), &norm_factors)?;
        let d_trait_assignments = CudaBuffer::from_slice(self.device.clone(), trait_assignments)?;
        let mut d_frequencies = CudaBuffer::<f32>::new(
            self.device.clone(),
            num_sequences * num_codons
        )?;
        let mut d_trait_frequencies = CudaBuffer::<f32>::new(
            self.device.clone(),
            num_traits * num_codons
        )?;
        
        // First calculate base frequencies
        let block_size = 256;
        let seqs_per_block = block_size / 64;
        let grid_size = ((num_sequences + seqs_per_block - 1) / seqs_per_block) as u32;
        
        let calc_config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: seqs_per_block as u32 * std::mem::size_of::<f32>() as u32,
        };
        
        let calc_kernel = self.device
            .get_func("frequency_module", &self.calc_kernel)
            .map_err(|e| CudaError::kernel(format!("Failed to get calc kernel: {}", e)))?;
        
        unsafe {
            calc_kernel.launch(
                calc_config,
                (
                    d_counts.as_device_ptr(),
                    d_frequencies.as_device_ptr_mut(),
                    d_norm_factors.as_device_ptr(),
                    num_sequences as u32,
                    num_traits as u32,
                    num_codons as u32,
                ),
            ).map_err(|e| CudaError::kernel(format!("Base frequency calc failed: {}", e)))?;
        }
        
        // Launch batch trait frequency kernel
        let codons_per_block = 16;
        let codon_blocks = (num_codons + codons_per_block - 1) / codons_per_block;
        let total_blocks = codon_blocks * num_traits;
        
        let batch_config = LaunchConfig {
            grid_dim: (total_blocks as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: codons_per_block as u32 * 
                (std::mem::size_of::<f32>() + std::mem::size_of::<u32>()) as u32,
        };
        
        let batch_kernel = self.device
            .get_func("frequency_module", &self.batch_kernel)
            .map_err(|e| CudaError::kernel(format!("Failed to get batch kernel: {}", e)))?;
        
        unsafe {
            batch_kernel.launch(
                batch_config,
                (
                    d_frequencies.as_device_ptr(),
                    d_trait_frequencies.as_device_ptr_mut(),
                    d_trait_assignments.as_device_ptr(),
                    num_sequences as u32,
                    num_codons as u32,
                    num_traits as u32,
                ),
            ).map_err(|e| CudaError::kernel(format!("Batch trait calc failed: {}", e)))?;
        }
        
        // Synchronize and copy results
        self.device.synchronize()
            .map_err(|e| CudaError::sync("Synchronization failed"))?;
        
        let base_frequencies = d_frequencies.to_vec()?;
        let trait_frequencies = d_trait_frequencies.to_vec()?;
        
        // Build frequency table
        let mut freq_table = CudaFrequencyTable {
            global_frequencies: HashMap::new(),
            trait_frequencies: HashMap::new(),
        };
        
        // Calculate global frequencies
        for codon_idx in 0..num_codons {
            let mut sum = 0.0;
            for seq_idx in 0..num_sequences {
                sum += base_frequencies[seq_idx * num_codons + codon_idx];
            }
            let codon = Self::index_to_codon(codon_idx);
            freq_table.global_frequencies.insert(codon, sum / num_sequences as f64);
        }
        
        // Extract trait-specific frequencies
        for trait_idx in 0..num_traits {
            let mut trait_freq = HashMap::new();
            for codon_idx in 0..num_codons {
                let freq = trait_frequencies[trait_idx * num_codons + codon_idx];
                let codon = Self::index_to_codon(codon_idx);
                trait_freq.insert(codon, freq as f64);
            }
            freq_table.trait_frequencies.insert(format!("trait_{}", trait_idx), trait_freq);
        }
        
        Ok(freq_table)
    }
}