use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use crate::cuda::{CudaBuffer, CudaError, CudaResult};
use crate::types::{DnaSequence, CodonCounts};

// CUDA kernel source for codon counting - optimized for GTX 2070
const CODON_COUNT_KERNEL: &str = r#"
// Optimized codon counting kernel for GTX 2070 (compute capability 7.5)
// Uses warp-level primitives and coalesced memory access
extern "C" __global__ void codon_count_kernel(
    const unsigned char* sequences,
    const unsigned int* seq_offsets,
    const unsigned int* seq_lengths,
    unsigned int* codon_counts,
    const unsigned int num_sequences
) {
    // Thread configuration
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid & 31;
    const int threads_per_block = blockDim.x;
    
    // Each block processes one sequence
    if (bid >= num_sequences) return;
    
    // Shared memory for local codon counts
    // Use bank conflict-free access pattern
    __shared__ unsigned int local_counts[64];
    __shared__ unsigned int warp_counts[8][64]; // 8 warps max per block
    
    // Initialize shared memory - coalesced access
    if (tid < 64) {
        local_counts[tid] = 0;
    }
    if (tid < 512) { // 8 warps * 64 codons
        warp_counts[tid / 64][tid % 64] = 0;
    }
    __syncthreads();
    
    // Get sequence boundaries
    const unsigned int seq_start = seq_offsets[bid];
    const unsigned int seq_length = seq_lengths[bid];
    const unsigned int seq_end = seq_start + seq_length;
    
    // Process sequence with coalesced memory access
    // Each warp processes a contiguous chunk
    const unsigned int warp_chunk_size = 32 * 3; // Process 32 codons per warp iteration
    
    for (unsigned int base_pos = seq_start + warp_id * warp_chunk_size; 
         base_pos + 2 < seq_end; 
         base_pos += (threads_per_block / 32) * warp_chunk_size) {
        
        // Each thread in warp processes multiple codons
        for (int i = 0; i < 3 && base_pos + lane_id * 3 + i * 3 + 2 < seq_end; i++) {
            unsigned int pos = base_pos + lane_id * 3 + i * 3;
            
            if (pos + 2 < seq_end) {
                // Read three nucleotides - coalesced access
                unsigned char n1 = sequences[pos];
                unsigned char n2 = sequences[pos + 1];
                unsigned char n3 = sequences[pos + 2];
                
                // Convert to codon index (0-63)
                unsigned int codon_idx = (n1 << 4) | (n2 << 2) | n3;
                
                // Increment warp-local count to reduce contention
                atomicAdd(&warp_counts[warp_id][codon_idx], 1);
            }
        }
    }
    
    __syncthreads();
    
    // Reduce warp counts to block counts
    if (tid < 64) {
        unsigned int total = 0;
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            total += warp_counts[w][tid];
        }
        local_counts[tid] = total;
    }
    
    __syncthreads();
    
    // Write results to global memory - coalesced
    if (tid < 64) {
        codon_counts[bid * 64 + tid] = local_counts[tid];
    }
}

// Sliding window kernel for large sequences
extern "C" __global__ void sliding_window_codon_kernel(
    const unsigned char* sequences,
    const unsigned int* seq_offsets,
    const unsigned int* seq_lengths,
    unsigned int* window_codon_counts,
    const unsigned int window_size,
    const unsigned int window_stride,
    const unsigned int num_sequences
) {
    // Grid-stride loop for processing multiple windows
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    
    // Shared memory for window processing
    extern __shared__ unsigned int shared_counts[];
    
    // Process windows across all sequences
    for (int work_idx = global_tid; work_idx < num_sequences; work_idx += total_threads) {
        const unsigned int seq_start = seq_offsets[work_idx];
        const unsigned int seq_length = seq_lengths[work_idx];
        
        // Calculate number of windows for this sequence
        const unsigned int num_windows = (seq_length > window_size) ? 
            ((seq_length - window_size) / window_stride + 1) : 1;
        
        // Process each window
        for (unsigned int w = 0; w < num_windows; w++) {
            const unsigned int window_start = seq_start + w * window_stride;
            const unsigned int window_end = min(window_start + window_size, seq_start + seq_length);
            
            // Initialize local counts
            if (threadIdx.x < 64) {
                shared_counts[threadIdx.x] = 0;
            }
            __syncthreads();
            
            // Count codons in this window
            for (unsigned int pos = window_start + threadIdx.x * 3; 
                 pos + 2 < window_end; 
                 pos += blockDim.x * 3) {
                
                unsigned char n1 = sequences[pos];
                unsigned char n2 = sequences[pos + 1];
                unsigned char n3 = sequences[pos + 2];
                
                unsigned int codon_idx = (n1 << 4) | (n2 << 2) | n3;
                atomicAdd(&shared_counts[codon_idx], 1);
            }
            
            __syncthreads();
            
            // Write window results
            if (threadIdx.x < 64) {
                const unsigned int output_idx = (work_idx * num_windows + w) * 64 + threadIdx.x;
                window_codon_counts[output_idx] = shared_counts[threadIdx.x];
            }
        }
    }
}
"#;

/// CUDA kernel for parallel codon counting
pub struct CodonCounter {
    device: Arc<CudaDevice>,
    kernel_name: String,
    sliding_window_kernel: String,
}

impl CodonCounter {
    pub fn new(device: &super::super::device::CudaDevice) -> CudaResult<Self> {
        let kernel_name = "codon_count_kernel".to_string();
        let sliding_window_kernel = "sliding_window_codon_kernel".to_string();
        
        // Compile kernel
        let ptx = cudarc::nvrtc::compile_ptx(CODON_COUNT_KERNEL)
            .map_err(|e| CudaError::kernel(format!("Failed to compile codon counter: {}", e)))?;
        
        // Load module with both kernels
        device.inner()
            .load_ptx(ptx, "codon_module", &[&kernel_name, &sliding_window_kernel])
            .map_err(|e| CudaError::kernel(format!("Failed to load kernel: {}", e)))?;
        
        Ok(Self {
            device: device.inner().clone(),
            kernel_name,
            sliding_window_kernel,
        })
    }
    
    pub fn count(&self, sequences: &[DnaSequence]) -> CudaResult<Vec<CodonCounts>> {
        let num_sequences = sequences.len();
        if num_sequences == 0 {
            return Ok(vec![]);
        }
        
        // Pack sequences into continuous buffer
        let (packed_sequences, offsets, lengths) = Self::pack_sequences(sequences);
        
        // Allocate device memory
        let d_sequences = CudaBuffer::from_slice(self.device.clone(), &packed_sequences)?;
        let d_offsets = CudaBuffer::from_slice(self.device.clone(), &offsets)?;
        let d_lengths = CudaBuffer::from_slice(self.device.clone(), &lengths)?;
        let mut d_counts = CudaBuffer::<u32>::new(self.device.clone(), num_sequences * 64)?;
        
        // Launch kernel
        let grid_size = num_sequences as u32;
        let block_size = 256; // Optimal for GTX 2070
        
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 64 * std::mem::size_of::<u32>() as u32,
        };
        
        let kernel = self.device
            .get_func("codon_module", &self.kernel_name)
            .map_err(|e| CudaError::kernel(format!("Failed to get kernel: {}", e)))?;
        
        unsafe {
            kernel.launch(
                config,
                (
                    d_sequences.as_device_ptr(),
                    d_offsets.as_device_ptr(),
                    d_lengths.as_device_ptr(),
                    d_counts.as_device_ptr_mut(),
                    num_sequences as u32,
                ),
            ).map_err(|e| CudaError::kernel(format!("Kernel launch failed: {}", e)))?;
        }
        
        // Synchronize and copy results
        self.device.synchronize()
            .map_err(|e| CudaError::sync(format!("Synchronization failed: {}", e)))?;
        
        let counts_vec = d_counts.to_vec()?;
        
        // Convert to CodonCounts
        let mut results = Vec::with_capacity(num_sequences);
        for i in 0..num_sequences {
            let mut codon_counts = CodonCounts::new();
            let base_idx = i * 64;
            
            // Unpack counts
            for j in 0..64 {
                let count = counts_vec[base_idx + j];
                // Convert index back to codon string
                let codon = Self::index_to_codon(j);
                codon_counts.insert(codon, count as usize);
            }
            
            results.push(codon_counts);
        }
        
        Ok(results)
    }
    
    /// Pack sequences into continuous buffer with 2-bit encoding
    fn pack_sequences(sequences: &[DnaSequence]) -> (Vec<u8>, Vec<u32>, Vec<u32>) {
        let mut packed = Vec::new();
        let mut offsets = Vec::new();
        let mut lengths = Vec::new();
        
        for seq in sequences {
            offsets.push(packed.len() as u32);
            lengths.push(seq.sequence.len() as u32);
            
            // Encode nucleotides: A=0, C=1, G=2, T=3
            for nucleotide in seq.sequence.bytes() {
                let encoded = match nucleotide {
                    b'A' | b'a' => 0,
                    b'C' | b'c' => 1,
                    b'G' | b'g' => 2,
                    b'T' | b't' => 3,
                    _ => 0, // Default to A for unknown
                };
                packed.push(encoded);
            }
        }
        
        (packed, offsets, lengths)
    }
    
    /// Convert codon index (0-63) to string
    fn index_to_codon(index: usize) -> String {
        const BASES: [char; 4] = ['A', 'C', 'G', 'T'];
        let n1 = (index >> 4) & 0x3;
        let n2 = (index >> 2) & 0x3;
        let n3 = index & 0x3;
        
        format!("{}{}{}", BASES[n1], BASES[n2], BASES[n3])
    }
    
    /// Count codons using sliding windows
    pub fn count_sliding_windows(
        &self,
        sequences: &[DnaSequence],
        window_size: usize,
        window_stride: usize,
    ) -> CudaResult<Vec<Vec<CodonCounts>>> {
        let num_sequences = sequences.len();
        if num_sequences == 0 {
            return Ok(vec![]);
        }
        
        // Pack sequences
        let (packed_sequences, offsets, lengths) = Self::pack_sequences(sequences);
        
        // Calculate total number of windows
        let mut total_windows = 0;
        let mut window_counts = Vec::with_capacity(num_sequences);
        for len in &lengths {
            let num_windows = if *len as usize > window_size {
                ((*len as usize - window_size) / window_stride + 1)
            } else {
                1
            };
            window_counts.push(num_windows);
            total_windows += num_windows;
        }
        
        // Allocate device memory
        let d_sequences = CudaBuffer::from_slice(self.device.clone(), &packed_sequences)?;
        let d_offsets = CudaBuffer::from_slice(self.device.clone(), &offsets)?;
        let d_lengths = CudaBuffer::from_slice(self.device.clone(), &lengths)?;
        let mut d_window_counts = CudaBuffer::<u32>::new(
            self.device.clone(), 
            total_windows * 64
        )?;
        
        // Launch sliding window kernel
        let block_size = 256;
        let grid_size = ((total_windows + block_size - 1) / block_size) as u32;
        
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 64 * std::mem::size_of::<u32>() as u32,
        };
        
        let kernel = self.device
            .get_func("codon_module", &self.sliding_window_kernel)
            .map_err(|e| CudaError::kernel(format!("Failed to get sliding window kernel: {}", e)))?;
        
        unsafe {
            kernel.launch(
                config,
                (
                    d_sequences.as_device_ptr(),
                    d_offsets.as_device_ptr(),
                    d_lengths.as_device_ptr(),
                    d_window_counts.as_device_ptr_mut(),
                    window_size as u32,
                    window_stride as u32,
                    num_sequences as u32,
                ),
            ).map_err(|e| CudaError::kernel(format!("Sliding window kernel launch failed: {}", e)))?;
        }
        
        // Synchronize and copy results
        self.device.synchronize()
            .map_err(|e| CudaError::sync(format!("Synchronization failed: {}", e)))?;
        
        let counts_vec = d_window_counts.to_vec()?;
        
        // Convert to nested CodonCounts
        let mut results = Vec::with_capacity(num_sequences);
        let mut offset = 0;
        
        for seq_idx in 0..num_sequences {
            let num_windows = window_counts[seq_idx];
            let mut seq_windows = Vec::with_capacity(num_windows);
            
            for window_idx in 0..num_windows {
                let mut codon_counts = CodonCounts::new();
                let base_idx = (offset + window_idx) * 64;
                
                for j in 0..64 {
                    let count = counts_vec[base_idx + j];
                    let codon = Self::index_to_codon(j);
                    if count > 0 {
                        codon_counts.insert(codon, count as usize);
                    }
                }
                
                seq_windows.push(codon_counts);
            }
            
            results.push(seq_windows);
            offset += num_windows;
        }
        
        Ok(results)
    }
}