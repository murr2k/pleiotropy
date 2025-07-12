use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use crate::cuda::{CudaBuffer, CudaError, CudaResult};

// CUDA kernel for matrix operations and eigenanalysis - optimized for GTX 2070
const MATRIX_OPS_KERNEL: &str = r#"
// Power iteration method for dominant eigenvalues - faster for large matrices
extern "C" __global__ void power_iteration_kernel(
    const float* matrix,
    float* eigenvector,
    float* eigenvalue,
    float* temp_vector,
    const unsigned int size,
    const unsigned int max_iterations
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid & 31;
    
    // Shared memory for vector operations
    extern __shared__ float shared_mem[];
    float* shared_vec = shared_mem;
    float* shared_norm = &shared_mem[size];
    
    // Initialize random vector
    if (tid < size) {
        eigenvector[tid] = 1.0f / sqrtf((float)size);
    }
    __syncthreads();
    
    for (int iter = 0; iter < max_iterations; iter++) {
        // Matrix-vector multiplication with warp-level optimization
        float local_sum = 0.0f;
        
        // Each thread processes multiple rows
        for (int row = tid; row < size; row += blockDim.x) {
            float row_sum = 0.0f;
            
            // Load vector into shared memory in chunks
            for (int chunk = 0; chunk < size; chunk += blockDim.x) {
                int vec_idx = chunk + tid;
                if (vec_idx < size) {
                    shared_vec[vec_idx] = eigenvector[vec_idx];
                }
                __syncthreads();
                
                // Compute partial dot product
                int chunk_end = min(chunk + blockDim.x, size);
                for (int col = chunk; col < chunk_end; col++) {
                    row_sum += matrix[row * size + col] * shared_vec[col - chunk];
                }
                __syncthreads();
            }
            
            temp_vector[row] = row_sum;
        }
        __syncthreads();
        
        // Normalize vector using parallel reduction
        float norm_sum = 0.0f;
        for (int i = tid; i < size; i += blockDim.x) {
            float val = temp_vector[i];
            norm_sum += val * val;
        }
        
        // Warp-level reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            norm_sum += __shfl_down_sync(0xFFFFFFFF, norm_sum, offset);
        }
        
        // Write warp sums to shared memory
        if (lane_id == 0) {
            shared_norm[warp_id] = norm_sum;
        }
        __syncthreads();
        
        // Final reduction
        if (tid == 0) {
            float total_norm = 0.0f;
            for (int i = 0; i < (blockDim.x + 31) / 32; i++) {
                total_norm += shared_norm[i];
            }
            shared_norm[0] = sqrtf(total_norm);
        }
        __syncthreads();
        
        float norm = shared_norm[0];
        
        // Normalize and copy back
        for (int i = tid; i < size; i += blockDim.x) {
            eigenvector[i] = temp_vector[i] / norm;
        }
        __syncthreads();
    }
    
    // Compute eigenvalue (Rayleigh quotient)
    if (tid == 0) {
        float dot_product = 0.0f;
        for (int i = 0; i < size; i++) {
            float Av_i = 0.0f;
            for (int j = 0; j < size; j++) {
                Av_i += matrix[i * size + j] * eigenvector[j];
            }
            dot_product += eigenvector[i] * Av_i;
        }
        *eigenvalue = dot_product;
    }
}

// Enhanced Jacobi method with parallel sweep optimization
extern "C" __global__ void parallel_jacobi_kernel(
    float* matrix,           // Input/output matrix (will contain eigenvectors)
    float* eigenvalues,      // Output eigenvalues
    float* eigenvectors,     // Output eigenvectors
    const unsigned int size,
    const unsigned int max_iterations,
    const float tolerance
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_pairs = size / 2;
    
    // Shared memory for rotation parameters
    extern __shared__ float shared_data[];
    float* cos_values = shared_data;
    float* sin_values = &shared_data[num_pairs];
    
    // Initialize eigenvector matrix to identity
    if (bid == 0) {
        for (int i = tid; i < size * size; i += blockDim.x) {
            int row = i / size;
            int col = i % size;
            eigenvectors[i] = (row == col) ? 1.0f : 0.0f;
        }
    }
    __syncthreads();
    
    // Parallel Jacobi sweeps
    for (int iter = 0; iter < max_iterations; iter++) {
        __shared__ float off_diagonal_norm;
        
        // Even-odd ordering for parallel updates
        for (int parity = 0; parity < 2; parity++) {
            // Each thread handles one rotation pair
            if (tid < num_pairs) {
                int i = 2 * tid + parity;
                int j = (i + 1) % size;
                
                if (i < size - 1) {
                    float a_ii = matrix[i * size + i];
                    float a_jj = matrix[j * size + j];
                    float a_ij = matrix[i * size + j];
                    
                    // Calculate rotation angle
                    float theta = 0.5f * atan2f(2.0f * a_ij, a_jj - a_ii);
                    cos_values[tid] = cosf(theta);
                    sin_values[tid] = sinf(theta);
                }
            }
            __syncthreads();
            
            // Apply rotations in parallel
            for (int pair_idx = 0; pair_idx < num_pairs; pair_idx++) {
                int i = 2 * pair_idx + parity;
                int j = (i + 1) % size;
                
                if (i < size - 1) {
                    float c = cos_values[pair_idx];
                    float s = sin_values[pair_idx];
                    
                    // Update matrix columns in parallel
                    for (int k = tid; k < size; k += blockDim.x) {
                        float m_ki = matrix[k * size + i];
                        float m_kj = matrix[k * size + j];
                        
                        matrix[k * size + i] = c * m_ki - s * m_kj;
                        matrix[k * size + j] = s * m_ki + c * m_kj;
                    }
                    __syncthreads();
                    
                    // Update matrix rows
                    for (int k = tid; k < size; k += blockDim.x) {
                        float m_ik = matrix[i * size + k];
                        float m_jk = matrix[j * size + k];
                        
                        matrix[i * size + k] = c * m_ik - s * m_jk;
                        matrix[j * size + k] = s * m_ik + c * m_jk;
                    }
                    
                    // Update eigenvectors
                    for (int k = tid; k < size; k += blockDim.x) {
                        float v_ki = eigenvectors[k * size + i];
                        float v_kj = eigenvectors[k * size + j];
                        
                        eigenvectors[k * size + i] = c * v_ki - s * v_kj;
                        eigenvectors[k * size + j] = s * v_ki + c * v_kj;
                    }
                    __syncthreads();
                }
            }
        }
        
        // Check convergence
        if (tid == 0) {
            off_diagonal_norm = 0.0f;
            for (int i = 0; i < size; i++) {
                for (int j = i + 1; j < size; j++) {
                    float val = matrix[i * size + j];
                    off_diagonal_norm += val * val;
                }
            }
            off_diagonal_norm = sqrtf(off_diagonal_norm);
        }
        __syncthreads();
        
        if (off_diagonal_norm < tolerance) break;
    }
    
    // Extract eigenvalues from diagonal
    for (int i = tid; i < size; i += blockDim.x) {
        eigenvalues[i] = matrix[i * size + i];
    }
}

// Matrix multiplication kernel for correlation matrix
extern "C" __global__ void correlation_matrix_kernel(
    const float* data_matrix,    // Input data (sequences x features)
    float* correlation_matrix,   // Output correlation matrix
    const unsigned int rows,     // Number of sequences
    const unsigned int cols,     // Number of features
    const float* means,          // Pre-computed means
    const float* std_devs        // Pre-computed standard deviations
) {
    const int i = blockIdx.x;
    const int j = blockIdx.y;
    const int tid = threadIdx.x;
    
    if (i >= cols || j >= cols || j < i) return; // Only compute upper triangle
    
    __shared__ float sum;
    if (tid == 0) sum = 0.0f;
    __syncthreads();
    
    // Compute correlation between features i and j
    for (int k = tid; k < rows; k += blockDim.x) {
        float xi = (data_matrix[k * cols + i] - means[i]) / std_devs[i];
        float xj = (data_matrix[k * cols + j] - means[j]) / std_devs[j];
        atomicAdd(&sum, xi * xj);
    }
    __syncthreads();
    
    if (tid == 0) {
        float corr = sum / (float)(rows - 1);
        correlation_matrix[i * cols + j] = corr;
        correlation_matrix[j * cols + i] = corr; // Symmetric
    }
}

// PCA-based trait separation kernel
extern "C" __global__ void pca_trait_separation_kernel(
    const float* codon_frequencies,  // Input: sequences x codons
    const float* eigenvectors,       // Pre-computed eigenvectors from correlation
    float* principal_components,     // Output: sequences x components
    float* trait_loadings,          // Output: traits x components
    const unsigned int num_sequences,
    const unsigned int num_codons,
    const unsigned int num_components
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int seq_idx = bid;
    
    if (seq_idx >= num_sequences) return;
    
    // Shared memory for eigenvector tiles
    extern __shared__ float shared_eigen[];
    
    // Transform sequences to principal component space
    for (int comp = 0; comp < num_components; comp++) {
        // Load eigenvector for this component
        for (int i = tid; i < num_codons; i += blockDim.x) {
            shared_eigen[i] = eigenvectors[comp * num_codons + i];
        }
        __syncthreads();
        
        // Compute projection
        float projection = 0.0f;
        for (int i = tid; i < num_codons; i += blockDim.x) {
            projection += codon_frequencies[seq_idx * num_codons + i] * shared_eigen[i];
        }
        
        // Reduce within block
        __shared__ float block_sum[256];
        block_sum[tid] = projection;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                block_sum[tid] += block_sum[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            principal_components[seq_idx * num_components + comp] = block_sum[0];
        }
        __syncthreads();
    }
}

// Singular Value Decomposition kernel for trait factorization
extern "C" __global__ void svd_trait_kernel(
    const float* trait_codon_matrix,  // traits x codons preference matrix
    float* U,                         // Left singular vectors (traits x traits)
    float* S,                         // Singular values
    float* V,                         // Right singular vectors (codons x codons)
    const unsigned int num_traits,
    const unsigned int num_codons,
    const unsigned int max_iterations
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    // Simplified SVD using power iteration for top k singular values
    const int k = min(num_traits, num_codons);
    
    // Each block computes one singular value/vector pair
    if (bid < k) {
        extern __shared__ float shared_workspace[];
        float* u_vec = shared_workspace;
        float* v_vec = &shared_workspace[num_traits];
        float* temp = &shared_workspace[num_traits + num_codons];
        
        // Initialize with random vector
        for (int i = tid; i < num_traits; i += blockDim.x) {
            u_vec[i] = (i == bid) ? 1.0f : 0.0f;
        }
        __syncthreads();
        
        // Power iteration
        for (int iter = 0; iter < max_iterations; iter++) {
            // Compute v = A^T * u
            for (int j = tid; j < num_codons; j += blockDim.x) {
                float sum = 0.0f;
                for (int i = 0; i < num_traits; i++) {
                    sum += trait_codon_matrix[i * num_codons + j] * u_vec[i];
                }
                v_vec[j] = sum;
            }
            __syncthreads();
            
            // Normalize v
            float v_norm = 0.0f;
            for (int j = tid; j < num_codons; j += blockDim.x) {
                v_norm += v_vec[j] * v_vec[j];
            }
            
            // Reduce norm
            __shared__ float norm_cache[256];
            norm_cache[tid] = v_norm;
            __syncthreads();
            
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    norm_cache[tid] += norm_cache[tid + s];
                }
                __syncthreads();
            }
            
            if (tid == 0) {
                temp[0] = sqrtf(norm_cache[0]);
            }
            __syncthreads();
            
            float norm = temp[0];
            for (int j = tid; j < num_codons; j += blockDim.x) {
                v_vec[j] /= norm;
            }
            __syncthreads();
            
            // Compute u = A * v
            for (int i = tid; i < num_traits; i += blockDim.x) {
                float sum = 0.0f;
                for (int j = 0; j < num_codons; j++) {
                    sum += trait_codon_matrix[i * num_codons + j] * v_vec[j];
                }
                u_vec[i] = sum;
            }
            __syncthreads();
            
            // Normalize u and compute singular value
            float u_norm = 0.0f;
            for (int i = tid; i < num_traits; i += blockDim.x) {
                u_norm += u_vec[i] * u_vec[i];
            }
            
            norm_cache[tid] = u_norm;
            __syncthreads();
            
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    norm_cache[tid] += norm_cache[tid + s];
                }
                __syncthreads();
            }
            
            if (tid == 0) {
                S[bid] = sqrtf(norm_cache[0]);
            }
            __syncthreads();
            
            float sigma = S[bid];
            for (int i = tid; i < num_traits; i += blockDim.x) {
                u_vec[i] /= sigma;
            }
            __syncthreads();
        }
        
        // Store results
        for (int i = tid; i < num_traits; i += blockDim.x) {
            U[i * k + bid] = u_vec[i];
        }
        for (int j = tid; j < num_codons; j += blockDim.x) {
            V[j * k + bid] = v_vec[j];
        }
    }
}
"#;

pub struct MatrixProcessor {
    device: Arc<CudaDevice>,
    power_iteration_kernel: String,
    parallel_jacobi_kernel: String,
    correlation_kernel: String,
    pca_kernel: String,
    svd_kernel: String,
}

impl MatrixProcessor {
    pub fn new(device: &super::super::device::CudaDevice) -> CudaResult<Self> {
        let power_iteration_kernel = "power_iteration_kernel".to_string();
        let parallel_jacobi_kernel = "parallel_jacobi_kernel".to_string();
        let correlation_kernel = "correlation_matrix_kernel".to_string();
        let pca_kernel = "pca_trait_separation_kernel".to_string();
        let svd_kernel = "svd_trait_kernel".to_string();
        
        // Compile kernels
        let ptx = cudarc::nvrtc::compile_ptx(MATRIX_OPS_KERNEL)
            .map_err(|e| CudaError::kernel(format!("Failed to compile matrix processor: {}", e)))?;
        
        // Load module
        device.inner()
            .load_ptx(ptx, "matrix_module", &[
                &power_iteration_kernel,
                &parallel_jacobi_kernel,
                &correlation_kernel,
                &pca_kernel,
                &svd_kernel,
            ])
            .map_err(|e| CudaError::kernel(format!("Failed to load kernels: {}", e)))?;
        
        Ok(Self {
            device: device.inner().clone(),
            power_iteration_kernel,
            parallel_jacobi_kernel,
            correlation_kernel,
            pca_kernel,
            svd_kernel,
        })
    }
    
    pub fn eigendecompose(
        &self,
        correlation_matrix: &[f32],
        size: usize,
    ) -> CudaResult<(Vec<f32>, Vec<f32>)> {
        if size * size != correlation_matrix.len() {
            return Err(CudaError::config("Matrix size mismatch"));
        }
        
        // Allocate device memory
        let mut d_matrix = CudaBuffer::from_slice(self.device.clone(), correlation_matrix)?;
        let mut d_eigenvalues = CudaBuffer::<f32>::new(self.device.clone(), size)?;
        let mut d_eigenvectors = CudaBuffer::<f32>::new(self.device.clone(), size * size)?;
        
        // Launch parallel Jacobi kernel for better performance
        let block_size = 256.min(size as u32);
        let shared_mem = (size * 2 * std::mem::size_of::<f32>()) as u32; // cos and sin values
        
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_mem,
        };
        
        let kernel = self.device
            .get_func("matrix_module", &self.parallel_jacobi_kernel)
            .map_err(|e| CudaError::kernel(format!("Failed to get parallel Jacobi kernel: {}", e)))?;
        
        unsafe {
            kernel.launch(
                config,
                (
                    d_matrix.as_device_ptr_mut(),
                    d_eigenvalues.as_device_ptr_mut(),
                    d_eigenvectors.as_device_ptr_mut(),
                    size as u32,
                    100u32, // Max iterations
                    1e-6f32, // Tolerance
                ),
            ).map_err(|e| CudaError::kernel(format!("Parallel Jacobi kernel failed: {}", e)))?;
        }
        
        // Synchronize and get results
        self.device.synchronize()
            .map_err(|e| CudaError::sync("Synchronization failed"))?;
        
        let eigenvectors = d_eigenvectors.to_vec()?;
        let eigenvalues = d_eigenvalues.to_vec()?;
        
        Ok((eigenvalues, eigenvectors))
    }
    
    /// Compute dominant eigenvalue/eigenvector using power iteration
    pub fn dominant_eigen(
        &self,
        matrix: &[f32],
        size: usize,
    ) -> CudaResult<(f32, Vec<f32>)> {
        if size * size != matrix.len() {
            return Err(CudaError::config("Matrix size mismatch"));
        }
        
        // Allocate device memory
        let d_matrix = CudaBuffer::from_slice(self.device.clone(), matrix)?;
        let mut d_eigenvector = CudaBuffer::<f32>::new(self.device.clone(), size)?;
        let mut d_eigenvalue = CudaBuffer::<f32>::new(self.device.clone(), 1)?;
        let d_temp = CudaBuffer::<f32>::new(self.device.clone(), size)?;
        
        // Launch power iteration kernel
        let block_size = 256.min(size as u32);
        let shared_mem = (size + (block_size / 32) as usize) * std::mem::size_of::<f32>() as u32;
        
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_mem,
        };
        
        let kernel = self.device
            .get_func("matrix_module", &self.power_iteration_kernel)
            .map_err(|e| CudaError::kernel(format!("Failed to get power iteration kernel: {}", e)))?;
        
        unsafe {
            kernel.launch(
                config,
                (
                    d_matrix.as_device_ptr(),
                    d_eigenvector.as_device_ptr_mut(),
                    d_eigenvalue.as_device_ptr_mut(),
                    d_temp.as_device_ptr(),
                    size as u32,
                    50u32, // Max iterations
                ),
            ).map_err(|e| CudaError::kernel(format!("Power iteration kernel failed: {}", e)))?;
        }
        
        // Synchronize and get results
        self.device.synchronize()
            .map_err(|e| CudaError::sync("Synchronization failed"))?;
        
        let eigenvector = d_eigenvector.to_vec()?;
        let eigenvalue = d_eigenvalue.to_vec()?;
        
        Ok((eigenvalue[0], eigenvector))
    }
    
    pub fn compute_correlation_matrix(
        &self,
        data_matrix: &[f32],
        rows: usize,
        cols: usize,
    ) -> CudaResult<Vec<f32>> {
        if rows * cols != data_matrix.len() {
            return Err(CudaError::config("Data matrix size mismatch"));
        }
        
        // Compute means and standard deviations
        let (means, std_devs) = Self::compute_stats(data_matrix, rows, cols);
        
        // Allocate device memory
        let d_data = CudaBuffer::from_slice(self.device.clone(), data_matrix)?;
        let d_means = CudaBuffer::from_slice(self.device.clone(), &means)?;
        let d_std_devs = CudaBuffer::from_slice(self.device.clone(), &std_devs)?;
        let mut d_correlation = CudaBuffer::<f32>::new(self.device.clone(), cols * cols)?;
        
        // Launch correlation kernel
        let config = LaunchConfig {
            grid_dim: (cols as u32, cols as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: std::mem::size_of::<f32>() as u32,
        };
        
        let kernel = self.device
            .get_func("matrix_module", &self.correlation_kernel)
            .map_err(|e| CudaError::kernel(format!("Failed to get correlation kernel: {}", e)))?;
        
        unsafe {
            kernel.launch(
                config,
                (
                    d_data.as_device_ptr(),
                    d_correlation.as_device_ptr_mut(),
                    rows as u32,
                    cols as u32,
                    d_means.as_device_ptr(),
                    d_std_devs.as_device_ptr(),
                ),
            ).map_err(|e| CudaError::kernel(format!("Correlation kernel failed: {}", e)))?;
        }
        
        // Synchronize and get results
        self.device.synchronize()
            .map_err(|e| CudaError::sync("Synchronization failed"))?;
        
        d_correlation.to_vec()
    }
    
    fn compute_stats(data: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
        let mut means = vec![0.0; cols];
        let mut std_devs = vec![0.0; cols];
        
        // Compute means
        for col in 0..cols {
            let mut sum = 0.0;
            for row in 0..rows {
                sum += data[row * cols + col];
            }
            means[col] = sum / rows as f32;
        }
        
        // Compute standard deviations
        for col in 0..cols {
            let mut sum_sq = 0.0;
            for row in 0..rows {
                let diff = data[row * cols + col] - means[col];
                sum_sq += diff * diff;
            }
            std_devs[col] = (sum_sq / (rows - 1) as f32).sqrt();
        }
        
        (means, std_devs)
    }
    
    /// Perform PCA-based trait separation
    pub fn pca_trait_separation(
        &self,
        codon_frequencies: &[f32], // sequences x codons
        num_sequences: usize,
        num_codons: usize,
        num_components: usize,
    ) -> CudaResult<(Vec<f32>, Vec<f32>)> { // (principal_components, loadings)
        // First compute correlation matrix
        let correlation = self.compute_correlation_matrix(codon_frequencies, num_sequences, num_codons)?;
        
        // Compute eigenvectors
        let (eigenvalues, eigenvectors) = self.eigendecompose(&correlation, num_codons)?;
        
        // Allocate device memory
        let d_frequencies = CudaBuffer::from_slice(self.device.clone(), codon_frequencies)?;
        let d_eigenvectors = CudaBuffer::from_slice(self.device.clone(), &eigenvectors)?;
        let mut d_principal_components = CudaBuffer::<f32>::new(
            self.device.clone(),
            num_sequences * num_components
        )?;
        let mut d_trait_loadings = CudaBuffer::<f32>::new(
            self.device.clone(),
            num_components * num_codons
        )?;
        
        // Launch PCA kernel
        let block_size = 256;
        let shared_mem = (num_codons * std::mem::size_of::<f32>()) as u32;
        
        let config = LaunchConfig {
            grid_dim: (num_sequences as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: shared_mem,
        };
        
        let kernel = self.device
            .get_func("matrix_module", &self.pca_kernel)
            .map_err(|e| CudaError::kernel(format!("Failed to get PCA kernel: {}", e)))?;
        
        unsafe {
            kernel.launch(
                config,
                (
                    d_frequencies.as_device_ptr(),
                    d_eigenvectors.as_device_ptr(),
                    d_principal_components.as_device_ptr_mut(),
                    d_trait_loadings.as_device_ptr_mut(),
                    num_sequences as u32,
                    num_codons as u32,
                    num_components as u32,
                ),
            ).map_err(|e| CudaError::kernel(format!("PCA kernel failed: {}", e)))?;
        }
        
        // Synchronize and get results
        self.device.synchronize()
            .map_err(|e| CudaError::sync("Synchronization failed"))?;
        
        let principal_components = d_principal_components.to_vec()?;
        let trait_loadings = d_trait_loadings.to_vec()?;
        
        Ok((principal_components, trait_loadings))
    }
    
    /// Perform SVD for trait-codon factorization
    pub fn svd_trait_factorization(
        &self,
        trait_codon_matrix: &[f32], // traits x codons preference matrix
        num_traits: usize,
        num_codons: usize,
    ) -> CudaResult<(Vec<f32>, Vec<f32>, Vec<f32>)> { // (U, S, V)
        let k = num_traits.min(num_codons);
        
        // Allocate device memory
        let d_matrix = CudaBuffer::from_slice(self.device.clone(), trait_codon_matrix)?;
        let mut d_U = CudaBuffer::<f32>::new(self.device.clone(), num_traits * k)?;
        let mut d_S = CudaBuffer::<f32>::new(self.device.clone(), k)?;
        let mut d_V = CudaBuffer::<f32>::new(self.device.clone(), num_codons * k)?;
        
        // Launch SVD kernel
        let block_size = 256;
        let workspace_size = (num_traits + num_codons + 1) * std::mem::size_of::<f32>();
        let shared_mem = workspace_size as u32;
        
        let config = LaunchConfig {
            grid_dim: (k as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: shared_mem,
        };
        
        let kernel = self.device
            .get_func("matrix_module", &self.svd_kernel)
            .map_err(|e| CudaError::kernel(format!("Failed to get SVD kernel: {}", e)))?;
        
        unsafe {
            kernel.launch(
                config,
                (
                    d_matrix.as_device_ptr(),
                    d_U.as_device_ptr_mut(),
                    d_S.as_device_ptr_mut(),
                    d_V.as_device_ptr_mut(),
                    num_traits as u32,
                    num_codons as u32,
                    30u32, // Max iterations
                ),
            ).map_err(|e| CudaError::kernel(format!("SVD kernel failed: {}", e)))?;
        }
        
        // Synchronize and get results
        self.device.synchronize()
            .map_err(|e| CudaError::sync("Synchronization failed"))?;
        
        let U = d_U.to_vec()?;
        let S = d_S.to_vec()?;
        let V = d_V.to_vec()?;
        
        Ok((U, S, V))
    }
    
    /// Identify separable trait components using eigenanalysis
    pub fn identify_trait_components(
        &self,
        codon_frequencies: &[f32],
        num_sequences: usize,
        num_codons: usize,
        variance_threshold: f32, // e.g., 0.95 for 95% variance
    ) -> CudaResult<Vec<(usize, f32, Vec<f32>)>> { // (component_idx, variance_explained, eigenvector)
        // Compute correlation matrix
        let correlation = self.compute_correlation_matrix(codon_frequencies, num_sequences, num_codons)?;
        
        // Compute eigendecomposition
        let (eigenvalues, eigenvectors) = self.eigendecompose(&correlation, num_codons)?;
        
        // Sort eigenvalues and eigenvectors by magnitude
        let mut eigen_pairs: Vec<(usize, f32)> = eigenvalues.iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();
        eigen_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Calculate total variance
        let total_variance: f32 = eigenvalues.iter().sum();
        
        // Select components that explain desired variance
        let mut cumulative_variance = 0.0;
        let mut selected_components = Vec::new();
        
        for (idx, eigenvalue) in eigen_pairs {
            let variance_explained = eigenvalue / total_variance;
            cumulative_variance += variance_explained;
            
            // Extract eigenvector for this component
            let eigenvector: Vec<f32> = (0..num_codons)
                .map(|i| eigenvectors[i * num_codons + idx])
                .collect();
            
            selected_components.push((idx, variance_explained, eigenvector));
            
            if cumulative_variance >= variance_threshold {
                break;
            }
        }
        
        Ok(selected_components)
    }
}