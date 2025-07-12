# CUDA API Reference

Complete API documentation for the CUDA acceleration module.

## Table of Contents

- [Module Structure](#module-structure)
- [Core Types](#core-types)
- [Main Interfaces](#main-interfaces)
- [Kernel Functions](#kernel-functions)
- [Memory Management](#memory-management)
- [Error Handling](#error-handling)
- [Configuration](#configuration)
- [Performance Monitoring](#performance-monitoring)
- [Examples](#examples)

## Module Structure

```
genomic_cryptanalysis::cuda
├── CudaAccelerator      // Main GPU interface
├── CudaDevice          // Device management
├── CudaError           // Error types
├── features            // Feature detection
├── memory              // Memory management
│   ├── CudaBuffer      // GPU memory buffer
│   └── PinnedBuffer    // Pinned host memory
└── kernels             // CUDA kernels
    ├── CodonCounter
    ├── FrequencyCalculator
    ├── PatternMatcher
    └── MatrixProcessor
```

## Core Types

### `CudaAccelerator`

Main interface for GPU-accelerated genomic analysis.

```rust
pub struct CudaAccelerator {
    // Private implementation
}

impl CudaAccelerator {
    /// Create a new CUDA accelerator using the default GPU (device 0)
    /// 
    /// # Errors
    /// Returns `CudaError::NotAvailable` if no CUDA devices found
    /// Returns `CudaError::InitializationFailed` if device init fails
    /// 
    /// # Example
    /// ```
    /// use genomic_cryptanalysis::cuda::CudaAccelerator;
    /// 
    /// let mut accelerator = CudaAccelerator::new()?;
    /// ```
    pub fn new() -> CudaResult<Self>

    /// Create accelerator with specific device
    /// 
    /// # Arguments
    /// * `device_id` - CUDA device index (0-based)
    /// 
    /// # Example
    /// ```
    /// // Use second GPU
    /// let mut accelerator = CudaAccelerator::new_with_device(1)?;
    /// ```
    pub fn new_with_device(device_id: i32) -> CudaResult<Self>

    /// Create with custom configuration
    /// 
    /// # Example
    /// ```
    /// let config = CudaConfig {
    ///     device_id: 0,
    ///     memory_pool_size_mb: 2048,
    ///     use_unified_memory: false,
    /// };
    /// let mut accelerator = CudaAccelerator::with_config(config)?;
    /// ```
    pub fn with_config(config: CudaConfig) -> CudaResult<Self>

    /// Get device information
    pub fn device_info(&self) -> &str

    /// Get current memory usage in bytes
    pub fn memory_usage(&self) -> usize

    /// Reset device and clear all allocations
    pub fn reset(&mut self) -> CudaResult<()>
}
```

### `CudaDevice`

CUDA device management and information.

```rust
pub struct CudaDevice {
    // Private implementation
}

impl CudaDevice {
    /// Get number of available CUDA devices
    /// 
    /// # Example
    /// ```
    /// let device_count = CudaDevice::count();
    /// println!("Found {} CUDA devices", device_count);
    /// ```
    pub fn count() -> usize

    /// Create device handle
    /// 
    /// # Arguments
    /// * `device_id` - Device index (0-based)
    pub fn new(device_id: i32) -> CudaResult<Self>

    /// Get device properties
    pub fn properties(&self) -> DeviceProperties

    /// Get device name
    pub fn name(&self) -> &str

    /// Get compute capability (major, minor)
    pub fn compute_capability(&self) -> (i32, i32)

    /// Get total memory in bytes
    pub fn total_memory(&self) -> usize

    /// Get available memory in bytes
    pub fn available_memory(&self) -> CudaResult<usize>

    /// Get number of streaming multiprocessors
    pub fn sm_count(&self) -> i32

    /// Get maximum threads per block
    pub fn max_threads_per_block(&self) -> i32

    /// Get maximum block dimensions
    pub fn max_block_dims(&self) -> (i32, i32, i32)

    /// Get maximum grid dimensions
    pub fn max_grid_dims(&self) -> (i32, i32, i32)

    /// Get warp size
    pub fn warp_size(&self) -> i32

    /// Check if device supports unified memory
    pub fn supports_unified_memory(&self) -> bool

    /// Set as current device
    pub fn set_current(&self) -> CudaResult<()>
}

#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub name: String,
    pub compute_capability: (i32, i32),
    pub total_memory: usize,
    pub sm_count: i32,
    pub max_threads_per_block: i32,
    pub max_block_dims: (i32, i32, i32),
    pub max_grid_dims: (i32, i32, i32),
    pub warp_size: i32,
    pub memory_clock_rate: i32,  // kHz
    pub memory_bus_width: i32,   // bits
    pub l2_cache_size: i32,      // bytes
    pub max_registers_per_block: i32,
    pub max_shared_memory_per_block: i32,
}
```

### Data Types

```rust
/// DNA sequence for CUDA processing
#[derive(Debug, Clone)]
pub struct DnaSequence {
    pub id: String,
    pub sequence: String,
}

/// Codon count map
pub type CodonCounts = HashMap<String, u32>;

/// CUDA-optimized frequency table
#[derive(Debug, Clone)]
pub struct CudaFrequencyTable {
    pub codon_frequencies: Vec<(String, f32)>,
    pub trait_frequencies: HashMap<String, Vec<f32>>,
}

/// Trait pattern for matching
#[derive(Debug, Clone)]
pub struct TraitPattern {
    pub name: String,
    pub codon_preferences: Vec<(String, f32)>,
    pub min_score: f32,
}

/// Pattern match result
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub trait_name: String,
    pub position: usize,
    pub score: f32,
    pub sequence_id: String,
}

/// Performance statistics
#[derive(Debug, Clone, Default)]
pub struct GpuStats {
    pub kernel_launches: usize,
    pub memory_transfers: usize,
    pub bytes_transferred: usize,
    pub compute_time_ms: f64,
    pub transfer_time_ms: f64,
    pub total_time_ms: f64,
}
```

## Main Interfaces

### Codon Counting

```rust
impl CudaAccelerator {
    /// Count codons in DNA sequences using GPU
    /// 
    /// # Arguments
    /// * `sequences` - DNA sequences to process
    /// 
    /// # Returns
    /// Vector of codon count maps, one per sequence
    /// 
    /// # Performance
    /// - Processes ~40 GB/s on RTX 2070
    /// - Automatically handles sequences of any length
    /// - Uses shared memory for optimal performance
    /// 
    /// # Example
    /// ```
    /// let sequences = vec![
    ///     DnaSequence { id: "seq1".into(), sequence: "ATGATGATG".into() },
    ///     DnaSequence { id: "seq2".into(), sequence: "GCGGCGGCG".into() },
    /// ];
    /// 
    /// let counts = accelerator.count_codons(&sequences)?;
    /// assert_eq!(counts[0].get("ATG"), Some(&3));
    /// ```
    pub fn count_codons(&mut self, sequences: &[DnaSequence]) -> CudaResult<Vec<CodonCounts>>

    /// Count codons using sliding windows
    /// 
    /// # Arguments
    /// * `sequences` - DNA sequences to process
    /// * `window_size` - Size of sliding window in base pairs
    /// * `window_stride` - Stride between windows in base pairs
    /// 
    /// # Returns
    /// Vector of vectors, outer = sequences, inner = windows
    /// 
    /// # Example
    /// ```
    /// let window_counts = accelerator.count_codons_sliding_windows(
    ///     &sequences,
    ///     300,  // 100 codons
    ///     150   // 50% overlap
    /// )?;
    /// ```
    pub fn count_codons_sliding_windows(
        &mut self,
        sequences: &[DnaSequence],
        window_size: usize,
        window_stride: usize
    ) -> CudaResult<Vec<Vec<CodonCounts>>>
}
```

### Frequency Calculation

```rust
impl CudaAccelerator {
    /// Calculate frequency tables from codon counts
    /// 
    /// # Arguments
    /// * `codon_counts` - Codon counts from count_codons()
    /// * `traits` - Trait information for frequency calculation
    /// 
    /// # Returns
    /// CUDA-optimized frequency table
    /// 
    /// # Example
    /// ```
    /// let traits = vec![
    ///     TraitInfo { name: "growth".into(), weight: 1.0 },
    ///     TraitInfo { name: "stress".into(), weight: 0.8 },
    /// ];
    /// 
    /// let freq_table = accelerator.calculate_frequencies(&counts, &traits)?;
    /// ```
    pub fn calculate_frequencies(
        &mut self,
        codon_counts: &[CodonCounts],
        traits: &[TraitInfo]
    ) -> CudaResult<CudaFrequencyTable>

    /// Calculate with custom normalization
    /// 
    /// # Arguments
    /// * `normalization` - Normalization method (see NormalizationMethod enum)
    pub fn calculate_frequencies_normalized(
        &mut self,
        codon_counts: &[CodonCounts],
        traits: &[TraitInfo],
        normalization: NormalizationMethod
    ) -> CudaResult<CudaFrequencyTable>
}

#[derive(Debug, Clone, Copy)]
pub enum NormalizationMethod {
    /// No normalization
    None,
    /// Normalize by total codon count
    TotalCount,
    /// Normalize by expected frequency
    Expected,
    /// Z-score normalization
    ZScore,
}
```

### Pattern Matching

```rust
impl CudaAccelerator {
    /// Match trait patterns against frequency table
    /// 
    /// # Arguments
    /// * `frequency_table` - Frequency table from calculate_frequencies()
    /// * `trait_patterns` - Patterns to search for
    /// 
    /// # Returns
    /// Vector of pattern matches
    /// 
    /// # Example
    /// ```
    /// let patterns = vec![
    ///     TraitPattern {
    ///         name: "carbon_metabolism".into(),
    ///         codon_preferences: vec![("ATG".into(), 1.2), ("GCG".into(), 1.1)],
    ///         min_score: 0.7,
    ///     },
    /// ];
    /// 
    /// let matches = accelerator.match_patterns(&freq_table, &patterns)?;
    /// ```
    pub fn match_patterns(
        &mut self,
        frequency_table: &CudaFrequencyTable,
        trait_patterns: &[TraitPattern]
    ) -> CudaResult<Vec<PatternMatch>>

    /// Match patterns with sliding windows
    /// 
    /// # Arguments
    /// * `sequence_frequencies` - Flattened frequency values
    /// * `trait_patterns` - Patterns to match
    /// * `window_size` - Window size in codons
    /// * `window_stride` - Window stride in codons
    /// * `seq_length` - Total sequence length
    /// 
    /// # Returns
    /// Tuple of (trait_name, position, confidence_score)
    pub fn match_patterns_sliding_window(
        &mut self,
        sequence_frequencies: &[f32],
        trait_patterns: &[TraitPattern],
        window_size: usize,
        window_stride: usize,
        seq_length: usize
    ) -> CudaResult<Vec<(String, usize, f64)>>

    /// Match NeuroDNA trait patterns
    /// 
    /// # Arguments
    /// * `frequency_table` - Frequency table
    /// * `neurodna_traits` - NeuroDNA trait definitions
    /// 
    /// # Example
    /// ```
    /// let mut neurodna_traits = HashMap::new();
    /// neurodna_traits.insert(
    ///     "metabolism".into(),
    ///     vec!["ATG".into(), "TAA".into(), "GCG".into()]
    /// );
    /// 
    /// let matches = accelerator.match_neurodna_patterns(
    ///     &freq_table,
    ///     &neurodna_traits
    /// )?;
    /// ```
    pub fn match_neurodna_patterns(
        &mut self,
        frequency_table: &CudaFrequencyTable,
        neurodna_traits: &HashMap<String, Vec<String>>
    ) -> CudaResult<Vec<PatternMatch>>
}
```

### Matrix Operations

```rust
impl CudaAccelerator {
    /// Perform eigenanalysis on correlation matrix
    /// 
    /// # Arguments
    /// * `correlation_matrix` - Flattened correlation matrix (row-major)
    /// * `size` - Matrix dimension (size x size)
    /// 
    /// # Returns
    /// Tuple of (eigenvalues, eigenvectors)
    /// 
    /// # Example
    /// ```
    /// let matrix = vec![1.0, 0.5, 0.5, 1.0]; // 2x2 matrix
    /// let (eigenvalues, eigenvectors) = accelerator.eigenanalysis(&matrix, 2)?;
    /// ```
    pub fn eigenanalysis(
        &mut self,
        correlation_matrix: &[f32],
        size: usize
    ) -> CudaResult<(Vec<f32>, Vec<f32>)>

    /// PCA-based trait separation
    /// 
    /// # Arguments
    /// * `codon_frequencies` - Input frequency matrix
    /// * `num_sequences` - Number of sequences (rows)
    /// * `num_codons` - Number of codons (columns)
    /// * `num_components` - Number of principal components to extract
    /// 
    /// # Returns
    /// Tuple of (components, explained_variance)
    pub fn pca_trait_separation(
        &mut self,
        codon_frequencies: &[f32],
        num_sequences: usize,
        num_codons: usize,
        num_components: usize
    ) -> CudaResult<(Vec<f32>, Vec<f32>)>

    /// Identify trait components using eigenanalysis
    /// 
    /// # Arguments
    /// * `variance_threshold` - Minimum explained variance ratio (0.0-1.0)
    /// 
    /// # Returns
    /// Vector of (component_index, variance_ratio, component_vector)
    pub fn identify_trait_components(
        &mut self,
        codon_frequencies: &[f32],
        num_sequences: usize,
        num_codons: usize,
        variance_threshold: f32
    ) -> CudaResult<Vec<(usize, f32, Vec<f32>)>>

    /// Compute correlation matrix
    /// 
    /// # Arguments
    /// * `data` - Input data matrix (row-major)
    /// * `num_rows` - Number of rows
    /// * `num_cols` - Number of columns
    /// 
    /// # Returns
    /// Correlation matrix (flattened, row-major)
    pub fn compute_correlation_matrix(
        &mut self,
        data: &[f32],
        num_rows: usize,
        num_cols: usize
    ) -> CudaResult<Vec<f32>>
}
```

## Kernel Functions

### CodonCounter Kernel

```rust
pub struct CodonCounter {
    // Private implementation
}

impl CodonCounter {
    /// Create new codon counter
    pub fn new(device: &CudaDevice) -> CudaResult<Self>

    /// Get optimal block size for sequence length
    pub fn optimal_block_size(&self, seq_length: usize) -> usize

    /// Count codons in batch
    /// 
    /// # Performance Notes
    /// - Uses shared memory for codon lookup
    /// - Coalesced global memory access
    /// - Atomic operations for count accumulation
    pub fn count(&mut self, sequences: &[DnaSequence]) -> CudaResult<Vec<CodonCounts>>

    /// Count with custom parameters
    pub fn count_with_params(
        &mut self,
        sequences: &[DnaSequence],
        params: CodonCounterParams
    ) -> CudaResult<Vec<CodonCounts>>
}

#[derive(Debug, Clone)]
pub struct CodonCounterParams {
    pub threads_per_block: usize,
    pub sequences_per_block: usize,
    pub use_shared_memory: bool,
    pub min_sequence_length: usize,
}

impl Default for CodonCounterParams {
    fn default() -> Self {
        Self {
            threads_per_block: 256,
            sequences_per_block: 4,
            use_shared_memory: true,
            min_sequence_length: 90, // 30 codons
        }
    }
}
```

### FrequencyCalculator Kernel

```rust
pub struct FrequencyCalculator {
    // Private implementation
}

impl FrequencyCalculator {
    /// Create new frequency calculator
    pub fn new(device: &CudaDevice) -> CudaResult<Self>

    /// Calculate frequencies from codon counts
    /// 
    /// # Algorithm
    /// 1. Sum total counts per sequence
    /// 2. Normalize to frequencies
    /// 3. Apply trait-specific weighting
    /// 4. Generate frequency table
    pub fn calculate(
        &mut self,
        codon_counts: &[CodonCounts],
        traits: &[TraitInfo]
    ) -> CudaResult<CudaFrequencyTable>

    /// Calculate with chi-square statistics
    pub fn calculate_with_stats(
        &mut self,
        codon_counts: &[CodonCounts],
        traits: &[TraitInfo]
    ) -> CudaResult<(CudaFrequencyTable, FrequencyStats)>
}

#[derive(Debug, Clone)]
pub struct FrequencyStats {
    pub chi_square_values: Vec<f32>,
    pub p_values: Vec<f32>,
    pub degrees_of_freedom: usize,
}
```

### PatternMatcher Kernel

```rust
pub struct PatternMatcher {
    // Private implementation
}

impl PatternMatcher {
    /// Create new pattern matcher
    pub fn new(device: &CudaDevice) -> CudaResult<Self>

    /// Match patterns against frequency table
    /// 
    /// # Algorithm
    /// - Sliding window correlation
    /// - GPU-accelerated scoring
    /// - Parallel pattern evaluation
    pub fn match_patterns(
        &mut self,
        frequency_table: &CudaFrequencyTable,
        trait_patterns: &[TraitPattern]
    ) -> CudaResult<Vec<PatternMatch>>

    /// Configure matching parameters
    pub fn set_params(&mut self, params: PatternMatcherParams)

    /// Get current parameters
    pub fn params(&self) -> &PatternMatcherParams
}

#[derive(Debug, Clone)]
pub struct PatternMatcherParams {
    pub threads_per_block: usize,
    pub patterns_per_thread: usize,
    pub use_texture_memory: bool,
    pub score_threshold: f32,
    pub max_matches_per_sequence: usize,
}
```

### MatrixProcessor Kernel

```rust
pub struct MatrixProcessor {
    // Private implementation
}

impl MatrixProcessor {
    /// Create new matrix processor
    pub fn new(device: &CudaDevice) -> CudaResult<Self>

    /// Eigendecomposition using cuSOLVER
    pub fn eigendecompose(
        &mut self,
        matrix: &[f32],
        size: usize
    ) -> CudaResult<(Vec<f32>, Vec<f32>)>

    /// Matrix multiplication using cuBLAS
    pub fn multiply(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize
    ) -> CudaResult<Vec<f32>>

    /// SVD decomposition
    pub fn svd(
        &mut self,
        matrix: &[f32],
        rows: usize,
        cols: usize
    ) -> CudaResult<(Vec<f32>, Vec<f32>, Vec<f32>)>
}
```

## Memory Management

### CudaBuffer

GPU memory buffer with RAII semantics.

```rust
pub struct CudaBuffer<T> {
    // Private implementation
}

impl<T: Copy> CudaBuffer<T> {
    /// Allocate buffer on GPU
    /// 
    /// # Example
    /// ```
    /// let mut buffer: CudaBuffer<f32> = CudaBuffer::new(1024)?;
    /// ```
    pub fn new(size: usize) -> CudaResult<Self>

    /// Allocate and initialize with value
    pub fn with_value(size: usize, value: T) -> CudaResult<Self>

    /// Copy data from host to device
    /// 
    /// # Example
    /// ```
    /// let data = vec![1.0_f32; 1024];
    /// let mut buffer = CudaBuffer::new(1024)?;
    /// buffer.copy_from_host(&data)?;
    /// ```
    pub fn copy_from_host(&mut self, data: &[T]) -> CudaResult<()>

    /// Copy data from device to host
    pub fn copy_to_host(&self, data: &mut [T]) -> CudaResult<()>

    /// Get raw pointer for kernel launch
    pub fn as_ptr(&self) -> *const T

    /// Get mutable raw pointer
    pub fn as_mut_ptr(&mut self) -> *mut T

    /// Get buffer size
    pub fn len(&self) -> usize

    /// Fill buffer with value
    pub fn fill(&mut self, value: T) -> CudaResult<()>

    /// Async copy from host
    pub fn copy_from_host_async(&mut self, data: &[T], stream: &CudaStream) -> CudaResult<()>

    /// Async copy to host
    pub fn copy_to_host_async(&self, data: &mut [T], stream: &CudaStream) -> CudaResult<()>
}
```

### PinnedBuffer

Page-locked host memory for fast transfers.

```rust
pub struct PinnedBuffer<T> {
    // Private implementation
}

impl<T: Copy> PinnedBuffer<T> {
    /// Allocate pinned host memory
    /// 
    /// # Example
    /// ```
    /// let buffer: PinnedBuffer<f32> = PinnedBuffer::new(1024)?;
    /// ```
    pub fn new(size: usize) -> CudaResult<Self>

    /// Get slice reference
    pub fn as_slice(&self) -> &[T]

    /// Get mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T]

    /// Convert to Vec (copies data)
    pub fn to_vec(&self) -> Vec<T>
}
```

### Memory Pool

```rust
pub struct MemoryPool {
    // Private implementation
}

impl MemoryPool {
    /// Create memory pool with size in bytes
    pub fn new(size_bytes: usize) -> CudaResult<Self>

    /// Allocate from pool
    pub fn allocate<T>(&mut self, count: usize) -> CudaResult<CudaBuffer<T>>

    /// Return buffer to pool
    pub fn deallocate<T>(&mut self, buffer: CudaBuffer<T>)

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats

    /// Clear all allocations
    pub fn clear(&mut self)
}

#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_size: usize,
    pub used_size: usize,
    pub free_size: usize,
    pub num_allocations: usize,
    pub fragmentation_ratio: f32,
}
```

## Error Handling

### Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("CUDA not available on this system")]
    NotAvailable,
    
    #[error("CUDA initialization failed: {0}")]
    InitializationFailed(String),
    
    #[error("GPU memory allocation failed: requested {requested} bytes, available {available} bytes")]
    AllocationFailed {
        requested: usize,
        available: usize,
    },
    
    #[error("Kernel execution failed: {kernel}: {error}")]
    KernelFailed {
        kernel: String,
        error: String,
    },
    
    #[error("Memory transfer failed: {0}")]
    TransferFailed(String),
    
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    
    #[error("Device error: {0}")]
    DeviceError(String),
    
    #[error("CUDA runtime error: {0}")]
    RuntimeError(String),
    
    #[error("cuBLAS error: {0}")]
    CublasError(String),
    
    #[error("cuSOLVER error: {0}")]
    CusolverError(String),
}

pub type CudaResult<T> = Result<T, CudaError>;
```

### Error Handling Utilities

```rust
/// Convert CUDA runtime errors to CudaError
pub fn check_cuda_error(code: cudart::cudaError_t) -> CudaResult<()>

/// Get last CUDA error and clear error state
pub fn get_last_error() -> Option<CudaError>

/// Panic hook for CUDA errors in debug mode
pub fn install_cuda_panic_hook()
```

## Configuration

### CudaConfig

```rust
#[derive(Debug, Clone)]
pub struct CudaConfig {
    /// Device ID (-1 for auto-select)
    pub device_id: i32,
    
    /// Memory pool size in MB (0 to disable)
    pub memory_pool_size_mb: usize,
    
    /// Use unified memory if available
    pub use_unified_memory: bool,
    
    /// Pinned memory allocation size in MB
    pub pinned_memory_size_mb: usize,
    
    /// Enable profiling
    pub enable_profiling: bool,
    
    /// Kernel configuration
    pub kernel_config: KernelConfig,
}

#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Default threads per block
    pub threads_per_block: usize,
    
    /// Codon counter specific
    pub codon_counter: CodonCounterParams,
    
    /// Pattern matcher specific
    pub pattern_matcher: PatternMatcherParams,
    
    /// Enable kernel timing
    pub enable_timing: bool,
}

impl Default for CudaConfig {
    fn default() -> Self {
        Self {
            device_id: -1,
            memory_pool_size_mb: 1024,
            use_unified_memory: false,
            pinned_memory_size_mb: 512,
            enable_profiling: false,
            kernel_config: KernelConfig::default(),
        }
    }
}
```

### Environment Variables

```bash
# Disable CUDA acceleration
export GENOMIC_CRYPTO_DISABLE_CUDA=1

# Force specific GPU
export CUDA_VISIBLE_DEVICES=2

# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1

# Set memory pool size (MB)
export GENOMIC_CRYPTO_CUDA_POOL_SIZE=2048

# Enable kernel profiling
export GENOMIC_CRYPTO_CUDA_PROFILE=1
```

## Performance Monitoring

### GpuMonitor

```rust
pub struct GpuMonitor {
    // Private implementation
}

impl GpuMonitor {
    /// Create monitor for device
    pub fn new(device_id: i32) -> CudaResult<Self>

    /// Start profiling session
    pub fn start_profiling(&mut self)

    /// Stop profiling and get results
    pub fn stop_profiling(&mut self) -> ProfilingResults

    /// Get current GPU utilization (0-100%)
    pub fn gpu_utilization(&self) -> CudaResult<f32>

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> CudaResult<MemoryUsage>

    /// Get temperature in Celsius
    pub fn temperature(&self) -> CudaResult<f32>

    /// Get power usage in watts
    pub fn power_usage(&self) -> CudaResult<f32>

    /// Get PCIe throughput
    pub fn pcie_throughput(&self) -> CudaResult<PcieThroughput>
}

#[derive(Debug, Clone)]
pub struct ProfilingResults {
    pub kernel_times: HashMap<String, Vec<f64>>,
    pub memory_transfers: Vec<TransferInfo>,
    pub total_time_ms: f64,
    pub gpu_active_time_ms: f64,
    pub efficiency: f32,
}

#[derive(Debug, Clone)]
pub struct MemoryUsage {
    pub used: usize,
    pub free: usize,
    pub total: usize,
}

#[derive(Debug, Clone)]
pub struct PcieThroughput {
    pub rx_bytes_per_sec: usize,
    pub tx_bytes_per_sec: usize,
}
```

### Performance Counters

```rust
/// Global performance counters
pub struct CudaCounters {
    pub kernel_launches: AtomicUsize,
    pub bytes_transferred: AtomicUsize,
    pub compute_time_ns: AtomicU64,
    pub transfer_time_ns: AtomicU64,
}

impl CudaCounters {
    /// Get global instance
    pub fn global() -> &'static Self

    /// Reset all counters
    pub fn reset(&self)

    /// Get summary statistics
    pub fn summary(&self) -> CounterSummary
}
```

## Examples

### Complete Analysis Pipeline

```rust
use genomic_cryptanalysis::cuda::*;
use genomic_cryptanalysis::types::*;

fn analyze_genome_cuda(
    fasta_path: &str,
    traits_path: &str
) -> Result<Vec<PleiotropicGene>> {
    // Initialize CUDA
    let mut accelerator = CudaAccelerator::new()?;
    println!("Using GPU: {}", accelerator.device_info());
    
    // Load data
    let sequences = load_sequences(fasta_path)?;
    let traits = load_traits(traits_path)?;
    
    // Count codons on GPU
    let codon_counts = accelerator.count_codons(&sequences)?;
    
    // Calculate frequencies
    let frequency_table = accelerator.calculate_frequencies(
        &codon_counts,
        &traits
    )?;
    
    // Define trait patterns
    let patterns = vec![
        TraitPattern {
            name: "metabolism".into(),
            codon_preferences: vec![
                ("ATG".into(), 1.2),
                ("GCG".into(), 1.1),
            ],
            min_score: 0.7,
        },
        // More patterns...
    ];
    
    // Match patterns
    let matches = accelerator.match_patterns(
        &frequency_table,
        &patterns
    )?;
    
    // Perform eigenanalysis for trait separation
    let correlation_matrix = build_correlation_matrix(&codon_counts);
    let (eigenvalues, eigenvectors) = accelerator.eigenanalysis(
        &correlation_matrix,
        64 // codon count
    )?;
    
    // Convert matches to pleiotropic genes
    let pleiotropic_genes = process_matches(matches, eigenvalues);
    
    Ok(pleiotropic_genes)
}
```

### Streaming Large Genomes

```rust
use futures::stream::{self, StreamExt};

async fn analyze_large_genome_streaming(
    genome_path: &str,
    window_size: usize,
    overlap: usize
) -> Result<Vec<AnalysisResult>> {
    let mut accelerator = CudaAccelerator::new()?;
    let mut results = Vec::new();
    
    // Create genome stream
    let genome_stream = create_genome_stream(genome_path, window_size, overlap)?;
    
    // Process windows in parallel
    let mut stream = stream::iter(genome_stream)
        .map(|window| async {
            let mut acc = accelerator.clone(); // Thread-safe clone
            tokio::task::spawn_blocking(move || {
                acc.analyze_window(window)
            }).await
        })
        .buffer_unordered(4); // Process 4 windows concurrently
    
    while let Some(result) = stream.next().await {
        match result {
            Ok(Ok(analysis)) => results.push(analysis),
            Ok(Err(e)) => eprintln!("Analysis error: {}", e),
            Err(e) => eprintln!("Task error: {}", e),
        }
    }
    
    Ok(results)
}
```

### Custom Kernel Integration

```rust
use genomic_cryptanalysis::cuda::*;

/// Custom kernel for specialized analysis
pub struct CustomAnalyzer {
    device: CudaDevice,
    kernel: CudaKernel,
}

impl CustomAnalyzer {
    pub fn new() -> CudaResult<Self> {
        let device = CudaDevice::new(0)?;
        
        // Load custom kernel
        let ptx = include_str!("custom_kernel.ptx");
        let kernel = CudaKernel::from_ptx(
            &device,
            ptx,
            "custom_analysis_kernel"
        )?;
        
        Ok(Self { device, kernel })
    }
    
    pub fn analyze(&mut self, data: &[f32]) -> CudaResult<Vec<f32>> {
        // Allocate GPU memory
        let mut input = CudaBuffer::new(data.len())?;
        let mut output = CudaBuffer::new(data.len())?;
        
        // Copy input data
        input.copy_from_host(data)?;
        
        // Launch kernel
        let block_size = 256;
        let grid_size = (data.len() + block_size - 1) / block_size;
        
        self.kernel.launch(
            grid_size as u32,
            block_size as u32,
            0, // shared memory
            &[
                input.as_ptr() as *const c_void,
                output.as_mut_ptr() as *mut c_void,
                &(data.len() as u32) as *const _ as *const c_void,
            ]
        )?;
        
        // Get results
        let mut results = vec![0.0; data.len()];
        output.copy_to_host(&mut results)?;
        
        Ok(results)
    }
}
```

### Multi-GPU Coordination

```rust
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct MultiGpuCoordinator {
    accelerators: Vec<Arc<Mutex<CudaAccelerator>>>,
}

impl MultiGpuCoordinator {
    pub fn new() -> CudaResult<Self> {
        let device_count = CudaDevice::count();
        let mut accelerators = Vec::new();
        
        for device_id in 0..device_count {
            let acc = CudaAccelerator::new_with_device(device_id as i32)?;
            accelerators.push(Arc::new(Mutex::new(acc)));
        }
        
        Ok(Self { accelerators })
    }
    
    pub async fn process_batch(
        &self,
        sequences: Vec<DnaSequence>
    ) -> Result<Vec<CodonCounts>> {
        let chunks: Vec<_> = sequences
            .chunks(sequences.len() / self.accelerators.len())
            .map(|c| c.to_vec())
            .collect();
        
        let mut handles = Vec::new();
        
        for (chunk, acc) in chunks.into_iter().zip(&self.accelerators) {
            let acc = Arc::clone(acc);
            let handle = tokio::spawn(async move {
                let mut accelerator = acc.lock().await;
                accelerator.count_codons(&chunk)
            });
            handles.push(handle);
        }
        
        let mut all_results = Vec::new();
        for handle in handles {
            let results = handle.await??;
            all_results.extend(results);
        }
        
        Ok(all_results)
    }
}
```

## See Also

- [CUDA Quick Start](CUDA_QUICK_START.md)
- [CUDA Acceleration Guide](CUDA_ACCELERATION_GUIDE.md)
- [Performance Tuning](CUDA_ACCELERATION_GUIDE.md#performance-tuning)
- [Troubleshooting](CUDA_ACCELERATION_GUIDE.md#troubleshooting)

---

*API Version: 1.0.0*  
*Last Updated: January 2024*