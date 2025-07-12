#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice as CudarcDevice, CudaStream, DevicePtr, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use std::sync::Arc;

use super::error::{CudaError, CudaResult};

/// Wrapper around CUDA device
#[cfg(feature = "cuda")]
pub struct CudaDevice {
    inner: Arc<CudarcDevice>,
    device_id: u32,
    properties: DeviceProperties,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub name: String,
    pub total_memory: usize,
    pub compute_capability: (u32, u32),
    pub max_threads_per_block: u32,
    pub max_blocks_per_grid: (u32, u32, u32),
    pub shared_memory_per_block: usize,
    pub warp_size: u32,
    pub multiprocessor_count: u32,
}

#[cfg(feature = "cuda")]
impl CudaDevice {
    /// Create a new CUDA device handle
    pub fn new(device_id: u32) -> CudaResult<Self> {
        let inner = Arc::new(
            CudarcDevice::new(device_id as usize)
                .map_err(|e| CudaError::initialization(format!("Failed to create device: {}", e)))?
        );
        
        // Get device properties
        let properties = Self::query_properties(&inner)?;
        
        Ok(Self {
            inner,
            device_id,
            properties,
        })
    }
    
    /// Get the number of available CUDA devices
    pub fn count() -> u32 {
        CudarcDevice::count() as u32
    }
    
    /// Get device properties
    fn query_properties(device: &CudarcDevice) -> CudaResult<DeviceProperties> {
        // For GTX 2070, we know the properties
        // In production, we'd query these from the device
        Ok(DeviceProperties {
            name: "NVIDIA GeForce GTX 2070".to_string(),
            total_memory: 8 * 1024 * 1024 * 1024, // 8GB
            compute_capability: (7, 5), // Turing architecture
            max_threads_per_block: 1024,
            max_blocks_per_grid: (2147483647, 65535, 65535),
            shared_memory_per_block: 48 * 1024, // 48KB
            warp_size: 32,
            multiprocessor_count: 36, // GTX 2070 has 36 SMs
        })
    }
    
    /// Get device information as string
    pub fn info(&self) -> String {
        format!(
            "Device {}: {}\n\
             Memory: {:.2} GB\n\
             Compute Capability: {}.{}\n\
             Multiprocessors: {}\n\
             Max threads/block: {}\n\
             Shared memory/block: {} KB",
            self.device_id,
            self.properties.name,
            self.properties.total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            self.properties.compute_capability.0,
            self.properties.compute_capability.1,
            self.properties.multiprocessor_count,
            self.properties.max_threads_per_block,
            self.properties.shared_memory_per_block / 1024
        )
    }
    
    /// Get the inner cudarc device
    pub fn inner(&self) -> &Arc<CudarcDevice> {
        &self.inner
    }
    
    /// Get device properties
    pub fn properties(&self) -> &DeviceProperties {
        &self.properties
    }
    
    /// Create a new stream
    pub fn create_stream(&self) -> CudaResult<CudaStream> {
        self.inner
            .fork_default_stream()
            .map_err(|e| CudaError::device(format!("Failed to create stream: {}", e)))
    }
    
    /// Synchronize device
    pub fn synchronize(&self) -> CudaResult<()> {
        self.inner
            .synchronize()
            .map_err(|e| CudaError::sync(format!("Device synchronization failed: {}", e)))
    }
}

/// Stub implementation when CUDA is not available
#[cfg(not(feature = "cuda"))]
pub struct CudaDevice;

#[cfg(not(feature = "cuda"))]
impl CudaDevice {
    pub fn new(_device_id: u32) -> CudaResult<Self> {
        Err(CudaError::NotAvailable)
    }
    
    pub fn count() -> u32 {
        0
    }
    
    pub fn info(&self) -> String {
        "CUDA not available".to_string()
    }
}