#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, DevicePtr, DeviceSlice};
#[cfg(feature = "cuda")]
use std::sync::Arc;

use super::error::{CudaError, CudaResult};
use std::marker::PhantomData;

/// CUDA device buffer wrapper
#[cfg(feature = "cuda")]
pub struct CudaBuffer<T> {
    ptr: DeviceSlice<T>,
    size: usize,
    device: Arc<CudaDevice>,
}

#[cfg(feature = "cuda")]
impl<T: cudarc::driver::DeviceRepr> CudaBuffer<T> {
    /// Allocate new device buffer
    pub fn new(device: Arc<CudaDevice>, size: usize) -> CudaResult<Self> {
        let ptr = device
            .alloc_zeros(size)
            .map_err(|e| CudaError::memory(format!("Failed to allocate {} elements: {}", size, e)))?;
        
        Ok(Self { ptr, size, device })
    }
    
    /// Allocate and copy from host
    pub fn from_slice(device: Arc<CudaDevice>, data: &[T]) -> CudaResult<Self> {
        let ptr = device
            .htod_sync_copy(data)
            .map_err(|e| CudaError::memory(format!("Failed to copy to device: {}", e)))?;
        
        Ok(Self {
            size: data.len(),
            ptr,
            device,
        })
    }
    
    /// Copy data to host
    pub fn to_vec(&self) -> CudaResult<Vec<T>> {
        self.device
            .dtoh_sync_copy(&self.ptr)
            .map_err(|e| CudaError::memory(format!("Failed to copy from device: {}", e)))
    }
    
    /// Get device pointer
    pub fn as_device_ptr(&self) -> &DeviceSlice<T> {
        &self.ptr
    }
    
    /// Get mutable device pointer
    pub fn as_device_ptr_mut(&mut self) -> &mut DeviceSlice<T> {
        &mut self.ptr
    }
    
    /// Get buffer size
    pub fn len(&self) -> usize {
        self.size
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

/// Pinned host memory for fast transfers
#[cfg(feature = "cuda")]
pub struct PinnedBuffer<T> {
    data: Vec<T>,
    _device: Arc<CudaDevice>,
}

#[cfg(feature = "cuda")]
impl<T: Clone + Default> PinnedBuffer<T> {
    /// Allocate pinned host memory
    pub fn new(device: Arc<CudaDevice>, size: usize) -> CudaResult<Self> {
        // In production, we'd use cudarc's pinned memory allocation
        // For now, use regular Vec as placeholder
        let data = vec![T::default(); size];
        
        Ok(Self {
            data,
            _device: device,
        })
    }
    
    /// Get slice reference
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
    
    /// Get mutable slice reference
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

/// Memory pool for efficient allocation
#[cfg(feature = "cuda")]
pub struct MemoryPool {
    device: Arc<CudaDevice>,
    // In production, implement actual pooling
}

#[cfg(feature = "cuda")]
impl MemoryPool {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }
    
    pub fn alloc<T: cudarc::driver::DeviceRepr>(&self, size: usize) -> CudaResult<CudaBuffer<T>> {
        CudaBuffer::new(self.device.clone(), size)
    }
}

/// Stub implementations when CUDA is not available
#[cfg(not(feature = "cuda"))]
pub struct CudaBuffer<T> {
    _phantom: PhantomData<T>,
}

#[cfg(not(feature = "cuda"))]
pub struct PinnedBuffer<T> {
    _phantom: PhantomData<T>,
}