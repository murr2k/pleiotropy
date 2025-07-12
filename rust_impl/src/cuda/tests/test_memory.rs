use crate::cuda::{CudaDevice, CudaBuffer, PinnedBuffer};

#[test]
fn test_cuda_buffer_allocation() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let buffer: CudaBuffer<f32> = CudaBuffer::new(device.inner().clone(), 1024)
        .expect("Failed to allocate buffer");
    
    assert_eq!(buffer.len(), 1024);
    assert!(!buffer.is_empty());
}

#[test]
fn test_cuda_buffer_copy_to_device() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    
    let buffer = CudaBuffer::from_slice(device.inner().clone(), &data)
        .expect("Failed to copy to device");
    
    assert_eq!(buffer.len(), 100);
}

#[test]
fn test_cuda_buffer_copy_from_device() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    
    let buffer = CudaBuffer::from_slice(device.inner().clone(), &data)
        .expect("Failed to copy to device");
    
    let result = buffer.to_vec().expect("Failed to copy from device");
    
    assert_eq!(result.len(), 100);
    for (i, &val) in result.iter().enumerate() {
        assert_eq!(val, i as f32);
    }
}

#[test]
fn test_pinned_buffer_allocation() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let buffer: PinnedBuffer<f32> = PinnedBuffer::new(device.inner().clone(), 1024)
        .expect("Failed to allocate pinned buffer");
    
    assert_eq!(buffer.as_slice().len(), 1024);
}

#[test]
fn test_memory_stress() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let mut buffers = Vec::new();
    
    // Allocate 100MB in 1MB chunks
    for _ in 0..100 {
        let buffer = CudaBuffer::<f32>::new(device.inner().clone(), 256 * 1024)
            .expect("Failed to allocate buffer");
        buffers.push(buffer);
    }
    
    assert_eq!(buffers.len(), 100);
    
    // Verify we can still do operations
    let test_data = vec![1.0f32; 1024];
    let test_buffer = CudaBuffer::from_slice(device.inner().clone(), &test_data)
        .expect("Failed to allocate after stress");
    
    let result = test_buffer.to_vec().expect("Failed to copy back");
    assert_eq!(result[0], 1.0);
}

#[test]
fn test_zero_size_allocation() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let buffer = CudaBuffer::<f32>::new(device.inner().clone(), 0)
        .expect("Failed to allocate zero-size buffer");
    
    assert_eq!(buffer.len(), 0);
    assert!(buffer.is_empty());
}

#[test]
fn test_device_synchronization() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    
    // Perform some operations
    let data = vec![1.0f32; 10000];
    let _buffer = CudaBuffer::from_slice(device.inner().clone(), &data)
        .expect("Failed to copy data");
    
    // Synchronize should complete without error
    device.synchronize().expect("Failed to synchronize");
}