use std::fmt;
use thiserror::Error;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaError as CudarcError;

/// CUDA-specific error types
#[derive(Error, Debug)]
pub enum CudaError {
    #[error("CUDA initialization failed: {0}")]
    InitializationError(String),
    
    #[error("CUDA device error: {0}")]
    DeviceError(String),
    
    #[error("CUDA memory error: {0}")]
    MemoryError(String),
    
    #[error("CUDA kernel error: {0}")]
    KernelError(String),
    
    #[error("CUDA synchronization error: {0}")]
    SyncError(String),
    
    #[error("Invalid CUDA configuration: {0}")]
    ConfigError(String),
    
    #[error("CUDA not available on this system")]
    NotAvailable,
    
    #[cfg(feature = "cuda")]
    #[error("Cudarc error: {0}")]
    CudarcError(#[from] CudarcError),
    
    #[error("Other CUDA error: {0}")]
    Other(String),
}

pub type CudaResult<T> = Result<T, CudaError>;

impl CudaError {
    pub fn initialization<S: Into<String>>(msg: S) -> Self {
        Self::InitializationError(msg.into())
    }
    
    pub fn device<S: Into<String>>(msg: S) -> Self {
        Self::DeviceError(msg.into())
    }
    
    pub fn memory<S: Into<String>>(msg: S) -> Self {
        Self::MemoryError(msg.into())
    }
    
    pub fn kernel<S: Into<String>>(msg: S) -> Self {
        Self::KernelError(msg.into())
    }
    
    pub fn sync<S: Into<String>>(msg: S) -> Self {
        Self::SyncError(msg.into())
    }
    
    pub fn config<S: Into<String>>(msg: S) -> Self {
        Self::ConfigError(msg.into())
    }
}

/// Macro for CUDA error checking
#[macro_export]
macro_rules! cuda_check {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Err($crate::cuda::error::CudaError::from(e)),
        }
    };
    ($expr:expr, $msg:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Err($crate::cuda::error::CudaError::Other(
                format!("{}: {}", $msg, e)
            )),
        }
    };
}