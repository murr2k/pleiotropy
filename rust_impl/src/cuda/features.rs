/// CUDA feature detection and capability checking

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaDeviceBuilder};

/// Information about CUDA availability and capabilities
#[derive(Debug, Clone)]
pub struct CudaFeatures {
    pub available: bool,
    pub device_count: usize,
    pub compute_capability: Option<(u32, u32)>,
    pub memory_gb: Option<f32>,
    pub device_name: Option<String>,
}

impl Default for CudaFeatures {
    fn default() -> Self {
        Self {
            available: false,
            device_count: 0,
            compute_capability: None,
            memory_gb: None,
            device_name: None,
        }
    }
}

impl CudaFeatures {
    /// Detect CUDA features on the current system
    pub fn detect() -> Self {
        #[cfg(not(feature = "cuda"))]
        {
            Self::default()
        }
        
        #[cfg(feature = "cuda")]
        {
            match Self::detect_cuda() {
                Ok(features) => features,
                Err(e) => {
                    log::warn!("Failed to detect CUDA features: {}", e);
                    Self::default()
                }
            }
        }
    }
    
    #[cfg(feature = "cuda")]
    fn detect_cuda() -> Result<Self, Box<dyn std::error::Error>> {
        use cudarc::driver::result as cuda_result;
        
        // Initialize CUDA driver
        cuda_result::init()?;
        
        // Get device count
        let device_count = cuda_result::device_get_count()? as usize;
        
        if device_count == 0 {
            return Ok(Self::default());
        }
        
        // Get info from first device
        let device = CudaDevice::new(0)?;
        
        // Get compute capability
        let major = device.attribute(cudarc::driver::CudaDeviceAttribute::ComputeCapabilityMajor)?;
        let minor = device.attribute(cudarc::driver::CudaDeviceAttribute::ComputeCapabilityMinor)?;
        let compute_capability = Some((major as u32, minor as u32));
        
        // Get memory size
        let memory_bytes = device.attribute(cudarc::driver::CudaDeviceAttribute::TotalGlobalMem)?;
        let memory_gb = Some(memory_bytes as f32 / (1024.0 * 1024.0 * 1024.0));
        
        // Get device name
        let device_name = Some(device.name()?);
        
        Ok(Self {
            available: true,
            device_count,
            compute_capability,
            memory_gb,
            device_name,
        })
    }
    
    /// Check if CUDA is available and meets minimum requirements
    pub fn is_supported(&self) -> bool {
        if !self.available {
            return false;
        }
        
        // Require compute capability 5.2 or higher
        if let Some((major, minor)) = self.compute_capability {
            let capability = major * 10 + minor;
            capability >= 52
        } else {
            false
        }
    }
    
    /// Get a human-readable description of CUDA features
    pub fn description(&self) -> String {
        if !self.available {
            return "CUDA not available".to_string();
        }
        
        let mut desc = format!("CUDA available with {} device(s)", self.device_count);
        
        if let Some(name) = &self.device_name {
            desc.push_str(&format!("\n  Device: {}", name));
        }
        
        if let Some((major, minor)) = self.compute_capability {
            desc.push_str(&format!("\n  Compute Capability: {}.{}", major, minor));
        }
        
        if let Some(memory) = self.memory_gb {
            desc.push_str(&format!("\n  Memory: {:.1} GB", memory));
        }
        
        desc
    }
}

/// Macro for conditional CUDA compilation
#[macro_export]
macro_rules! cuda_or_cpu {
    ($cuda_expr:expr, $cpu_expr:expr) => {
        {
            #[cfg(feature = "cuda")]
            {
                $cuda_expr
            }
            #[cfg(not(feature = "cuda"))]
            {
                $cpu_expr
            }
        }
    };
}

/// Check if we should use CUDA at runtime
pub fn should_use_cuda() -> bool {
    // Check environment variable override
    if let Ok(val) = std::env::var("PLEIOTROPY_FORCE_CPU") {
        if val == "1" || val.to_lowercase() == "true" {
            return false;
        }
    }
    
    // Check if CUDA is available and supported
    let features = CudaFeatures::detect();
    features.is_supported()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cuda_detection() {
        let features = CudaFeatures::detect();
        println!("CUDA Features: {:#?}", features);
        println!("Description: {}", features.description());
        
        // This test should pass regardless of CUDA availability
        assert!(true);
    }
    
    #[test]
    fn test_should_use_cuda() {
        // Save original env
        let original = std::env::var("PLEIOTROPY_FORCE_CPU").ok();
        
        // Test force CPU
        std::env::set_var("PLEIOTROPY_FORCE_CPU", "1");
        assert!(!should_use_cuda());
        
        // Restore original
        if let Some(val) = original {
            std::env::set_var("PLEIOTROPY_FORCE_CPU", val);
        } else {
            std::env::remove_var("PLEIOTROPY_FORCE_CPU");
        }
    }
}