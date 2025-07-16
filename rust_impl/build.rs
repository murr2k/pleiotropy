use std::env;
use std::path::PathBuf;

fn main() {
    // Only run CUDA-specific build steps if the cuda feature is enabled
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        println!("cargo:rerun-if-env-changed=CUDA_ROOT");
        println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_ROOT_DIR");
        
        // Find CUDA installation
        let cuda_path = find_cuda_root();
        
        if let Some(cuda_root) = cuda_path {
            println!("cargo:rustc-env=CUDA_ROOT={}", cuda_root.display());
            
            // Add CUDA library paths
            let cuda_lib = cuda_root.join("lib64");
            if !cuda_lib.exists() {
                // Try alternative paths for different CUDA installations
                let alt_lib = cuda_root.join("lib");
                if alt_lib.exists() {
                    println!("cargo:rustc-link-search=native={}", alt_lib.display());
                }
            } else {
                println!("cargo:rustc-link-search=native={}", cuda_lib.display());
            }
            
            // Link CUDA libraries
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=cuda");
            
            // Set include path for CUDA headers
            let cuda_include = cuda_root.join("include");
            if cuda_include.exists() {
                println!("cargo:include={}", cuda_include.display());
            }
            
            // Check for minimum CUDA version (11.0)
            check_cuda_version(&cuda_root);
            
            // Generate PTX compilation flags
            generate_ptx_flags();
            
        } else {
            // CUDA not found but feature requested
            println!("cargo:warning=CUDA feature enabled but CUDA toolkit not found!");
            println!("cargo:warning=Please install CUDA toolkit or set CUDA_PATH environment variable");
            println!("cargo:warning=Building without CUDA support...");
            
            // Fail the build if strict CUDA requirement
            if env::var("CUDA_REQUIRED").is_ok() {
                panic!("CUDA toolkit is required but not found!");
            }
        }
    }
    
    // Platform-specific settings
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    match target_os.as_str() {
        "windows" => {
            println!("cargo:rustc-link-search=native=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64");
        }
        "linux" => {
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
            println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        }
        "macos" => {
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib");
        }
        _ => {}
    }
}

fn find_cuda_root() -> Option<PathBuf> {
    // Check environment variables in order of preference
    let env_vars = ["CUDA_PATH", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR"];
    
    for var in &env_vars {
        if let Ok(path) = env::var(var) {
            let cuda_path = PathBuf::from(path);
            if cuda_path.exists() {
                println!("cargo:info=Found CUDA at: {}", cuda_path.display());
                return Some(cuda_path);
            }
        }
    }
    
    // Check common installation paths
    let common_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-11.8",
        "/usr/local/cuda-12.0",
        "/opt/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0",
    ];
    
    for path in &common_paths {
        let cuda_path = PathBuf::from(path);
        if cuda_path.exists() {
            println!("cargo:info=Found CUDA at: {}", cuda_path.display());
            return Some(cuda_path);
        }
    }
    
    None
}

fn check_cuda_version(cuda_root: &PathBuf) {
    let version_file = cuda_root.join("version.txt");
    if version_file.exists() {
        if let Ok(version_content) = std::fs::read_to_string(&version_file) {
            println!("cargo:info=CUDA Version: {}", version_content.trim());
            
            // Parse version and check if >= 11.0
            if let Some(version_str) = version_content.split_whitespace().last() {
                if let Some(version) = version_str
                    .split('.')
                    .next()
                    .and_then(|v| v.parse::<u32>().ok())
                {
                    if version < 11 {
                        println!("cargo:warning=CUDA version {} is older than recommended version 11.0", version_str);
                    }
                }
            }
        }
    }
}

fn generate_ptx_flags() {
    // Generate PTX compilation flags for different GPU architectures
    // GTX 2070 is compute capability 7.5
    println!("cargo:rustc-env=NVCC_FLAGS=-arch=sm_75 -gencode=arch=compute_75,code=sm_75");
    
    // Also generate PTX for older architectures for compatibility
    println!("cargo:rustc-env=NVCC_COMPUTE_CAPS=75,70,61,52");
    
    // Optimization flags
    println!("cargo:rustc-env=NVCC_OPTIMIZE=-O3 --use_fast_math");
}