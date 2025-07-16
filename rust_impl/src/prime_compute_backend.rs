/// Unified compute backend for prime factorization
/// Seamlessly switches between CPU and CUDA implementations

use crate::prime_factorization::{PrimeFactorizer as CpuFactorizer, Factorization};
use anyhow::Result;
use std::time::Instant;

#[cfg(feature = "cuda")]
use crate::cuda::{CudaAccelerator, kernels::PrimeFactorizer as CudaFactorizer};

/// Unified prime factorization backend
pub struct PrimeComputeBackend {
    /// CPU implementation
    cpu_factorizer: CpuFactorizer,
    
    /// CUDA implementation (if available)
    #[cfg(feature = "cuda")]
    cuda_factorizer: Option<CudaFactorizer>,
    
    /// Force CPU usage
    force_cpu: bool,
    
    /// Performance statistics
    stats: PrimeFactorizationStats,
}

#[derive(Default, Debug, Clone)]
pub struct PrimeFactorizationStats {
    pub cpu_factorizations: usize,
    pub cuda_factorizations: usize,
    pub cuda_failures: usize,
    pub total_numbers_factored: usize,
    pub avg_cpu_time_ms: f64,
    pub avg_cuda_time_ms: f64,
    pub largest_factor_found: u64,
}

impl PrimeComputeBackend {
    /// Create a new compute backend with automatic GPU detection
    pub fn new() -> Result<Self> {
        let cpu_factorizer = CpuFactorizer::new();
        
        #[cfg(feature = "cuda")]
        let cuda_factorizer = {
            if std::env::var("PLEIOTROPY_USE_CUDA").unwrap_or_default() != "0" {
                match CudaAccelerator::new() {
                    Ok(acc) => {
                        match CudaFactorizer::new(acc.device.clone()) {
                            Ok(factorizer) => {
                                log::info!("CUDA prime factorization initialized");
                                Some(factorizer)
                            }
                            Err(e) => {
                                log::warn!("Failed to initialize CUDA prime factorizer: {}", e);
                                None
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("Failed to initialize CUDA: {}", e);
                        None
                    }
                }
            } else {
                log::info!("CUDA disabled by user preference");
                None
            }
        };
        
        Ok(Self {
            cpu_factorizer,
            #[cfg(feature = "cuda")]
            cuda_factorizer,
            force_cpu: false,
            stats: PrimeFactorizationStats::default(),
        })
    }
    
    /// Force CPU usage
    pub fn set_force_cpu(&mut self, force: bool) {
        self.force_cpu = force;
        if force {
            log::info!("Forcing CPU backend for prime factorization");
        }
    }
    
    /// Check if CUDA is available
    pub fn is_cuda_available(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            !self.force_cpu && self.cuda_factorizer.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
    
    /// Get performance statistics
    pub fn get_stats(&self) -> &PrimeFactorizationStats {
        &self.stats
    }
    
    /// Factorize a single number
    pub fn factorize(&mut self, number: u64) -> Result<Factorization> {
        let start = Instant::now();
        
        #[cfg(feature = "cuda")]
        {
            if !self.force_cpu && self.cuda_factorizer.is_some() {
                match self.cuda_factorizer.as_ref().unwrap().factorize_single(number) {
                    Ok(factors) => {
                        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                        self.stats.cuda_factorizations += 1;
                        self.stats.total_numbers_factored += 1;
                        self.update_cuda_stats(elapsed);
                        
                        // Update largest factor
                        if let Some(&max_factor) = factors.iter().max() {
                            self.stats.largest_factor_found = 
                                self.stats.largest_factor_found.max(max_factor);
                        }
                        
                        return Ok(Factorization::from_factors(number, factors));
                    }
                    Err(e) => {
                        log::warn!("CUDA factorization failed, falling back to CPU: {}", e);
                        self.stats.cuda_failures += 1;
                    }
                }
            }
        }
        
        // CPU fallback
        let result = self.cpu_factorizer.factorize(number);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        
        self.stats.cpu_factorizations += 1;
        self.stats.total_numbers_factored += 1;
        self.update_cpu_stats(elapsed);
        
        // Update largest factor
        if let Some(&max_factor) = result.factors.iter().max() {
            self.stats.largest_factor_found = 
                self.stats.largest_factor_found.max(max_factor);
        }
        
        Ok(result)
    }
    
    /// Factorize multiple numbers
    pub fn factorize_batch(&mut self, numbers: &[u64]) -> Result<Vec<Factorization>> {
        if numbers.is_empty() {
            return Ok(vec![]);
        }
        
        let start = Instant::now();
        
        #[cfg(feature = "cuda")]
        {
            if !self.force_cpu && self.cuda_factorizer.is_some() && numbers.len() > 10 {
                // Use CUDA for batches larger than 10
                match self.use_cuda_batch(numbers) {
                    Ok(results) => {
                        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                        self.stats.cuda_factorizations += numbers.len();
                        self.stats.total_numbers_factored += numbers.len();
                        self.update_cuda_stats(elapsed);
                        return Ok(results);
                    }
                    Err(e) => {
                        log::warn!("CUDA batch factorization failed: {}", e);
                        self.stats.cuda_failures += 1;
                    }
                }
            }
        }
        
        // CPU fallback
        let results = self.cpu_factorizer.factorize_batch(numbers);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        
        self.stats.cpu_factorizations += numbers.len();
        self.stats.total_numbers_factored += numbers.len();
        self.update_cpu_stats(elapsed);
        
        // Update largest factor
        for result in &results {
            if let Some(&max_factor) = result.factors.iter().max() {
                self.stats.largest_factor_found = 
                    self.stats.largest_factor_found.max(max_factor);
            }
        }
        
        Ok(results)
    }
    
    #[cfg(feature = "cuda")]
    fn use_cuda_batch(&self, numbers: &[u64]) -> Result<Vec<Factorization>> {
        let factorizer = self.cuda_factorizer.as_ref().unwrap();
        
        // Determine which algorithm to use based on number size
        let max_number = numbers.iter().max().copied().unwrap_or(0);
        
        let factor_lists = if max_number < 1_000_000_000_000 {
            // Use basic trial division for smaller numbers
            factorizer.factorize_batch(numbers)?
        } else {
            // Use advanced Pollard's rho for larger numbers
            factorizer.factorize_advanced(numbers)?
        };
        
        // Convert to Factorization objects
        Ok(numbers.iter()
            .zip(factor_lists.iter())
            .map(|(&number, factors)| Factorization::from_factors(number, factors.clone()))
            .collect())
    }
    
    fn update_cpu_stats(&mut self, time_ms: f64) {
        let count = self.stats.cpu_factorizations as f64;
        self.stats.avg_cpu_time_ms = 
            (self.stats.avg_cpu_time_ms * (count - 1.0) + time_ms) / count;
    }
    
    fn update_cuda_stats(&mut self, time_ms: f64) {
        let count = self.stats.cuda_factorizations as f64;
        self.stats.avg_cuda_time_ms = 
            (self.stats.avg_cuda_time_ms * (count - 1.0) + time_ms) / count;
    }
    
    /// Print performance comparison
    pub fn print_performance_summary(&self) {
        println!("\n=== Prime Factorization Performance Summary ===");
        println!("Total numbers factored: {}", self.stats.total_numbers_factored);
        println!("CPU factorizations: {}", self.stats.cpu_factorizations);
        
        #[cfg(feature = "cuda")]
        {
            println!("CUDA factorizations: {}", self.stats.cuda_factorizations);
            println!("CUDA failures: {}", self.stats.cuda_failures);
            
            if self.stats.cuda_factorizations > 0 && self.stats.cpu_factorizations > 0 {
                let speedup = self.stats.avg_cpu_time_ms / self.stats.avg_cuda_time_ms;
                println!("\nAverage CPU time: {:.3} ms", self.stats.avg_cpu_time_ms);
                println!("Average CUDA time: {:.3} ms", self.stats.avg_cuda_time_ms);
                println!("CUDA speedup: {:.2}x", speedup);
            }
        }
        
        println!("Largest factor found: {}", self.stats.largest_factor_found);
    }
}

/// Demo function to test the unified backend
pub fn demo_prime_factorization() -> Result<()> {
    let mut backend = PrimeComputeBackend::new()?;
    
    println!("Prime Factorization Compute Backend Demo");
    println!("CUDA available: {}", backend.is_cuda_available());
    
    // Test with the target number
    let target = 100822548703u64;
    println!("\nFactorizing target number: {}", target);
    
    let result = backend.factorize(target)?;
    println!("Factors: {:?}", result.factors);
    println!("Prime factors: {:?}", result.prime_factors);
    println!("Verification: {}", result.verify());
    
    // Test batch factorization
    let test_numbers = vec![
        12, 35, 77, 1001, 10007,
        100822548703,
        999999999989, // Large prime
        1234567890123,
        9876543210987,
    ];
    
    println!("\nBatch factorization test:");
    let results = backend.factorize_batch(&test_numbers)?;
    
    for (number, result) in test_numbers.iter().zip(results.iter()) {
        println!("{} = {:?}", number, result.factors);
    }
    
    // Performance comparison
    if backend.is_cuda_available() {
        println!("\nPerformance comparison (CPU vs CUDA):");
        
        // Generate test data
        let mut test_data = Vec::new();
        for i in 0..1000 {
            test_data.push(1000000 + i * 1337);
        }
        
        // Test CPU
        backend.set_force_cpu(true);
        let cpu_start = Instant::now();
        let _ = backend.factorize_batch(&test_data)?;
        let cpu_time = cpu_start.elapsed();
        
        // Test CUDA
        backend.set_force_cpu(false);
        let cuda_start = Instant::now();
        let _ = backend.factorize_batch(&test_data)?;
        let cuda_time = cuda_start.elapsed();
        
        println!("CPU time: {:?}", cpu_time);
        println!("CUDA time: {:?}", cuda_time);
        println!("Speedup: {:.2}x", cpu_time.as_secs_f64() / cuda_time.as_secs_f64());
    }
    
    backend.print_performance_summary();
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_unified_backend() {
        let mut backend = PrimeComputeBackend::new().unwrap();
        
        // Test target number
        let target = 100822548703u64;
        let result = backend.factorize(target).unwrap();
        
        assert!(result.verify());
        assert_eq!(result.factors.len(), 2);
        assert!(result.factors.contains(&316907));
        assert!(result.factors.contains(&318089));
    }
    
    #[test]
    fn test_batch_processing() {
        let mut backend = PrimeComputeBackend::new().unwrap();
        
        let numbers = vec![12, 35, 100822548703];
        let results = backend.factorize_batch(&numbers).unwrap();
        
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].factors, vec![2, 2, 3]);
        assert_eq!(results[1].factors, vec![5, 7]);
        assert!(results[2].verify());
    }
}