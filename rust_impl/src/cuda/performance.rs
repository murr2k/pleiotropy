use std::time::{Duration, Instant};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub kernel_times: HashMap<String, Duration>,
    pub memory_transfers: HashMap<String, (usize, Duration)>, // (bytes, duration)
    pub total_gpu_time: Duration,
    pub total_cpu_time: Duration,
    pub speedup_factor: f64,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            kernel_times: HashMap::new(),
            memory_transfers: HashMap::new(),
            total_gpu_time: Duration::ZERO,
            total_cpu_time: Duration::ZERO,
            speedup_factor: 1.0,
        }
    }
    
    pub fn record_kernel(&mut self, name: &str, duration: Duration) {
        self.kernel_times.insert(name.to_string(), duration);
        self.total_gpu_time += duration;
    }
    
    pub fn record_transfer(&mut self, name: &str, bytes: usize, duration: Duration) {
        self.memory_transfers.insert(name.to_string(), (bytes, duration));
    }
    
    pub fn calculate_speedup(&mut self, cpu_time: Duration) {
        self.total_cpu_time = cpu_time;
        if self.total_gpu_time.as_secs_f64() > 0.0 {
            self.speedup_factor = cpu_time.as_secs_f64() / self.total_gpu_time.as_secs_f64();
        }
    }
    
    pub fn bandwidth_gb_per_sec(&self, transfer_name: &str) -> Option<f64> {
        self.memory_transfers.get(transfer_name).map(|(bytes, duration)| {
            (*bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / duration.as_secs_f64()
        })
    }
    
    pub fn report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("=== CUDA Performance Report ===\n\n");
        
        report.push_str("Kernel Execution Times:\n");
        for (kernel, duration) in &self.kernel_times {
            report.push_str(&format!("  {}: {:.3} ms\n", kernel, duration.as_secs_f64() * 1000.0));
        }
        
        report.push_str("\nMemory Transfer Statistics:\n");
        for (transfer, (bytes, duration)) in &self.memory_transfers {
            let bandwidth = (*bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / duration.as_secs_f64();
            report.push_str(&format!(
                "  {}: {} bytes in {:.3} ms ({:.2} GB/s)\n",
                transfer,
                bytes,
                duration.as_secs_f64() * 1000.0,
                bandwidth
            ));
        }
        
        report.push_str(&format!("\nTotal GPU Time: {:.3} ms\n", self.total_gpu_time.as_secs_f64() * 1000.0));
        report.push_str(&format!("Total CPU Time: {:.3} ms\n", self.total_cpu_time.as_secs_f64() * 1000.0));
        report.push_str(&format!("Speedup Factor: {:.2}x\n", self.speedup_factor));
        
        report
    }
}

pub struct PerformanceProfiler {
    start_time: Instant,
    metrics: PerformanceMetrics,
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            metrics: PerformanceMetrics::new(),
        }
    }
    
    pub fn time_kernel<F, R>(&mut self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        self.metrics.record_kernel(name, duration);
        result
    }
    
    pub fn time_transfer<F, R>(&mut self, name: &str, bytes: usize, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        self.metrics.record_transfer(name, bytes, duration);
        result
    }
    
    pub fn finish(mut self, cpu_comparison_time: Option<Duration>) -> PerformanceMetrics {
        if let Some(cpu_time) = cpu_comparison_time {
            self.metrics.calculate_speedup(cpu_time);
        }
        self.metrics
    }
}

/// GTX 2070 specific optimization parameters
pub struct Gtx2070Optimizer {
    pub optimal_block_size: u32,
    pub optimal_grid_size: u32,
    pub shared_memory_per_block: usize,
    pub max_threads_per_sm: u32,
}

impl Default for Gtx2070Optimizer {
    fn default() -> Self {
        Self {
            optimal_block_size: 256, // Good balance for most kernels
            optimal_grid_size: 144,   // 36 SMs * 4 blocks per SM
            shared_memory_per_block: 48 * 1024, // 48KB
            max_threads_per_sm: 1024,
        }
    }
}

impl Gtx2070Optimizer {
    pub fn calculate_occupancy(&self, threads_per_block: u32, shared_mem_per_block: usize) -> f64 {
        // Simplified occupancy calculation
        let blocks_per_sm_threads = self.max_threads_per_sm / threads_per_block;
        let blocks_per_sm_shared = self.shared_memory_per_block / shared_mem_per_block.max(1);
        let blocks_per_sm = blocks_per_sm_threads.min(blocks_per_sm_shared as u32);
        
        let active_warps = blocks_per_sm * (threads_per_block / 32);
        let max_warps = self.max_threads_per_sm / 32;
        
        (active_warps as f64) / (max_warps as f64)
    }
    
    pub fn optimize_launch_config(
        &self,
        total_work_items: usize,
        shared_mem_per_item: usize,
    ) -> (u32, u32) {
        let block_size = self.optimal_block_size;
        let grid_size = ((total_work_items as u32 + block_size - 1) / block_size)
            .min(self.optimal_grid_size);
        
        (grid_size, block_size)
    }
}