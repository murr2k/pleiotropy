//! Performance monitoring and reporting for prime factorization
//!
//! Generates detailed performance reports with visualizations

use crate::prime_factorization::{FactorizationResult, TimingBreakdown};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};

/// Performance test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTestResult {
    pub timestamp: u64,
    pub test_name: String,
    pub number: u64,
    pub factors: Vec<u64>,
    pub cpu_timing: TimingStats,
    pub cuda_timing: Option<TimingStats>,
    pub speedup: Option<f64>,
    pub system_info: SystemInfo,
}

/// Timing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStats {
    pub min_ns: u64,
    pub max_ns: u64,
    pub mean_ns: u64,
    pub median_ns: u64,
    pub std_dev_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
    pub samples: usize,
}

/// System information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub cpu_info: String,
    pub gpu_info: Option<String>,
    pub os: String,
    pub rust_version: String,
    pub cuda_version: Option<String>,
}

/// Performance monitor
pub struct PerformanceMonitor {
    results: Vec<PerformanceTestResult>,
    system_info: SystemInfo,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            system_info: SystemInfo::detect(),
        }
    }
    
    /// Record a test result
    pub fn record_result(
        &mut self,
        test_name: String,
        number: u64,
        factors: Vec<u64>,
        cpu_timings: Vec<Duration>,
        cuda_timings: Option<Vec<Duration>>,
    ) {
        let cpu_timing = Self::calculate_stats(&cpu_timings);
        let cuda_timing = cuda_timings.map(|t| Self::calculate_stats(&t));
        
        let speedup = match (&cpu_timing, &cuda_timing) {
            (cpu, Some(cuda)) => Some(cpu.median_ns as f64 / cuda.median_ns as f64),
            _ => None,
        };
        
        let result = PerformanceTestResult {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            test_name,
            number,
            factors,
            cpu_timing,
            cuda_timing,
            speedup,
            system_info: self.system_info.clone(),
        };
        
        self.results.push(result);
    }
    
    /// Calculate timing statistics
    fn calculate_stats(timings: &[Duration]) -> TimingStats {
        let mut nanos: Vec<u64> = timings.iter().map(|d| d.as_nanos() as u64).collect();
        nanos.sort_unstable();
        
        let min_ns = nanos[0];
        let max_ns = nanos[nanos.len() - 1];
        let sum: u64 = nanos.iter().sum();
        let mean_ns = sum / nanos.len() as u64;
        let median_ns = nanos[nanos.len() / 2];
        
        // Standard deviation
        let variance = nanos.iter()
            .map(|&n| {
                let diff = n as i64 - mean_ns as i64;
                (diff * diff) as u64
            })
            .sum::<u64>() / nanos.len() as u64;
        let std_dev_ns = (variance as f64).sqrt() as u64;
        
        // Percentiles
        let p95_idx = (nanos.len() as f64 * 0.95) as usize;
        let p99_idx = (nanos.len() as f64 * 0.99) as usize;
        let p95_ns = nanos[p95_idx.min(nanos.len() - 1)];
        let p99_ns = nanos[p99_idx.min(nanos.len() - 1)];
        
        TimingStats {
            min_ns,
            max_ns,
            mean_ns,
            median_ns,
            std_dev_ns,
            p95_ns,
            p99_ns,
            samples: nanos.len(),
        }
    }
    
    /// Generate HTML report
    pub fn generate_html_report(&self, output_path: &Path) -> std::io::Result<()> {
        let mut file = File::create(output_path)?;
        
        writeln!(file, "<!DOCTYPE html>")?;
        writeln!(file, "<html>")?;
        writeln!(file, "<head>")?;
        writeln!(file, "<title>Prime Factorization Performance Report</title>")?;
        writeln!(file, "<style>")?;
        writeln!(file, "{}", include_str!("report_style.css"))?;
        writeln!(file, "</style>")?;
        writeln!(file, "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>")?;
        writeln!(file, "</head>")?;
        writeln!(file, "<body>")?;
        
        writeln!(file, "<h1>Prime Factorization Performance Report</h1>")?;
        
        // System info
        writeln!(file, "<div class='section'>")?;
        writeln!(file, "<h2>System Information</h2>")?;
        writeln!(file, "<table>")?;
        writeln!(file, "<tr><th>Component</th><th>Details</th></tr>")?;
        writeln!(file, "<tr><td>CPU</td><td>{}</td></tr>", self.system_info.cpu_info)?;
        writeln!(file, "<tr><td>GPU</td><td>{}</td></tr>", 
            self.system_info.gpu_info.as_deref().unwrap_or("Not available"))?;
        writeln!(file, "<tr><td>OS</td><td>{}</td></tr>", self.system_info.os)?;
        writeln!(file, "<tr><td>Rust Version</td><td>{}</td></tr>", self.system_info.rust_version)?;
        writeln!(file, "<tr><td>CUDA Version</td><td>{}</td></tr>", 
            self.system_info.cuda_version.as_deref().unwrap_or("Not available"))?;
        writeln!(file, "</table>")?;
        writeln!(file, "</div>")?;
        
        // Results table
        writeln!(file, "<div class='section'>")?;
        writeln!(file, "<h2>Test Results</h2>")?;
        writeln!(file, "<table>")?;
        writeln!(file, "<tr>")?;
        writeln!(file, "<th>Test Name</th>")?;
        writeln!(file, "<th>Number</th>")?;
        writeln!(file, "<th>Factors</th>")?;
        writeln!(file, "<th>CPU Time (median)</th>")?;
        writeln!(file, "<th>CUDA Time (median)</th>")?;
        writeln!(file, "<th>Speedup</th>")?;
        writeln!(file, "</tr>")?;
        
        for result in &self.results {
            writeln!(file, "<tr>")?;
            writeln!(file, "<td>{}</td>", result.test_name)?;
            writeln!(file, "<td>{}</td>", result.number)?;
            writeln!(file, "<td>{:?}</td>", result.factors)?;
            writeln!(file, "<td>{}</td>", Self::format_time_ns(result.cpu_timing.median_ns))?;
            writeln!(file, "<td>{}</td>", 
                result.cuda_timing.as_ref()
                    .map(|t| Self::format_time_ns(t.median_ns))
                    .unwrap_or_else(|| "N/A".to_string()))?;
            writeln!(file, "<td>{}</td>", 
                result.speedup
                    .map(|s| format!("{:.2}x", s))
                    .unwrap_or_else(|| "N/A".to_string()))?;
            writeln!(file, "</tr>")?;
        }
        
        writeln!(file, "</table>")?;
        writeln!(file, "</div>")?;
        
        // Performance charts
        writeln!(file, "<div class='section'>")?;
        writeln!(file, "<h2>Performance Visualization</h2>")?;
        writeln!(file, "<div id='speedup-chart'></div>")?;
        writeln!(file, "<div id='timing-chart'></div>")?;
        writeln!(file, "</div>")?;
        
        // JavaScript for charts
        writeln!(file, "<script>")?;
        self.write_chart_js(&mut file)?;
        writeln!(file, "</script>")?;
        
        writeln!(file, "</body>")?;
        writeln!(file, "</html>")?;
        
        Ok(())
    }
    
    /// Format time in nanoseconds to human-readable string
    fn format_time_ns(ns: u64) -> String {
        if ns >= 1_000_000_000 {
            format!("{:.3}s", ns as f64 / 1_000_000_000.0)
        } else if ns >= 1_000_000 {
            format!("{:.3}ms", ns as f64 / 1_000_000.0)
        } else if ns >= 1_000 {
            format!("{:.3}μs", ns as f64 / 1_000.0)
        } else {
            format!("{}ns", ns)
        }
    }
    
    /// Write JavaScript for charts
    fn write_chart_js(&self, file: &mut File) -> std::io::Result<()> {
        // Speedup chart data
        let test_names: Vec<_> = self.results.iter()
            .filter(|r| r.speedup.is_some())
            .map(|r| format!("'{}'", r.test_name))
            .collect();
        let speedups: Vec<_> = self.results.iter()
            .filter_map(|r| r.speedup)
            .collect();
        
        if !speedups.is_empty() {
            writeln!(file, "var speedupData = {{")?;
            writeln!(file, "  x: [{}],", test_names.join(", "))?;
            writeln!(file, "  y: [{:?}],", speedups)?;
            writeln!(file, "  type: 'bar',")?;
            writeln!(file, "  name: 'Speedup'")?;
            writeln!(file, "}};")?;
            
            writeln!(file, "var speedupLayout = {{")?;
            writeln!(file, "  title: 'CUDA Speedup vs CPU',")?;
            writeln!(file, "  yaxis: {{ title: 'Speedup Factor' }}")?;
            writeln!(file, "}};")?;
            
            writeln!(file, "Plotly.newPlot('speedup-chart', [speedupData], speedupLayout);")?;
        }
        
        // Timing comparison chart
        let cpu_times: Vec<_> = self.results.iter()
            .map(|r| r.cpu_timing.median_ns as f64 / 1_000_000.0) // Convert to ms
            .collect();
        let cuda_times: Vec<_> = self.results.iter()
            .filter_map(|r| r.cuda_timing.as_ref().map(|t| t.median_ns as f64 / 1_000_000.0))
            .collect();
        
        writeln!(file, "var cpuTrace = {{")?;
        writeln!(file, "  x: [{}],", test_names.join(", "))?;
        writeln!(file, "  y: {:?},", cpu_times)?;
        writeln!(file, "  type: 'bar',")?;
        writeln!(file, "  name: 'CPU'")?;
        writeln!(file, "}};")?;
        
        if !cuda_times.is_empty() {
            writeln!(file, "var cudaTrace = {{")?;
            writeln!(file, "  x: [{}],", test_names.join(", "))?;
            writeln!(file, "  y: {:?},", cuda_times)?;
            writeln!(file, "  type: 'bar',")?;
            writeln!(file, "  name: 'CUDA'")?;
            writeln!(file, "}};")?;
            
            writeln!(file, "var timingData = [cpuTrace, cudaTrace];")?;
        } else {
            writeln!(file, "var timingData = [cpuTrace];")?;
        }
        
        writeln!(file, "var timingLayout = {{")?;
        writeln!(file, "  title: 'Execution Time Comparison',")?;
        writeln!(file, "  yaxis: {{ title: 'Time (ms)' }},")?;
        writeln!(file, "  barmode: 'group'")?;
        writeln!(file, "}};")?;
        
        writeln!(file, "Plotly.newPlot('timing-chart', timingData, timingLayout);")?;
        
        Ok(())
    }
    
    /// Save results to JSON
    pub fn save_json(&self, path: &Path) -> std::io::Result<()> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, &self.results)?;
        Ok(())
    }
    
    /// Load results from JSON
    pub fn load_json(path: &Path) -> std::io::Result<Vec<PerformanceTestResult>> {
        let file = File::open(path)?;
        let results = serde_json::from_reader(file)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(results)
    }
}

impl SystemInfo {
    /// Detect system information
    fn detect() -> Self {
        Self {
            cpu_info: Self::detect_cpu(),
            gpu_info: Self::detect_gpu(),
            os: Self::detect_os(),
            rust_version: env!("RUSTC_VERSION").to_string(),
            cuda_version: Self::detect_cuda_version(),
        }
    }
    
    fn detect_cpu() -> String {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/cpuinfo")
                .ok()
                .and_then(|content| {
                    content.lines()
                        .find(|line| line.starts_with("model name"))
                        .and_then(|line| line.split(':').nth(1))
                        .map(|s| s.trim().to_string())
                })
                .unwrap_or_else(|| "Unknown CPU".to_string())
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            std::env::var("PROCESSOR_IDENTIFIER")
                .unwrap_or_else(|_| "Unknown CPU".to_string())
        }
    }
    
    fn detect_gpu() -> Option<String> {
        #[cfg(feature = "cuda")]
        {
            // Try to get GPU info from CUDA
            Some("CUDA-capable GPU detected".to_string())
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }
    
    fn detect_os() -> String {
        format!("{} {}", std::env::consts::OS, std::env::consts::ARCH)
    }
    
    fn detect_cuda_version() -> Option<String> {
        #[cfg(feature = "cuda")]
        {
            Some("CUDA enabled".to_string())
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }
}

/// CSS styles for the report
const REPORT_STYLE_CSS: &str = r#"
body {
    font-family: Arial, sans-serif;
    margin: 20px;
    background-color: #f5f5f5;
}

h1 {
    color: #333;
    text-align: center;
}

.section {
    background: white;
    padding: 20px;
    margin: 20px 0;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
}

th, td {
    padding: 10px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}

th {
    background-color: #4CAF50;
    color: white;
}

tr:hover {
    background-color: #f5f5f5;
}

#speedup-chart, #timing-chart {
    width: 100%;
    height: 400px;
    margin: 20px 0;
}
"#;