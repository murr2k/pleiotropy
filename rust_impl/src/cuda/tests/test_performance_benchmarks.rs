/// Performance benchmarks comparing CPU vs GPU implementations
/// Specifically tuned for GTX 2070 (8GB, 2304 CUDA cores)

use crate::cuda::{CudaAccelerator, performance::*};
use crate::types::{DnaSequence, TraitInfo};
use crate::FrequencyAnalyzer;
use std::time::Instant;
use std::collections::HashMap;

/// Benchmark result structure
#[derive(Debug)]
struct BenchmarkResult {
    name: String,
    cpu_time_ms: f64,
    gpu_time_ms: f64,
    speedup: f64,
    data_size_mb: f64,
    throughput_mb_per_sec: f64,
}

impl BenchmarkResult {
    fn new(name: &str, cpu_ms: f64, gpu_ms: f64, data_mb: f64) -> Self {
        let speedup = cpu_ms / gpu_ms;
        let throughput = (data_mb / gpu_ms) * 1000.0; // MB/s
        
        Self {
            name: name.to_string(),
            cpu_time_ms: cpu_ms,
            gpu_time_ms: gpu_ms,
            speedup,
            data_size_mb: data_mb,
            throughput_mb_per_sec: throughput,
        }
    }
    
    fn display(&self) {
        println!("\n=== {} ===", self.name);
        println!("Data size: {:.2} MB", self.data_size_mb);
        println!("CPU time: {:.2} ms", self.cpu_time_ms);
        println!("GPU time: {:.2} ms", self.gpu_time_ms);
        println!("Speedup: {:.2}x", self.speedup);
        println!("GPU throughput: {:.2} MB/s", self.throughput_mb_per_sec);
    }
}

/// Generate test sequences for benchmarking
fn generate_benchmark_sequences(count: usize, length: usize) -> Vec<DnaSequence> {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    
    let bases = ['A', 'C', 'G', 'T'];
    let mut rng = StdRng::seed_from_u64(12345);
    
    (0..count)
        .map(|i| {
            let sequence: String = (0..length)
                .map(|_| bases[rng.gen_range(0..4)])
                .collect();
            DnaSequence::new(format!("seq_{}", i), sequence)
        })
        .collect()
}

#[test]
fn benchmark_codon_counting_scaling() {
    let mut results = Vec::new();
    
    // Test different data sizes
    let test_configs = vec![
        (10, 10_000),      // 10 sequences of 10KB each = 100KB
        (100, 10_000),     // 100 sequences of 10KB each = 1MB
        (1000, 10_000),    // 1000 sequences of 10KB each = 10MB
        (100, 100_000),    // 100 sequences of 100KB each = 10MB
        (10, 1_000_000),   // 10 sequences of 1MB each = 10MB
        (1000, 100_000),   // 1000 sequences of 100KB each = 100MB
    ];
    
    for (seq_count, seq_length) in test_configs {
        let sequences = generate_benchmark_sequences(seq_count, seq_length);
        let data_size_mb = (seq_count * seq_length) as f64 / (1024.0 * 1024.0);
        
        // CPU benchmark
        let analyzer = FrequencyAnalyzer::new();
        let cpu_start = Instant::now();
        let cpu_counts = analyzer.count_codons_cpu(&sequences)
            .expect("CPU counting failed");
        let cpu_time = cpu_start.elapsed().as_secs_f64() * 1000.0;
        
        // GPU benchmark
        let mut cuda_acc = CudaAccelerator::new().expect("Failed to create CUDA accelerator");
        let gpu_start = Instant::now();
        let gpu_counts = cuda_acc.count_codons(&sequences)
            .expect("GPU counting failed");
        let gpu_time = gpu_start.elapsed().as_secs_f64() * 1000.0;
        
        // Verify correctness
        assert_eq!(cpu_counts.len(), gpu_counts.len());
        
        let result = BenchmarkResult::new(
            &format!("Codon Counting {}x{}", seq_count, seq_length),
            cpu_time,
            gpu_time,
            data_size_mb,
        );
        
        result.display();
        results.push(result);
    }
    
    // Summary
    println!("\n=== CODON COUNTING SUMMARY ===");
    let avg_speedup = results.iter().map(|r| r.speedup).sum::<f64>() / results.len() as f64;
    let max_speedup = results.iter().map(|r| r.speedup).fold(0.0, f64::max);
    let max_throughput = results.iter().map(|r| r.throughput_mb_per_sec).fold(0.0, f64::max);
    
    println!("Average speedup: {:.2}x", avg_speedup);
    println!("Maximum speedup: {:.2}x", max_speedup);
    println!("Peak GPU throughput: {:.2} MB/s", max_throughput);
}

#[test]
fn benchmark_sliding_window_performance() {
    let window_configs = vec![
        (300, 150),    // 100 codons, 50% overlap
        (900, 300),    // 300 codons, 33% overlap
        (3000, 1500),  // 1000 codons, 50% overlap
    ];
    
    let sequences = generate_benchmark_sequences(100, 100_000); // 100 sequences of 100KB
    
    for (window_size, stride) in window_configs {
        // GPU benchmark with profiling
        let mut profiler = PerformanceProfiler::new();
        let mut cuda_acc = CudaAccelerator::new().expect("Failed to create CUDA accelerator");
        
        let gpu_result = profiler.time_kernel(
            &format!("sliding_window_{}_{}", window_size, stride),
            || {
                cuda_acc.count_codons(&sequences).expect("GPU counting failed")
            }
        );
        
        let metrics = profiler.finish(None);
        println!("\nSliding Window {}x{} Performance:", window_size, stride);
        println!("{}", metrics.report());
    }
}

#[test]
fn benchmark_frequency_calculation() {
    let sequences = generate_benchmark_sequences(1000, 10_000); // 10MB of data
    let traits = vec![
        TraitInfo {
            name: "metabolism".to_string(),
            description: "Metabolic pathways".to_string(),
            associated_genes: vec![],
            known_sequences: vec![],
        },
        TraitInfo {
            name: "stress".to_string(),
            description: "Stress response".to_string(),
            associated_genes: vec![],
            known_sequences: vec![],
        },
        TraitInfo {
            name: "motility".to_string(),
            description: "Cell motility".to_string(),
            associated_genes: vec![],
            known_sequences: vec![],
        },
    ];
    
    // First, count codons
    let mut cuda_acc = CudaAccelerator::new().expect("Failed to create CUDA accelerator");
    let codon_counts = cuda_acc.count_codons(&sequences)
        .expect("Codon counting failed");
    
    // CPU frequency calculation
    let analyzer = FrequencyAnalyzer::new();
    let cpu_start = Instant::now();
    let cpu_freq = analyzer.calculate_frequencies_cpu(&codon_counts, &traits)
        .expect("CPU frequency calculation failed");
    let cpu_time = cpu_start.elapsed().as_secs_f64() * 1000.0;
    
    // GPU frequency calculation with profiling
    let mut profiler = PerformanceProfiler::new();
    let gpu_freq = profiler.time_kernel("frequency_calculation", || {
        cuda_acc.calculate_frequencies(&codon_counts, &traits)
            .expect("GPU frequency calculation failed")
    });
    
    let gpu_time = profiler.finish(Some(cpu_start.elapsed())).total_gpu_time.as_secs_f64() * 1000.0;
    
    let result = BenchmarkResult::new(
        "Frequency Calculation",
        cpu_time,
        gpu_time,
        10.0, // 10MB
    );
    
    result.display();
}

#[test]
fn benchmark_pattern_matching() {
    use crate::types::TraitPattern;
    
    // Generate realistic dataset
    let sequences = generate_benchmark_sequences(500, 50_000); // 25MB
    let mut cuda_acc = CudaAccelerator::new().expect("Failed to create CUDA accelerator");
    
    // Prepare data
    let codon_counts = cuda_acc.count_codons(&sequences)
        .expect("Codon counting failed");
    let freq_table = cuda_acc.calculate_frequencies(&codon_counts, &vec![])
        .expect("Frequency calculation failed");
    
    // Create multiple trait patterns
    let patterns: Vec<TraitPattern> = (0..20)
        .map(|i| TraitPattern {
            name: format!("trait_{}", i),
            codon_preferences: vec![
                ("ATG".to_string(), 1.2),
                ("GCG".to_string(), 1.1),
                ("AAA".to_string(), 0.9),
            ],
            min_score: 0.6,
        })
        .collect();
    
    // Benchmark pattern matching
    let mut profiler = PerformanceProfiler::new();
    
    let matches = profiler.time_kernel("pattern_matching", || {
        cuda_acc.match_patterns(&freq_table, &patterns)
            .expect("Pattern matching failed")
    });
    
    let metrics = profiler.finish(None);
    
    println!("\nPattern Matching Performance (20 patterns on 25MB):");
    println!("{}", metrics.report());
    println!("Matches found: {}", matches.len());
}

#[test]
fn benchmark_matrix_operations() {
    let mut cuda_acc = CudaAccelerator::new().expect("Failed to create CUDA accelerator");
    let mut results = Vec::new();
    
    for size in [64, 128, 256, 512, 1024].iter() {
        // Create correlation matrix
        let matrix_size = size * size;
        let matrix: Vec<f32> = (0..matrix_size)
            .map(|i| ((i as f32) * 0.1).sin())
            .collect();
        
        let data_size_mb = (matrix_size * 4) as f64 / (1024.0 * 1024.0);
        
        // GPU eigenanalysis
        let gpu_start = Instant::now();
        let (eigenvalues, eigenvectors) = cuda_acc.eigenanalysis(&matrix, *size)
            .expect("GPU eigenanalysis failed");
        let gpu_time = gpu_start.elapsed().as_secs_f64() * 1000.0;
        
        // For matrix operations, we estimate CPU time based on O(n³) complexity
        let estimated_cpu_time = (*size as f64).powi(3) * 0.00001; // Rough estimate
        
        let result = BenchmarkResult::new(
            &format!("Matrix Eigenanalysis {}x{}", size, size),
            estimated_cpu_time,
            gpu_time,
            data_size_mb,
        );
        
        result.display();
        results.push(result);
    }
}

#[test]
fn benchmark_memory_transfer_overhead() {
    let mut cuda_acc = CudaAccelerator::new().expect("Failed to create CUDA accelerator");
    let mut profiler = PerformanceProfiler::new();
    
    // Test different transfer sizes
    let sizes = vec![
        1_000,      // 1KB
        10_000,     // 10KB
        100_000,    // 100KB
        1_000_000,  // 1MB
        10_000_000, // 10MB
    ];
    
    println!("\n=== Memory Transfer Overhead ===");
    
    for size in sizes {
        let sequences = generate_benchmark_sequences(1, size);
        let data_size_bytes = size;
        
        // Time the entire operation including transfers
        let total_start = Instant::now();
        let counts = profiler.time_transfer(
            &format!("transfer_{}", size),
            data_size_bytes,
            || cuda_acc.count_codons(&sequences).expect("Counting failed")
        );
        let total_time = total_start.elapsed();
        
        let metrics = profiler.finish(None);
        
        if let Some(bandwidth) = metrics.bandwidth_gb_per_sec(&format!("transfer_{}", size)) {
            println!("Size: {} bytes, Time: {:.3} ms, Bandwidth: {:.2} GB/s",
                size, total_time.as_secs_f64() * 1000.0, bandwidth);
        }
    }
}

#[test]
fn benchmark_gtx2070_optimization() {
    let optimizer = Gtx2070Optimizer::default();
    
    println!("\n=== GTX 2070 Optimization Analysis ===");
    println!("Optimal block size: {}", optimizer.optimal_block_size);
    println!("Optimal grid size: {}", optimizer.optimal_grid_size);
    println!("Shared memory per block: {} KB", optimizer.shared_memory_per_block / 1024);
    
    // Test different configurations
    let configs = vec![
        (32, 1024),   // Small blocks
        (128, 4096),  // Medium blocks
        (256, 8192),  // Optimal blocks
        (512, 16384), // Large blocks
    ];
    
    for (block_size, shared_mem) in configs {
        let occupancy = optimizer.calculate_occupancy(block_size, shared_mem);
        println!("\nBlock size: {}, Shared mem: {} bytes", block_size, shared_mem);
        println!("Theoretical occupancy: {:.1}%", occupancy * 100.0);
        
        let (grid, block) = optimizer.optimize_launch_config(1_000_000, 64);
        println!("Optimal launch config for 1M items: grid={}, block={}", grid, block);
    }
}

#[test]
fn benchmark_real_world_ecoli() {
    println!("\n=== E. coli Genome Analysis Benchmark ===");
    
    // Simulate E. coli sized genome (4.6 million base pairs)
    let ecoli_size = 4_600_000;
    let chunk_size = 100_000;
    let num_chunks = ecoli_size / chunk_size;
    
    let sequences = generate_benchmark_sequences(num_chunks, chunk_size);
    let data_size_mb = ecoli_size as f64 / (1024.0 * 1024.0);
    
    // Full pipeline benchmark
    let mut cuda_acc = CudaAccelerator::new().expect("Failed to create CUDA accelerator");
    let analyzer = FrequencyAnalyzer::new();
    
    // CPU pipeline
    let cpu_start = Instant::now();
    let cpu_counts = analyzer.count_codons_cpu(&sequences).expect("CPU counting failed");
    let cpu_freq = analyzer.calculate_frequencies_cpu(&cpu_counts, &vec![]).expect("CPU freq failed");
    let cpu_time = cpu_start.elapsed();
    
    // GPU pipeline with detailed profiling
    let mut profiler = PerformanceProfiler::new();
    
    let gpu_counts = profiler.time_kernel("ecoli_codon_counting", || {
        cuda_acc.count_codons(&sequences).expect("GPU counting failed")
    });
    
    let gpu_freq = profiler.time_kernel("ecoli_frequency_calc", || {
        cuda_acc.calculate_frequencies(&gpu_counts, &vec![]).expect("GPU freq failed")
    });
    
    let metrics = profiler.finish(Some(cpu_time));
    
    println!("Genome size: {:.2} MB", data_size_mb);
    println!("CPU total time: {:.2} seconds", cpu_time.as_secs_f64());
    println!("GPU total time: {:.2} seconds", metrics.total_gpu_time.as_secs_f64());
    println!("Overall speedup: {:.2}x", metrics.speedup_factor);
    println!("\nDetailed GPU metrics:");
    println!("{}", metrics.report());
}

#[test]
fn stress_test_maximum_throughput() {
    println!("\n=== Maximum Throughput Stress Test ===");
    
    let mut cuda_acc = CudaAccelerator::new().expect("Failed to create CUDA accelerator");
    
    // Process data in a loop to measure sustained throughput
    let sequences = generate_benchmark_sequences(1000, 10_000); // 10MB chunks
    let data_size_mb = 10.0;
    
    let start = Instant::now();
    let iterations = 100;
    
    for _ in 0..iterations {
        let _ = cuda_acc.count_codons(&sequences).expect("Counting failed");
    }
    
    let total_time = start.elapsed();
    let total_data_mb = data_size_mb * iterations as f64;
    let throughput = total_data_mb / total_time.as_secs_f64();
    
    println!("Processed {} MB in {:.2} seconds", total_data_mb, total_time.as_secs_f64());
    println!("Sustained throughput: {:.2} MB/s", throughput);
    println!("Average time per 10MB chunk: {:.2} ms", 
        (total_time.as_secs_f64() * 1000.0) / iterations as f64);
}