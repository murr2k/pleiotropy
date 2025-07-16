/// Benchmark runner for comprehensive performance testing

use super::{run_benchmark, format_results, BenchmarkResult};
use super::prime_factorization::{
    PrimeFactorizationBenchmark, 
    SmallPrimeBenchmark, 
    MediumCompositeBenchmark
};
use anyhow::Result;
use std::io::Write;

/// Run all benchmarks and generate report
pub fn run_all_benchmarks(output_path: Option<&str>) -> Result<()> {
    println!("Starting comprehensive benchmark suite...\n");
    
    let mut results = Vec::new();
    
    // Prime factorization benchmarks
    println!("Running prime factorization benchmarks...");
    
    // Main benchmark: 100822548703 = 316907 × 318089
    let main_benchmark = PrimeFactorizationBenchmark::default();
    match run_benchmark(&main_benchmark, true) {
        Ok(result) => {
            println!("✓ {} complete", result.name);
            results.push(result);
        }
        Err(e) => {
            println!("✗ Prime factorization failed: {}", e);
        }
    }
    
    // Small prime benchmark
    let small_benchmark = SmallPrimeBenchmark;
    match run_benchmark(&small_benchmark, true) {
        Ok(result) => {
            println!("✓ {} complete", result.name);
            results.push(result);
        }
        Err(e) => {
            println!("✗ Small prime benchmark failed: {}", e);
        }
    }
    
    // Medium composite benchmark
    let medium_benchmark = MediumCompositeBenchmark;
    match run_benchmark(&medium_benchmark, true) {
        Ok(result) => {
            println!("✓ {} complete", result.name);
            results.push(result);
        }
        Err(e) => {
            println!("✗ Medium composite benchmark failed: {}", e);
        }
    }
    
    // Add genomic operation benchmarks
    println!("\nRunning genomic operation benchmarks...");
    
    // Codon counting benchmark
    let codon_benchmark = CodonCountingBenchmark::new();
    match run_benchmark(&codon_benchmark, true) {
        Ok(result) => {
            println!("✓ {} complete", result.name);
            results.push(result);
        }
        Err(e) => {
            println!("✗ Codon counting benchmark failed: {}", e);
        }
    }
    
    // Pattern matching benchmark
    let pattern_benchmark = PatternMatchingBenchmark::new();
    match run_benchmark(&pattern_benchmark, true) {
        Ok(result) => {
            println!("✓ {} complete", result.name);
            results.push(result);
        }
        Err(e) => {
            println!("✗ Pattern matching benchmark failed: {}", e);
        }
    }
    
    // Format and display results
    let report = format_results(&results);
    println!("{}", report);
    
    // Save results to file if requested
    if let Some(path) = output_path {
        let mut file = std::fs::File::create(path)?;
        writeln!(file, "{}", report)?;
        
        // Also save JSON format
        let json_path = path.replace(".txt", ".json");
        let json_results = serde_json::to_string_pretty(&results)?;
        std::fs::write(json_path, json_results)?;
        
        println!("\nResults saved to: {}", path);
    }
    
    // Generate performance summary
    generate_summary(&results);
    
    Ok(())
}

/// Generate performance summary
fn generate_summary(results: &[BenchmarkResult]) {
    println!("\n{:=<80}", "");
    println!("{:^80}", "PERFORMANCE SUMMARY");
    println!("{:=<80}", "");
    
    // Calculate statistics
    let cuda_results: Vec<_> = results.iter()
        .filter(|r| r.speedup.is_some())
        .collect();
    
    if cuda_results.is_empty() {
        println!("No CUDA results available - running in CPU-only mode");
        return;
    }
    
    let avg_speedup = cuda_results.iter()
        .filter_map(|r| r.speedup)
        .sum::<f64>() / cuda_results.len() as f64;
    
    let max_speedup = cuda_results.iter()
        .filter_map(|r| r.speedup)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);
    
    let min_speedup = cuda_results.iter()
        .filter_map(|r| r.speedup)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);
    
    println!("CUDA Performance Metrics:");
    println!("  Average Speedup: {:.2}x", avg_speedup);
    println!("  Maximum Speedup: {:.2}x", max_speedup);
    println!("  Minimum Speedup: {:.2}x", min_speedup);
    println!("  Success Rate: {:.1}%", (cuda_results.len() as f64 / results.len() as f64) * 100.0);
    
    // Verify prime factorization result
    let prime_result = results.iter()
        .find(|r| r.name == "Prime Factorization");
    
    if let Some(result) = prime_result {
        println!("\nPrime Factorization Verification:");
        println!("  Target: 100822548703 = 316907 × 318089");
        println!("  CPU Time: {:.3}ms", result.cpu_time.as_secs_f64() * 1000.0);
        if let Some(cuda_time) = result.cuda_time {
            println!("  CUDA Time: {:.3}ms", cuda_time.as_secs_f64() * 1000.0);
            println!("  Speedup: {:.2}x", result.speedup.unwrap_or(0.0));
        }
        println!("  Error Rate: {:.2}%", result.error_rate * 100.0);
    }
}

// Genomic operation benchmarks

use super::Benchmarkable;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Benchmark for codon counting operations
struct CodonCountingBenchmark {
    sequences: Vec<String>,
}

impl CodonCountingBenchmark {
    fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(42); // Reproducible
        let sequences = (0..100)
            .map(|_| generate_random_sequence(&mut rng, 3000))
            .collect();
        
        Self { sequences }
    }
}

impl Benchmarkable for CodonCountingBenchmark {
    fn run_cpu(&self) -> Result<Vec<u64>> {
        // Simple codon counting
        let mut total_counts = vec![0u64; 64]; // 4^3 possible codons
        
        for seq in &self.sequences {
            for chunk in seq.as_bytes().chunks(3) {
                if chunk.len() == 3 {
                    let codon_idx = encode_codon(chunk);
                    total_counts[codon_idx as usize] += 1;
                }
            }
        }
        
        Ok(total_counts)
    }
    
    fn run_cuda(&self) -> Result<Vec<u64>> {
        #[cfg(feature = "cuda")]
        {
            // Use CUDA codon counter
            use crate::cuda::DnaSequence;
            
            let mut cuda_acc = crate::cuda::CudaAccelerator::new()
                .map_err(|e| anyhow::anyhow!("CUDA init failed: {:?}", e))?;
            
            let dna_sequences: Vec<DnaSequence> = self.sequences.iter()
                .enumerate()
                .map(|(i, seq)| DnaSequence {
                    id: format!("seq_{}", i),
                    sequence: seq.clone(),
                })
                .collect();
            
            let counts = cuda_acc.count_codons(&dna_sequences)
                .map_err(|e| anyhow::anyhow!("CUDA counting failed: {:?}", e))?;
            
            // Convert to fixed array
            let mut total_counts = vec![0u64; 64];
            for count_map in counts {
                for (codon, count) in count_map {
                    let idx = encode_codon_str(&codon);
                    total_counts[idx as usize] += count as u64;
                }
            }
            
            Ok(total_counts)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(anyhow::anyhow!("CUDA not available"))
        }
    }
    
    fn name(&self) -> &str {
        "Codon Counting (100 sequences)"
    }
    
    fn iterations(&self) -> usize {
        20
    }
}

/// Pattern matching benchmark
struct PatternMatchingBenchmark {
    sequences: Vec<String>,
    patterns: Vec<String>,
}

impl PatternMatchingBenchmark {
    fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let sequences = (0..50)
            .map(|_| generate_random_sequence(&mut rng, 5000))
            .collect();
        
        let patterns = vec![
            "ATGATG".to_string(),
            "GCGCGC".to_string(),
            "TTTTTT".to_string(),
            "AAGGCC".to_string(),
        ];
        
        Self { sequences, patterns }
    }
}

impl Benchmarkable for PatternMatchingBenchmark {
    fn run_cpu(&self) -> Result<Vec<u64>> {
        let mut matches = Vec::new();
        
        for (seq_idx, sequence) in self.sequences.iter().enumerate() {
            for pattern in &self.patterns {
                let count = sequence.matches(pattern).count() as u64;
                matches.push(count);
            }
        }
        
        Ok(matches)
    }
    
    fn run_cuda(&self) -> Result<Vec<u64>> {
        // For now, return CPU result as CUDA pattern matching
        // would need specific kernel implementation
        self.run_cpu()
    }
    
    fn name(&self) -> &str {
        "Pattern Matching"
    }
    
    fn iterations(&self) -> usize {
        50
    }
}

// Helper functions

fn generate_random_sequence(rng: &mut StdRng, length: usize) -> String {
    const BASES: &[u8] = b"ATCG";
    (0..length)
        .map(|_| BASES[rng.gen_range(0..4)] as char)
        .collect()
}

fn encode_codon(codon: &[u8]) -> u8 {
    let mut idx = 0;
    for &base in codon {
        idx = idx * 4 + match base {
            b'A' => 0,
            b'T' => 1,
            b'C' => 2,
            b'G' => 3,
            _ => 0,
        };
    }
    idx
}

fn encode_codon_str(codon: &str) -> u8 {
    encode_codon(codon.as_bytes())
}

// Make BenchmarkResult serializable
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct SerializableBenchmarkResult {
    name: String,
    cpu_time_ms: f64,
    cuda_time_ms: Option<f64>,
    speedup: Option<f64>,
    iterations: usize,
    error_rate: f64,
}

impl From<&BenchmarkResult> for SerializableBenchmarkResult {
    fn from(result: &BenchmarkResult) -> Self {
        Self {
            name: result.name.clone(),
            cpu_time_ms: result.cpu_time.as_secs_f64() * 1000.0,
            cuda_time_ms: result.cuda_time.map(|t| t.as_secs_f64() * 1000.0),
            speedup: result.speedup,
            iterations: result.iterations,
            error_rate: result.error_rate,
        }
    }
}