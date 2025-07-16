/// Benchmark executable for comparing CPU vs CUDA performance

use genomic_pleiotropy_cryptanalysis::benchmark::runner;
use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about = "Run CUDA vs CPU benchmarks", long_about = None)]
struct Args {
    /// Output file for benchmark results
    #[arg(short, long, default_value = "benchmark_results.txt")]
    output: String,
    
    /// Run only prime factorization benchmarks
    #[arg(long)]
    prime_only: bool,
    
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize logging
    let log_level = if args.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .init();
    
    println!("\n{:=^80}", " GENOMIC PLEIOTROPY CUDA BENCHMARK ");
    println!("{:^80}", "Comparing CPU vs CUDA Performance");
    println!("{:=^80}\n", "");
    
    // Check CUDA availability
    #[cfg(feature = "cuda")]
    {
        if genomic_pleiotropy_cryptanalysis::cuda::cuda_available() {
            println!("✓ CUDA is available");
            if let Some(info) = genomic_pleiotropy_cryptanalysis::cuda::cuda_info() {
                println!("{}", info);
            }
        } else {
            println!("✗ CUDA is not available - running CPU-only benchmarks");
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        println!("✗ CUDA support not compiled - rebuild with --features cuda");
    }
    
    println!();
    
    // Run benchmarks
    runner::run_all_benchmarks(Some(&args.output))?;
    
    println!("\n✓ Benchmark complete!");
    
    Ok(())
}