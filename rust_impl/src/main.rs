use anyhow::Result;
use clap::Parser;
use genomic_cryptanalysis::{GenomicCryptanalysis, TraitInfo};
use log::info;
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[clap(author, version, about = "Genomic Pleiotropy Cryptanalysis Tool")]
struct Args {
    /// Input genome file (FASTA format)
    #[clap(short, long)]
    input: PathBuf,

    /// Known traits file (JSON format)
    #[clap(short, long)]
    traits: Option<PathBuf>,

    /// Output directory for results
    #[clap(short, long, default_value = "output")]
    output: PathBuf,

    /// Minimum number of traits for a gene to be considered pleiotropic
    #[clap(short, long, default_value = "2")]
    min_traits: usize,

    /// Verbose output
    #[clap(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    if args.verbose {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Debug)
            .init();
    } else {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Info)
            .init();
    }

    info!("Starting genomic cryptanalysis...");

    // Create output directory
    fs::create_dir_all(&args.output)?;

    // Load known traits if provided
    let known_traits = if let Some(traits_path) = args.traits {
        let traits_json = fs::read_to_string(traits_path)?;
        serde_json::from_str(&traits_json)?
    } else {
        // Use default E. coli traits
        default_ecoli_traits()
    };

    // Initialize cryptanalysis engine
    let mut analyzer = GenomicCryptanalysis::new();

    // Analyze genome
    info!("Analyzing genome file: {:?}", args.input);
    let analysis = analyzer.analyze_genome(&args.input, known_traits)?;

    info!(
        "Analysis complete. Processed {} sequences",
        analysis.sequences
    );

    // Find pleiotropic genes
    let pleiotropic_genes = analyzer.find_pleiotropic_genes(&analysis, args.min_traits);
    info!("Found {} pleiotropic genes", pleiotropic_genes.len());

    // Save results
    let results_path = args.output.join("analysis_results.json");
    let results_json = serde_json::to_string_pretty(&analysis)?;
    fs::write(&results_path, results_json)?;
    info!("Results saved to: {:?}", results_path);

    // Save pleiotropic genes
    let pleiotropic_path = args.output.join("pleiotropic_genes.json");
    let pleiotropic_json = serde_json::to_string_pretty(&pleiotropic_genes)?;
    fs::write(&pleiotropic_path, pleiotropic_json)?;
    info!("Pleiotropic genes saved to: {:?}", pleiotropic_path);

    // Generate summary report
    generate_summary_report(&args.output, &analysis, &pleiotropic_genes)?;

    Ok(())
}

fn default_ecoli_traits() -> Vec<TraitInfo> {
    vec![
        TraitInfo {
            name: "carbon_metabolism".to_string(),
            description: "Carbon source utilization and metabolism".to_string(),
            associated_genes: vec!["crp".to_string(), "cyaA".to_string()],
            known_sequences: vec![],
        },
        TraitInfo {
            name: "stress_response".to_string(),
            description: "Response to environmental stress".to_string(),
            associated_genes: vec!["rpoS".to_string(), "hns".to_string()],
            known_sequences: vec![],
        },
        TraitInfo {
            name: "motility".to_string(),
            description: "Flagellar synthesis and chemotaxis".to_string(),
            associated_genes: vec!["fliA".to_string(), "flhDC".to_string()],
            known_sequences: vec![],
        },
        TraitInfo {
            name: "regulatory".to_string(),
            description: "Gene expression regulation".to_string(),
            associated_genes: vec!["crp".to_string(), "fis".to_string(), "ihfA".to_string()],
            known_sequences: vec![],
        },
        TraitInfo {
            name: "high_expression".to_string(),
            description: "Highly expressed genes".to_string(),
            associated_genes: vec!["rplA".to_string(), "rpsA".to_string()],
            known_sequences: vec![],
        },
        TraitInfo {
            name: "structural".to_string(),
            description: "Structural proteins and components".to_string(),
            associated_genes: vec!["ompA".to_string(), "ftsZ".to_string()],
            known_sequences: vec![],
        },
    ]
}

fn generate_summary_report(
    output_dir: &PathBuf,
    analysis: &genomic_cryptanalysis::PleiotropyAnalysis,
    pleiotropic_genes: &[genomic_cryptanalysis::PleiotropicGene],
) -> Result<()> {
    let mut report = String::new();
    
    report.push_str("# Genomic Pleiotropy Cryptanalysis Report\n\n");
    report.push_str(&format!("## Summary Statistics\n\n"));
    report.push_str(&format!("- Total sequences analyzed: {}\n", analysis.sequences));
    report.push_str(&format!("- Total traits identified: {}\n", analysis.identified_traits.len()));
    report.push_str(&format!("- Pleiotropic genes found: {}\n\n", pleiotropic_genes.len()));
    
    report.push_str("## Top Pleiotropic Genes\n\n");
    report.push_str("| Gene ID | Number of Traits | Confidence | Traits |\n");
    report.push_str("|---------|------------------|------------|--------|\n");
    
    for gene in pleiotropic_genes.iter().take(10) {
        report.push_str(&format!(
            "| {} | {} | {:.3} | {} |\n",
            gene.gene_id,
            gene.traits.len(),
            gene.confidence,
            gene.traits.join(", ")
        ));
    }
    
    report.push_str("\n## Trait Distribution\n\n");
    
    let mut trait_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for sig in &analysis.identified_traits {
        for trait_name in &sig.trait_names {
            *trait_counts.entry(trait_name.clone()).or_insert(0) += 1;
        }
    }
    
    let mut sorted_traits: Vec<_> = trait_counts.into_iter().collect();
    sorted_traits.sort_by(|a, b| b.1.cmp(&a.1));
    
    report.push_str("| Trait | Occurrences |\n");
    report.push_str("|-------|-------------|\n");
    
    for (trait_name, count) in sorted_traits {
        report.push_str(&format!("| {} | {} |\n", trait_name, count));
    }
    
    let report_path = output_dir.join("summary_report.md");
    fs::write(report_path, report)?;
    
    Ok(())
}