#!/usr/bin/env python3
"""
Execute genomic pleiotropy analysis on downloaded genomes
"""

import os
import json
import subprocess
import time
from datetime import datetime
import hashlib

def get_available_genomes():
    """Get list of downloaded genome files"""
    genome_dir = "genomes"
    genomes = []
    
    if os.path.exists(genome_dir):
        for file in os.listdir(genome_dir):
            if file.endswith('.fasta'):
                genomes.append(file)
    
    return sorted(genomes)

def run_analysis(genome_file, output_dir):
    """Run genomic cryptanalysis on a single genome"""
    genome_path = os.path.join("genomes", genome_file)
    organism_name = genome_file.replace('.fasta', '')
    
    # Create output directory
    exp_dir = os.path.join(output_dir, f"exp_{organism_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Get trait definitions
    traits_file = os.path.join("genomes", f"{organism_name}_traits.json")
    if not os.path.exists(traits_file):
        # Use standard traits
        traits_file = "standard_traits.json"
    
    # Run analysis
    cmd = [
        "../rust_impl/target/release/genomic_cryptanalysis",
        "--input", genome_path,
        "--traits", traits_file,
        "--output", exp_dir,
        "--min-traits", "2"
    ]
    
    start_time = time.time()
    result = {
        "genome": genome_file,
        "organism": organism_name,
        "start_time": datetime.now().isoformat(),
        "command": " ".join(cmd),
        "output_dir": exp_dir
    }
    
    try:
        # Run the analysis
        process = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        result["success"] = process.returncode == 0
        result["duration"] = end_time - start_time
        result["stdout"] = process.stdout
        result["stderr"] = process.stderr
        
        # Parse results if successful
        if result["success"]:
            results_file = os.path.join(exp_dir, "analysis_results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    result["analysis_results"] = json.load(f)
        
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["duration"] = time.time() - start_time
    
    return result

def create_default_traits(filepath):
    """Create default trait definitions"""
    default_traits = [
        {
            "name": "regulatory",
            "description": "Gene regulation and control",
            "codon_patterns": {
                "CTG": 1.2, "GAA": 1.1, "AAA": 1.1,
                "CGT": 0.8, "AGC": 1.0
            }
        },
        {
            "name": "stress_response", 
            "description": "Response to environmental stress",
            "codon_patterns": {
                "GCG": 1.1, "CCG": 1.2, "AAG": 1.1,
                "TTA": 0.9, "CTA": 0.9
            }
        },
        {
            "name": "metabolism",
            "description": "Metabolic processes",
            "codon_patterns": {
                "ATG": 1.0, "GAC": 1.1, "TGC": 1.1,
                "GGT": 1.0, "ACC": 1.0
            }
        },
        {
            "name": "virulence",
            "description": "Pathogenicity factors",
            "codon_patterns": {
                "TTG": 1.2, "ATT": 1.1, "CTT": 1.1,
                "GCA": 0.9, "TCA": 0.9
            }
        }
    ]
    
    with open(filepath, 'w') as f:
        json.dump(default_traits, f, indent=2)

def main():
    """Run experiments on all available genomes"""
    print("Starting genomic pleiotropy experiments...")
    print("=" * 60)
    
    # Get available genomes
    genomes = get_available_genomes()
    print(f"Found {len(genomes)} genomes to analyze")
    
    if not genomes:
        print("No genomes found! Please download genomes first.")
        return
    
    # Create results directory
    results_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run analysis on each genome
    all_results = []
    successful = 0
    
    for i, genome in enumerate(genomes, 1):
        print(f"\n[{i}/{len(genomes)}] Analyzing {genome}...")
        
        result = run_analysis(genome, results_dir)
        all_results.append(result)
        
        if result["success"]:
            successful += 1
            print(f"✓ Success in {result['duration']:.2f}s")
        else:
            print(f"✗ Failed: {result.get('error', 'Unknown error')}")
    
    # Save summary
    summary = {
        "experiment_date": datetime.now().isoformat(),
        "total_genomes": len(genomes),
        "successful": successful,
        "failed": len(genomes) - successful,
        "results_directory": results_dir,
        "individual_results": all_results
    }
    
    summary_file = os.path.join(results_dir, "experiment_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Experiments complete!")
    print(f"Success rate: {successful}/{len(genomes)} ({successful/len(genomes)*100:.1f}%)")
    print(f"Results saved to: {results_dir}")
    print(f"Summary: {summary_file}")

if __name__ == "__main__":
    main()