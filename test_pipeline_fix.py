#!/usr/bin/env python3
"""
Test script to validate the fixed trait detection pipeline
"""

import json
import subprocess
import os
from pathlib import Path

def create_test_genome():
    """Create a test genome with known pleiotropic genes"""
    test_fasta = """
>crp_0001 cAMP receptor protein - carbon metabolism and regulation
ATGGTGCTTGGCAAACCGCAAACAGACCCGACTCTCGAACTGCACGCTGAAAAAGGGCTGAAAGAAGAAGAACTGCTGCTGCTGGATGATGATGATCGCCGCCGCAAAAAAGGG
>fis_0002 Factor for inversion stimulation - regulatory and structural
ATGAAAGAAGAAGAACTGAAAAAAGCGCGCGATGATGATCGCCGCCGCGGGTTTTAAAGATGATGATCGCCGCCGCCGCGGGGATGATGATCGCCGCCGCAAAAAAGGG
>rpoS_0003 RNA polymerase sigma S - stress response and regulation
ATGAGTCAGAATACGCTGAAAGTTGTTGAAGGTATTTTAGGTAAAGAAGAAGAACTGGATGATGATCGCCGCCGCAAAGAAGAAGAACTGCTGCTGAAAAAAGGG
>flhD_0004 Flagellar master regulator - motility and regulation  
ATGCATATTCGTATGGCAGAAGATGCAGAACGTCTGAAAGAAGAAGAACTGCTGCTGGATGATGATGATCGCCGCCGCGGGGATGATGATCGCCGCCGCAAAAAAGGG
>ompA_0005 Outer membrane protein A - structural
ATGAAAAAGACAGCTATCGCGATTGCAGTGGCACTGGCTGGTTTCGCTACCGTAGCGCAGGCCGATGATGATCGCCGCCGCGGGGATGATGATCGCCGCCGCAAAAAAGGG
""".strip()
    
    with open("test_genome.fasta", "w") as f:
        f.write(test_fasta)
    
    return "test_genome.fasta"

def create_test_traits():
    """Create trait definitions for testing"""
    traits = [
        {
            "name": "carbon_metabolism",
            "description": "Carbon source utilization and metabolism",
            "associated_genes": ["crp"],
            "known_sequences": []
        },
        {
            "name": "stress_response",
            "description": "Response to environmental stress",
            "associated_genes": ["rpoS"],
            "known_sequences": []
        },
        {
            "name": "motility",
            "description": "Flagellar synthesis and chemotaxis",
            "associated_genes": ["flhD"],
            "known_sequences": []
        },
        {
            "name": "regulatory",
            "description": "Gene expression regulation",
            "associated_genes": ["crp", "fis", "rpoS", "flhD"],
            "known_sequences": []
        },
        {
            "name": "structural",
            "description": "Structural proteins and components",
            "associated_genes": ["ompA", "fis"],
            "known_sequences": []
        }
    ]
    
    with open("test_traits.json", "w") as f:
        json.dump(traits, f, indent=2)
    
    return "test_traits.json"

def run_rust_analysis(genome_file, traits_file):
    """Run the Rust analysis pipeline"""
    output_dir = "test_pipeline_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if Rust binary exists
    rust_binary = Path("rust_impl/target/release/pleiotropy_core")
    if not rust_binary.exists():
        rust_binary = Path("rust_impl/target/debug/pleiotropy_core")
    
    if not rust_binary.exists():
        print("WARNING: Rust binary not found. Please build with 'cargo build' first.")
        return None
    
    cmd = [
        str(rust_binary),
        "-i", genome_file,
        "-t", traits_file,
        "-o", output_dir,
        "-m", "2",  # Minimum 2 traits for pleiotropy
        "-v"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"Rust analysis exit code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        return output_dir
    except Exception as e:
        print(f"Error running Rust analysis: {e}")
        return None

def analyze_results(output_dir):
    """Analyze the results from the pipeline"""
    if not output_dir:
        return
    
    # Read analysis results
    results_file = Path(output_dir) / "analysis_results.json"
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        
        print("\n=== ANALYSIS RESULTS ===")
        print(f"Total sequences analyzed: {results['sequences']}")
        print(f"Total traits identified: {len(results['identified_traits'])}")
        
        if results['identified_traits']:
            print("\nIdentified traits by gene:")
            for trait in results['identified_traits'][:10]:  # Show first 10
                print(f"  Gene: {trait['gene_id']}")
                print(f"    Traits: {', '.join(trait['trait_names'])}")
                print(f"    Confidence: {trait['confidence_score']:.3f}")
    
    # Read pleiotropic genes
    pleiotropic_file = Path(output_dir) / "pleiotropic_genes.json"
    if pleiotropic_file.exists():
        with open(pleiotropic_file) as f:
            pleiotropic = json.load(f)
        
        print(f"\n=== PLEIOTROPIC GENES ===")
        print(f"Total pleiotropic genes found: {len(pleiotropic)}")
        
        if pleiotropic:
            print("\nTop pleiotropic genes:")
            for gene in pleiotropic[:5]:  # Show top 5
                print(f"  Gene: {gene['gene_id']}")
                print(f"    Traits: {', '.join(gene['traits'])}")
                print(f"    Confidence: {gene['confidence']:.3f}")
        else:
            print("  No pleiotropic genes detected!")
            print("\nDEBUG: This indicates the pipeline may still have issues.")
            print("Expected to find at least: crp, fis, rpoS as pleiotropic")

def main():
    """Main test function"""
    print("=== Testing Fixed Trait Detection Pipeline ===\n")
    
    # Create test data
    print("Creating test genome...")
    genome_file = create_test_genome()
    
    print("Creating trait definitions...")
    traits_file = create_test_traits()
    
    # Run analysis
    print("\nRunning Rust analysis pipeline...")
    output_dir = run_rust_analysis(genome_file, traits_file)
    
    # Analyze results
    analyze_results(output_dir)
    
    # Expected results
    print("\n=== EXPECTED RESULTS ===")
    print("Expected pleiotropic genes (≥2 traits):")
    print("  - crp: carbon_metabolism, regulatory")
    print("  - fis: regulatory, structural")
    print("  - rpoS: stress_response, regulatory")
    print("  - flhD: motility, regulatory")
    
    # Cleanup
    print("\nCleaning up test files...")
    for f in ["test_genome.fasta", "test_traits.json"]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    main()