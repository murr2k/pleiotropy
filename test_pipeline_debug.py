#!/usr/bin/env python3
"""Test the pipeline to understand why it's not detecting traits"""

import json
import subprocess
import os

def test_pipeline():
    print("Testing Genomic Pleiotropy Pipeline...")
    
    # Run the pipeline
    cmd = [
        "./rust_impl/target/release/genomic_cryptanalysis",
        "--input", "test_synthetic.fasta",
        "--traits", "test_synthetic_traits.json",
        "--output", "debug_output",
        "--min-traits", "1",  # Lower threshold
        "--verbose"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("\nSTDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)
    
    # Check the results
    if os.path.exists("debug_output/analysis_results.json"):
        with open("debug_output/analysis_results.json", "r") as f:
            results = json.load(f)
        
        print("\n=== ANALYSIS RESULTS ===")
        print(f"Sequences analyzed: {results['sequences']}")
        print(f"Traits identified: {results['identified_traits']}")
        
        # Check trait-specific frequencies
        print("\n=== TRAIT-SPECIFIC FREQUENCIES ===")
        for codon_data in results['frequency_table']['codon_frequencies'][:5]:
            codon = codon_data['codon']
            print(f"\nCodon {codon} ({codon_data['amino_acid']}):")
            print(f"  Global frequency: {codon_data['global_frequency']:.3f}")
            for trait, freq in codon_data['trait_specific_frequency'].items():
                if freq > 0:
                    print(f"  {trait}: {freq:.3f}")
    
    # Check if we need to look at intermediate files
    print("\n=== CHECKING FOR INTERMEDIATE FILES ===")
    for root, dirs, files in os.walk("debug_output"):
        for file in files:
            filepath = os.path.join(root, file)
            print(f"Found: {filepath}")
            if file.endswith(".json"):
                with open(filepath, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and len(str(data)) < 500:
                    print(f"  Content: {json.dumps(data, indent=2)}")

if __name__ == "__main__":
    test_pipeline()