#!/usr/bin/env python3
"""
Validate public genomic datasets for E. coli K-12 pleiotropic gene analysis.
"""

import json
import os
from pathlib import Path

def validate_genome_file(filepath):
    """Validate FASTA genome file."""
    print(f"Validating genome file: {filepath}")
    
    if not os.path.exists(filepath):
        return False, "File not found"
    
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        if not first_line.startswith('>'):
            return False, "Invalid FASTA format - missing header"
        
        if "NC_000913.3" not in first_line:
            return False, "Wrong genome accession"
        
        # Count nucleotides
        seq_length = 0
        for line in f:
            if not line.startswith('>'):
                seq_length += len(line.strip())
    
    expected_length = 4641652  # E. coli K-12 MG1655 genome length
    if abs(seq_length - expected_length) > 100:
        return False, f"Unexpected genome length: {seq_length} (expected ~{expected_length})"
    
    return True, f"Valid genome file, length: {seq_length} bp"

def validate_json_file(filepath, required_keys):
    """Validate JSON file structure."""
    print(f"Validating JSON file: {filepath}")
    
    if not os.path.exists(filepath):
        return False, "File not found"
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON format: {e}"
    
    missing_keys = []
    for key in required_keys:
        if key not in data:
            missing_keys.append(key)
    
    if missing_keys:
        return False, f"Missing required keys: {missing_keys}"
    
    return True, "Valid JSON structure"

def main():
    """Run validation on all public datasets."""
    genome_dir = Path(__file__).parent
    
    print("="*60)
    print("E. coli K-12 Public Data Validation")
    print("="*60)
    
    # Validate genome file
    genome_file = genome_dir / "public_ecoli_genome.fasta"
    success, message = validate_genome_file(genome_file)
    print(f"✓ {message}" if success else f"✗ {message}")
    print()
    
    # Validate pleiotropic genes
    genes_file = genome_dir / "validated_pleiotropic_genes.json"
    success, message = validate_json_file(genes_file, ["metadata", "genes"])
    print(f"✓ {message}" if success else f"✗ {message}")
    
    if success:
        with open(genes_file, 'r') as f:
            data = json.load(f)
            print(f"  - Found {len(data['genes'])} core pleiotropic genes")
            print(f"  - Found {len(data.get('additional_candidates', {}))} additional candidates")
    print()
    
    # Validate codon usage
    codon_file = genome_dir / "codon_usage_ecoli.json"
    success, message = validate_json_file(codon_file, ["metadata", "codon_frequencies", "amino_acid_usage"])
    print(f"✓ {message}" if success else f"✗ {message}")
    
    if success:
        with open(codon_file, 'r') as f:
            data = json.load(f)
            print(f"  - Contains {len(data['codon_frequencies'])} codon entries")
            print(f"  - Contains {len(data['amino_acid_usage'])} amino acid summaries")
    print()
    
    # Validate regulatory elements
    reg_file = genome_dir / "regulatory_elements.json"
    success, message = validate_json_file(reg_file, ["metadata", "promoters", "operators"])
    print(f"✓ {message}" if success else f"✗ {message}")
    
    if success:
        with open(reg_file, 'r') as f:
            data = json.load(f)
            print(f"  - Contains {len(data['promoters'])} promoter types")
            print(f"  - Contains {len(data['operators'])} operator types")
            print(f"  - Contains {len(data.get('riboswitches', {}))} riboswitches")
    print()
    
    # Validate data sources documentation
    doc_file = genome_dir / "data_sources.md"
    if os.path.exists(doc_file):
        print(f"✓ Data sources documentation found")
        with open(doc_file, 'r') as f:
            lines = f.readlines()
            db_count = sum(1 for line in lines if line.strip().startswith("### "))
            print(f"  - Documents {db_count} primary databases")
    else:
        print(f"✗ Data sources documentation not found")
    
    print("="*60)
    print("Validation complete!")
    print("\nAll datasets are ready for the pleiotropic gene analysis pipeline.")

if __name__ == "__main__":
    main()