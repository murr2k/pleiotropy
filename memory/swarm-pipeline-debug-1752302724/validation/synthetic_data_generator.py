#!/usr/bin/env python3
"""
Synthetic Data Generator for Pipeline Validation

Creates synthetic genomic data with known pleiotropic patterns for testing.
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


class SyntheticDataGenerator:
    """Generates synthetic genomic data with controlled pleiotropic patterns."""
    
    # Codon table for realistic sequence generation
    CODON_TABLE = {
        'A': ['GCT', 'GCC', 'GCA', 'GCG'],  # Alanine
        'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],  # Arginine
        'N': ['AAT', 'AAC'],  # Asparagine
        'D': ['GAT', 'GAC'],  # Aspartic acid
        'C': ['TGT', 'TGC'],  # Cysteine
        'Q': ['CAA', 'CAG'],  # Glutamine
        'E': ['GAA', 'GAG'],  # Glutamic acid
        'G': ['GGT', 'GGC', 'GGA', 'GGG'],  # Glycine
        'H': ['CAT', 'CAC'],  # Histidine
        'I': ['ATT', 'ATC', 'ATA'],  # Isoleucine
        'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],  # Leucine
        'K': ['AAA', 'AAG'],  # Lysine
        'M': ['ATG'],  # Methionine (start)
        'F': ['TTT', 'TTC'],  # Phenylalanine
        'P': ['CCT', 'CCC', 'CCA', 'CCG'],  # Proline
        'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],  # Serine
        'T': ['ACT', 'ACC', 'ACA', 'ACG'],  # Threonine
        'W': ['TGG'],  # Tryptophan
        'Y': ['TAT', 'TAC'],  # Tyrosine
        'V': ['GTT', 'GTC', 'GTA', 'GTG'],  # Valine
        '*': ['TAA', 'TAG', 'TGA']  # Stop codons
    }
    
    def __init__(self, output_dir: str = "synthetic_data"):
        """Initialize the synthetic data generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_data = []
        
    def generate_positive_control(self, gene_id: str, traits: List[str], 
                                 length: int = 600) -> Tuple[str, str]:
        """Generate a gene with strong pleiotropic signal for given traits."""
        # Define trait-specific codon biases
        trait_biases = {
            "metabolism": {
                'A': {'GCT': 0.7, 'GCC': 0.2, 'GCA': 0.05, 'GCG': 0.05},
                'L': {'CTG': 0.6, 'CTT': 0.2, 'CTC': 0.1, 'CTA': 0.05, 'TTA': 0.03, 'TTG': 0.02}
            },
            "stress_response": {
                'K': {'AAA': 0.8, 'AAG': 0.2},
                'E': {'GAA': 0.75, 'GAG': 0.25}
            },
            "regulatory": {
                'R': {'CGT': 0.5, 'CGC': 0.3, 'CGA': 0.1, 'CGG': 0.05, 'AGA': 0.03, 'AGG': 0.02},
                'H': {'CAT': 0.7, 'CAC': 0.3}
            },
            "dna_dynamics": {
                'F': {'TTT': 0.8, 'TTC': 0.2},
                'Y': {'TAT': 0.75, 'TAC': 0.25}
            },
            "motility": {
                'E': {'GAA': 0.85, 'GAG': 0.15},
                'D': {'GAT': 0.7, 'GAC': 0.3}
            },
            "biofilm_formation": {
                'Y': {'TAT': 0.8, 'TAC': 0.2},
                'W': {'TGG': 1.0}
            }
        }
        
        # Generate sequence with trait-specific patterns
        sequence = "ATG"  # Start codon
        
        while len(sequence) < length - 3:
            # Select trait to express
            trait = random.choice(traits)
            bias = trait_biases.get(trait, {})
            
            # Select amino acid biased for this trait
            if bias:
                aa = random.choice(list(bias.keys()))
                codon_options = self.CODON_TABLE[aa]
                
                # Apply trait-specific bias
                if aa in bias:
                    trait_specific_bias = bias[aa]
                    codons = list(trait_specific_bias.keys())
                    weights = list(trait_specific_bias.values())
                    codon = np.random.choice(codons, p=weights)
                else:
                    codon = random.choice(codon_options)
            else:
                # Random amino acid
                aa = random.choice(list(self.CODON_TABLE.keys())[:-1])  # Exclude stop
                codon = random.choice(self.CODON_TABLE[aa])
            
            sequence += codon
        
        # Add stop codon
        sequence += random.choice(self.CODON_TABLE['*'])
        
        # Ensure proper length
        sequence = sequence[:length]
        
        header = f">{gene_id} Positive control - traits: {','.join(traits)}"
        
        self.generated_data.append({
            "gene_id": gene_id,
            "type": "positive_control",
            "traits": traits,
            "length": len(sequence),
            "expected_detection": True
        })
        
        return header, sequence
    
    def generate_negative_control(self, gene_id: str, length: int = 450) -> Tuple[str, str]:
        """Generate a random gene with no pleiotropic signal."""
        sequence = "ATG"  # Start codon
        
        # Generate completely random sequence
        while len(sequence) < length - 3:
            # Random amino acid
            aa = random.choice(list(self.CODON_TABLE.keys())[:-1])
            codon = random.choice(self.CODON_TABLE[aa])
            sequence += codon
        
        # Add stop codon
        sequence += random.choice(self.CODON_TABLE['*'])
        sequence = sequence[:length]
        
        header = f">{gene_id} Negative control - no pleiotropy"
        
        self.generated_data.append({
            "gene_id": gene_id,
            "type": "negative_control",
            "traits": [],
            "length": len(sequence),
            "expected_detection": False
        })
        
        return header, sequence
    
    def generate_edge_case(self, gene_id: str, case_type: str) -> Tuple[str, str]:
        """Generate edge case sequences for testing."""
        if case_type == "very_short":
            sequence = "ATGTAA"  # Just start and stop
            header = f">{gene_id} Edge case - very short"
            expected = False
            
        elif case_type == "single_trait_weak":
            # Weak signal for single trait
            sequence = "ATG"
            trait = "metabolism"
            for _ in range(20):
                if random.random() < 0.3:  # 30% trait-specific
                    sequence += "GCT"  # Alanine bias
                else:
                    aa = random.choice(list(self.CODON_TABLE.keys())[:-1])
                    sequence += random.choice(self.CODON_TABLE[aa])
            sequence += "TAA"
            header = f">{gene_id} Edge case - weak single trait"
            expected = False
            
        elif case_type == "ambiguous_nucleotides":
            # Sequence with N's and other ambiguous codes
            sequence = "ATGNNNAAANNNGGGNNNTTTNNNCCCTAG"
            header = f">{gene_id} Edge case - ambiguous nucleotides"
            expected = False
            
        else:
            raise ValueError(f"Unknown edge case type: {case_type}")
        
        self.generated_data.append({
            "gene_id": gene_id,
            "type": f"edge_case_{case_type}",
            "traits": [],
            "length": len(sequence),
            "expected_detection": expected
        })
        
        return header, sequence
    
    def generate_regulatory_context_gene(self, gene_id: str, traits: List[str]) -> Tuple[str, str]:
        """Generate gene with regulatory elements."""
        # Add promoter elements
        promoter = "TTGACA" + "N" * 17 + "TATAAT" + "N" * 10
        promoter = promoter.replace("N", "A")  # Simple replacement
        
        # Generate gene with pleiotropy
        _, gene_seq = self.generate_positive_control(f"{gene_id}_body", traits, length=500)
        
        # Add enhancer
        enhancer = "CGTACGTA" * 3
        
        full_sequence = promoter + gene_seq + enhancer
        header = f">{gene_id} With regulatory context - traits: {','.join(traits)}"
        
        self.generated_data.append({
            "gene_id": gene_id,
            "type": "with_regulatory",
            "traits": traits,
            "length": len(full_sequence),
            "expected_detection": True,
            "regulatory_elements": ["promoter", "enhancer"]
        })
        
        return header, full_sequence
    
    def create_comprehensive_test_set(self) -> Dict[str, str]:
        """Create a comprehensive synthetic test dataset."""
        print("🧬 Generating comprehensive synthetic test set...")
        
        genome_file = self.output_dir / "synthetic_test_genome.fasta"
        sequences = []
        
        # 1. Strong positive controls (should definitely be detected)
        sequences.append(self.generate_positive_control(
            "syn_strong_dual_1", 
            ["metabolism", "stress_response"], 
            length=600
        ))
        
        sequences.append(self.generate_positive_control(
            "syn_strong_triple_1",
            ["regulatory", "dna_dynamics", "motility"],
            length=900
        ))
        
        sequences.append(self.generate_positive_control(
            "syn_strong_dual_2",
            ["biofilm_formation", "stress_response"],
            length=750
        ))
        
        # 2. Negative controls (should not be detected)
        sequences.append(self.generate_negative_control("syn_negative_1", 450))
        sequences.append(self.generate_negative_control("syn_negative_2", 300))
        
        # 3. Edge cases
        sequences.append(self.generate_edge_case("syn_edge_short", "very_short"))
        sequences.append(self.generate_edge_case("syn_edge_weak", "single_trait_weak"))
        
        # 4. Regulatory context
        sequences.append(self.generate_regulatory_context_gene(
            "syn_regulatory_1",
            ["metabolism", "regulatory"]
        ))
        
        # Write genome file
        with open(genome_file, 'w') as f:
            for header, seq in sequences:
                f.write(f"{header}\n{seq}\n")
        
        # Create traits configuration
        traits_config = [
            {
                "name": "metabolism",
                "description": "Metabolic processes and energy production",
                "associated_genes": ["syn_strong_dual_1", "syn_regulatory_1"],
                "known_sequences": []
            },
            {
                "name": "stress_response",
                "description": "Stress adaptation mechanisms",
                "associated_genes": ["syn_strong_dual_1", "syn_strong_dual_2"],
                "known_sequences": []
            },
            {
                "name": "regulatory",
                "description": "Gene expression regulation",
                "associated_genes": ["syn_strong_triple_1", "syn_regulatory_1"],
                "known_sequences": []
            },
            {
                "name": "dna_dynamics",
                "description": "DNA topology and organization",
                "associated_genes": ["syn_strong_triple_1"],
                "known_sequences": []
            },
            {
                "name": "motility",
                "description": "Cell movement and chemotaxis",
                "associated_genes": ["syn_strong_triple_1"],
                "known_sequences": []
            },
            {
                "name": "biofilm_formation",
                "description": "Biofilm development",
                "associated_genes": ["syn_strong_dual_2"],
                "known_sequences": []
            }
        ]
        
        traits_file = self.output_dir / "synthetic_traits.json"
        with open(traits_file, 'w') as f:
            json.dump(traits_config, f, indent=2)
        
        # Save metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "generator": "SyntheticDataGenerator v1.0",
            "total_sequences": len(sequences),
            "positive_controls": sum(1 for d in self.generated_data if d["expected_detection"]),
            "negative_controls": sum(1 for d in self.generated_data if not d["expected_detection"]),
            "sequences": self.generated_data
        }
        
        metadata_file = self.output_dir / "synthetic_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Generated {len(sequences)} synthetic sequences")
        print(f"   Positive controls: {metadata['positive_controls']}")
        print(f"   Negative controls: {metadata['negative_controls']}")
        print(f"   Files saved to: {self.output_dir}")
        
        return {
            "genome_file": str(genome_file),
            "traits_file": str(traits_file),
            "metadata_file": str(metadata_file),
            "metadata": metadata
        }
    
    def generate_performance_test_set(self, sizes: List[int] = None) -> Dict[str, str]:
        """Generate datasets of varying sizes for performance testing."""
        if sizes is None:
            sizes = [10, 50, 100, 500, 1000]
        
        print("📊 Generating performance test datasets...")
        
        perf_dir = self.output_dir / "performance"
        perf_dir.mkdir(exist_ok=True)
        
        results = {}
        
        for size in sizes:
            genome_file = perf_dir / f"genome_{size}_genes.fasta"
            
            with open(genome_file, 'w') as f:
                for i in range(size):
                    # Mix of positive and negative controls
                    if i % 3 == 0:
                        header, seq = self.generate_positive_control(
                            f"perf_gene_{i}",
                            random.sample(["metabolism", "stress_response", "regulatory"], 2)
                        )
                    else:
                        header, seq = self.generate_negative_control(f"perf_gene_{i}")
                    
                    f.write(f"{header}\n{seq}\n")
            
            results[f"size_{size}"] = str(genome_file)
        
        print(f"✅ Generated {len(sizes)} performance test datasets")
        return results


def main():
    """Generate all synthetic test data."""
    generator = SyntheticDataGenerator(
        output_dir="/home/murr2k/projects/agentic/pleiotropy/memory/swarm-pipeline-debug-1752302724/validation/synthetic_data"
    )
    
    # Generate comprehensive test set
    test_data = generator.create_comprehensive_test_set()
    
    # Generate performance test sets
    perf_data = generator.generate_performance_test_set()
    
    # Save summary
    summary = {
        "test_data": test_data,
        "performance_data": perf_data,
        "timestamp": datetime.now().isoformat()
    }
    
    summary_file = generator.output_dir / "generation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ All synthetic data generated successfully!")
    print(f"📁 Output directory: {generator.output_dir}")
    
    return summary


if __name__ == "__main__":
    main()