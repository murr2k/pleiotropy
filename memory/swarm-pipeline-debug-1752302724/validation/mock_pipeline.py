#!/usr/bin/env python3
"""
Mock Pipeline for Validation Testing

Simulates the pipeline behavior with controllable results for testing the validation framework.
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib


class MockPipeline:
    """Mock implementation of the pleiotropy detection pipeline."""
    
    def __init__(self):
        """Initialize mock pipeline."""
        self.known_positive_patterns = {
            "syn_strong_dual_1": {
                "traits": ["metabolism", "stress_response"],
                "confidence": 0.92
            },
            "syn_strong_triple_1": {
                "traits": ["regulatory", "dna_dynamics", "motility"],
                "confidence": 0.88
            },
            "syn_strong_dual_2": {
                "traits": ["biofilm_formation", "stress_response"],
                "confidence": 0.85
            },
            "syn_regulatory_1": {
                "traits": ["metabolism", "regulatory"],
                "confidence": 0.83
            },
            "synthetic_gene_1": {
                "traits": ["metabolism", "stress_response"],
                "confidence": 0.90
            },
            "synthetic_gene_2": {
                "traits": ["regulatory", "dna_dynamics", "motility"],
                "confidence": 0.87
            },
            "synthetic_gene_4": {
                "traits": ["biofilm_formation", "cell_envelope"],
                "confidence": 0.84
            }
        }
        
        # Known E. coli genes
        self.ecoli_genes = {
            "crp": {
                "traits": ["carbon_metabolism", "regulatory"],
                "confidence": 0.95
            },
            "fis": {
                "traits": ["dna_dynamics", "regulatory"],
                "confidence": 0.93
            },
            "rpoS": {
                "traits": ["stress_response", "regulatory"],
                "confidence": 0.91
            },
            "hns": {
                "traits": ["dna_dynamics", "regulatory"],
                "confidence": 0.89
            }
        }
    
    def run(self, genome_file: str, traits_file: str, output_dir: str,
            min_traits: int = 2, confidence_threshold: Optional[float] = None) -> int:
        """Simulate pipeline execution."""
        start_time = time.time()
        
        # Parse input files
        sequences = self._parse_fasta(genome_file)
        traits = self._parse_traits(traits_file)
        
        # Default confidence threshold
        if confidence_threshold is None:
            confidence_threshold = 0.75
        
        # Detect pleiotropic genes
        pleiotropic_genes = []
        analysis_results = {
            "sequences": len(sequences),
            "traits_analyzed": len(traits),
            "min_traits_threshold": min_traits,
            "confidence_threshold": confidence_threshold,
            "frequency_table": self._generate_mock_frequency_table()
        }
        
        # Process each sequence
        for seq_id, sequence in sequences.items():
            # Check if it's a known positive
            detected_traits = []
            confidence = 0.0
            
            # Check synthetic patterns
            if seq_id in self.known_positive_patterns:
                pattern = self.known_positive_patterns[seq_id]
                detected_traits = pattern["traits"]
                confidence = pattern["confidence"]
            
            # Check E. coli genes
            elif any(known_gene in seq_id for known_gene in self.ecoli_genes):
                for gene_name, gene_data in self.ecoli_genes.items():
                    if gene_name in seq_id:
                        detected_traits = gene_data["traits"]
                        confidence = gene_data["confidence"]
                        break
            
            # Random chance for false positives (5%)
            elif random.random() < 0.05:
                detected_traits = random.sample([t["name"] for t in traits], 2)
                confidence = random.uniform(0.70, 0.80)
            
            # Add to results if meets criteria
            if len(detected_traits) >= min_traits and confidence >= confidence_threshold:
                pleiotropic_genes.append({
                    "gene_id": seq_id,
                    "traits": detected_traits,
                    "confidence": confidence,
                    "sequence_length": len(sequence),
                    "codon_bias_score": random.uniform(0.6, 0.9),
                    "regulatory_score": random.uniform(0.5, 0.8)
                })
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save analysis results
        with open(output_path / "analysis_results.json", 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Save pleiotropic genes
        with open(output_path / "pleiotropic_genes.json", 'w') as f:
            json.dump(pleiotropic_genes, f, indent=2)
        
        # Save summary report
        summary = f"""# Pleiotropy Analysis Summary

## Overview
- Sequences analyzed: {len(sequences)}
- Pleiotropic genes found: {len(pleiotropic_genes)}
- Confidence threshold: {confidence_threshold}
- Minimum traits: {min_traits}

## Results
- Detection rate: {len(pleiotropic_genes) / len(sequences) * 100:.1f}%
- Average confidence: {sum(g['confidence'] for g in pleiotropic_genes) / len(pleiotropic_genes):.3f}
- Execution time: {time.time() - start_time:.2f}s

## Top Predictions
"""
        for gene in sorted(pleiotropic_genes, key=lambda x: x['confidence'], reverse=True)[:5]:
            summary += f"- {gene['gene_id']}: {', '.join(gene['traits'])} (confidence: {gene['confidence']:.3f})\n"
        
        with open(output_path / "summary_report.md", 'w') as f:
            f.write(summary)
        
        return 0  # Success
    
    def _parse_fasta(self, fasta_file: str) -> Dict[str, str]:
        """Parse FASTA file and return sequences."""
        sequences = {}
        current_id = None
        current_seq = []
        
        try:
            with open(fasta_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_id:
                            sequences[current_id] = ''.join(current_seq)
                        current_id = line[1:].split()[0]
                        current_seq = []
                    else:
                        current_seq.append(line)
                
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
        except Exception as e:
            print(f"Error parsing FASTA: {e}")
        
        return sequences
    
    def _parse_traits(self, traits_file: str) -> List[Dict[str, Any]]:
        """Parse traits configuration file."""
        try:
            with open(traits_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error parsing traits: {e}")
            return []
    
    def _generate_mock_frequency_table(self) -> Dict[str, Any]:
        """Generate mock codon frequency table."""
        codons = ['TTT', 'TTC', 'TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG',
                  'ATT', 'ATC', 'ATA', 'ATG', 'GTT', 'GTC', 'GTA', 'GTG',
                  'TCT', 'TCC', 'TCA', 'TCG', 'CCT', 'CCC', 'CCA', 'CCG',
                  'ACT', 'ACC', 'ACA', 'ACG', 'GCT', 'GCC', 'GCA', 'GCG',
                  'TAT', 'TAC', 'TAA', 'TAG', 'CAT', 'CAC', 'CAA', 'CAG',
                  'AAT', 'AAC', 'AAA', 'AAG', 'GAT', 'GAC', 'GAA', 'GAG',
                  'TGT', 'TGC', 'TGA', 'TGG', 'CGT', 'CGC', 'CGA', 'CGG',
                  'AGT', 'AGC', 'AGA', 'AGG', 'GGT', 'GGC', 'GGA', 'GGG']
        
        total_codons = random.randint(10000, 50000)
        codon_frequencies = []
        
        for codon in codons:
            count = random.randint(10, 1000)
            freq = count / total_codons
            codon_frequencies.append({
                "codon": codon,
                "count": count,
                "global_frequency": freq,
                "trait_frequencies": {
                    "metabolism": freq * random.uniform(0.8, 1.2),
                    "stress_response": freq * random.uniform(0.8, 1.2),
                    "regulatory": freq * random.uniform(0.8, 1.2)
                }
            })
        
        return {
            "total_codons": total_codons,
            "codon_frequencies": codon_frequencies
        }


def main():
    """Run mock pipeline with test arguments."""
    import sys
    
    if len(sys.argv) < 7:
        print("Usage: mock_pipeline.py --input <genome> --traits <traits> --output <dir> [--min-traits N] [--confidence-threshold X]")
        return 1
    
    # Parse arguments
    args = {}
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--input':
            args['genome_file'] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--traits':
            args['traits_file'] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--output':
            args['output_dir'] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--min-traits':
            args['min_traits'] = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--confidence-threshold':
            args['confidence_threshold'] = float(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    # Run pipeline
    pipeline = MockPipeline()
    return pipeline.run(**args)


if __name__ == "__main__":
    import sys
    sys.exit(main())