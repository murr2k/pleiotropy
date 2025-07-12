"""
Test Data Generator for Genomic Pleiotropy Cryptanalysis

Generates realistic test data for all components including:
- Genomic sequences with known traits
- Trial database entries
- Frequency tables
- Gene-trait associations
"""

import json
import random
import string
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
from faker import Faker

fake = Faker()


class TestDataGeneratorUtility:
    """Generate test data for the genomic pleiotropy project."""
    
    # Genetic code
    CODONS = None
    REGULATORY_MOTIFS = {
        'promoter': ['TATAAT', 'TTGACA', 'CAAT', 'GCGC'],
        'enhancer': ['GGAGG', 'CCACC', 'GGGCGG'],
        'silencer': ['ATAAA', 'TTTTT', 'CCAAT'],
        'terminator': ['TTTATT', 'AAATAA']
    }
    
    # Common traits in bacteria
    TRAIT_CATEGORIES = {
        'metabolism': ['glucose_utilization', 'lactose_metabolism', 'amino_acid_synthesis'],
        'stress': ['heat_shock', 'oxidative_stress', 'osmotic_stress'],
        'motility': ['flagellar_assembly', 'chemotaxis', 'pili_formation'],
        'virulence': ['toxin_production', 'adhesion', 'invasion'],
        'regulation': ['quorum_sensing', 'two_component_system', 'sigma_factors']
    }
    
    @classmethod
    def initialize(cls, seed: int = 42):
        """Initialize generator with seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)
        cls.CODONS = cls._generate_codon_table()
    
    @staticmethod
    def _generate_codon_table() -> Dict[str, str]:
        """Generate standard genetic code codon table."""
        bases = ['A', 'T', 'G', 'C']
        codons = {}
        amino_acids = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
        
        i = 0
        for first in bases:
            for second in bases:
                for third in bases:
                    codon = first + second + third
                    codons[codon] = amino_acids[i]
                    i += 1
        
        return codons
    
    def generate_sequence(self, 
                         length: int = 1000,
                         trait_bias: Optional[Dict[str, float]] = None,
                         add_regulatory: bool = True) -> str:
        """
        Generate a DNA sequence with optional trait bias.
        
        Args:
            length: Length of sequence (will be adjusted to multiple of 3)
            trait_bias: Dictionary of trait names to bias strength (0-1)
            add_regulatory: Whether to add regulatory elements
            
        Returns:
            DNA sequence string
        """
        # Adjust length to multiple of 3
        length = (length // 3) * 3
        
        # Base codon frequencies
        codon_probs = {codon: 1.0 for codon in self.codons.keys()}
        
        # Apply trait biases
        if trait_bias:
            for trait, strength in trait_bias.items():
                # Bias certain codons based on trait
                if 'expression' in trait or 'growth' in trait:
                    # Prefer optimal codons
                    for codon in ['CTG', 'CGT', 'AAA', 'GAA']:
                        if codon in codon_probs:
                            codon_probs[codon] *= (1 + strength)
                            
                elif 'stress' in trait:
                    # Prefer rare codons
                    for codon in ['CTA', 'AGA', 'AGG', 'CCC']:
                        if codon in codon_probs:
                            codon_probs[codon] *= (1 + strength)
        
        # Normalize probabilities
        total_prob = sum(codon_probs.values())
        codon_probs = {k: v/total_prob for k, v in codon_probs.items()}
        
        # Generate sequence
        codons_list = list(codon_probs.keys())
        probs_list = list(codon_probs.values())
        
        sequence_codons = np.random.choice(
            codons_list, 
            size=length // 3, 
            p=probs_list
        )
        
        sequence = ''.join(sequence_codons)
        
        # Add regulatory elements
        if add_regulatory and length > 100:
            sequence = self._add_regulatory_elements(sequence)
        
        return sequence
    
    def _add_regulatory_elements(self, sequence: str) -> str:
        """Add regulatory elements to sequence."""
        seq_list = list(sequence)
        
        # Add promoter at beginning
        if len(sequence) > 20:
            promoter = random.choice(self.regulatory_motifs['promoter'])
            for i, base in enumerate(promoter):
                seq_list[i] = base
        
        # Add random enhancers/silencers
        if len(sequence) > 100:
            # Add 1-3 regulatory elements
            for _ in range(random.randint(1, 3)):
                motif_type = random.choice(['enhancer', 'silencer'])
                motif = random.choice(self.regulatory_motifs[motif_type])
                
                # Find random position (avoiding start/end)
                pos = random.randint(30, len(sequence) - len(motif) - 30)
                for i, base in enumerate(motif):
                    seq_list[pos + i] = base
        
        return ''.join(seq_list)
    
    def generate_gene(self,
                     gene_id: str,
                     traits: List[str],
                     length: int = 900) -> Dict:
        """
        Generate a complete gene with metadata.
        
        Args:
            gene_id: Gene identifier
            traits: List of traits associated with this gene
            length: Gene length in base pairs
            
        Returns:
            Dictionary with gene information
        """
        # Calculate trait biases
        trait_bias = {trait: random.uniform(0.3, 0.8) for trait in traits}
        
        # Generate sequence
        sequence = self.generate_sequence(length, trait_bias)
        
        # Generate annotations
        annotations = {
            'start': 0,
            'end': length,
            'strand': random.choice(['+', '-']),
            'type': 'CDS',
            'product': f"{gene_id} protein",
            'traits': traits,
            'confidence': {trait: random.uniform(0.6, 0.95) for trait in traits}
        }
        
        return {
            'id': gene_id,
            'sequence': sequence,
            'annotations': annotations,
            'length': length
        }
    
    def generate_genome(self,
                       n_genes: int = 100,
                       avg_traits_per_gene: float = 2.5) -> List[Dict]:
        """
        Generate a complete genome with multiple genes.
        
        Args:
            n_genes: Number of genes to generate
            avg_traits_per_gene: Average number of traits per gene
            
        Returns:
            List of gene dictionaries
        """
        genome = []
        all_traits = [trait for traits in self.trait_categories.values() 
                     for trait in traits]
        
        for i in range(n_genes):
            # Select traits for this gene
            n_traits = np.random.poisson(avg_traits_per_gene - 1) + 1
            n_traits = min(n_traits, len(all_traits))
            
            gene_traits = random.sample(all_traits, n_traits)
            
            # Generate gene
            gene = self.generate_gene(
                gene_id=f"gene_{i:04d}",
                traits=gene_traits,
                length=random.randint(300, 3000)
            )
            
            genome.append(gene)
        
        return genome
    
    def generate_frequency_table(self, genome: List[Dict]) -> Dict:
        """
        Generate frequency table from genome.
        
        Args:
            genome: List of genes
            
        Returns:
            Frequency table dictionary
        """
        # Count codons globally and per trait
        global_counts = {}
        trait_counts = {}
        
        for gene in genome:
            sequence = gene['sequence']
            traits = gene['annotations']['traits']
            
            # Count codons in this gene
            for i in range(0, len(sequence) - 2, 3):
                codon = sequence[i:i+3]
                if len(codon) == 3 and all(b in 'ATCG' for b in codon):
                    # Global count
                    global_counts[codon] = global_counts.get(codon, 0) + 1
                    
                    # Trait-specific counts
                    for trait in traits:
                        if trait not in trait_counts:
                            trait_counts[trait] = {}
                        trait_counts[trait][codon] = trait_counts[trait].get(codon, 0) + 1
        
        # Calculate frequencies
        total_global = sum(global_counts.values())
        codon_frequencies = []
        
        for codon in self.codons.keys():
            freq_entry = {
                'codon': codon,
                'amino_acid': self.codons[codon],
                'global_frequency': global_counts.get(codon, 0) / max(total_global, 1),
                'trait_frequencies': {}
            }
            
            # Add trait-specific frequencies
            for trait, counts in trait_counts.items():
                total_trait = sum(counts.values())
                freq_entry['trait_frequencies'][trait] = counts.get(codon, 0) / max(total_trait, 1)
            
            codon_frequencies.append(freq_entry)
        
        # Create trait definitions
        trait_definitions = []
        for category, traits in self.trait_categories.items():
            for trait in traits:
                if trait in trait_counts:  # Only include traits present in genome
                    trait_definitions.append({
                        'name': trait,
                        'description': f"{trait.replace('_', ' ').title()} trait",
                        'category': category,
                        'known_genes': [g['id'] for g in genome 
                                      if trait in g['annotations']['traits']][:5]
                    })
        
        return {
            'codon_frequencies': codon_frequencies,
            'trait_definitions': trait_definitions,
            'total_codons_analyzed': total_global,
            'generation_date': datetime.now().isoformat()
        }
    
    def generate_trial_data(self,
                           n_trials: int = 100,
                           genome: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Generate trial database entries.
        
        Args:
            n_trials: Number of trials to generate
            genome: Optional genome to use for realistic data
            
        Returns:
            List of trial dictionaries
        """
        if genome is None:
            genome = self.generate_genome(20)  # Small genome for trials
        
        trials = []
        organisms = ['E. coli K-12', 'E. coli DH5α', 'E. coli BL21', 
                    'Salmonella enterica', 'Klebsiella pneumoniae']
        
        for i in range(n_trials):
            # Select random gene
            gene = random.choice(genome)
            
            # Generate trial data
            trial = {
                'id': f"TRIAL_{i:06d}",
                'organism': random.choice(organisms),
                'gene_id': gene['id'],
                'experiment_date': fake.date_between(
                    start_date='-2y', 
                    end_date='today'
                ).isoformat(),
                'researcher': fake.name(),
                'institution': fake.company(),
                'traits_tested': random.sample(
                    gene['annotations']['traits'],
                    k=random.randint(1, len(gene['annotations']['traits']))
                ),
                'traits_confirmed': [],
                'method': random.choice([
                    'RNA-seq', 'ChIP-seq', 'CRISPR screen', 
                    'Knockout mutation', 'Overexpression'
                ]),
                'confidence_scores': {},
                'raw_data': {
                    'sequence_quality': random.uniform(0.8, 1.0),
                    'coverage': random.uniform(10, 100),
                    'replicates': random.randint(3, 6)
                },
                'notes': fake.paragraph(nb_sentences=2),
                'status': random.choice(['completed', 'in_progress', 'validated']),
                'created_at': fake.date_time_between(
                    start_date='-1y',
                    end_date='now'
                ).isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # Randomly confirm some traits
            for trait in trial['traits_tested']:
                if random.random() > 0.3:  # 70% confirmation rate
                    trial['traits_confirmed'].append(trait)
                    trial['confidence_scores'][trait] = random.uniform(0.7, 0.99)
            
            trials.append(trial)
        
        return trials
    
    def generate_performance_test_data(self,
                                     n_sequences: int = 1000,
                                     sequence_length: int = 1000) -> Dict:
        """
        Generate large dataset for performance testing.
        
        Args:
            n_sequences: Number of sequences
            sequence_length: Length of each sequence
            
        Returns:
            Dictionary with test data
        """
        sequences = []
        for i in range(n_sequences):
            if i % 100 == 0:
                print(f"Generating sequence {i}/{n_sequences}")
            
            # Random traits
            n_traits = random.randint(1, 5)
            traits = random.sample(
                [t for ts in self.trait_categories.values() for t in ts],
                n_traits
            )
            
            # Generate sequence with traits
            trait_bias = {trait: random.uniform(0.1, 0.5) for trait in traits}
            sequence = self.generate_sequence(sequence_length, trait_bias)
            
            sequences.append({
                'id': f'perf_test_{i:06d}',
                'sequence': sequence,
                'expected_traits': traits
            })
        
        return {
            'sequences': sequences,
            'metadata': {
                'n_sequences': n_sequences,
                'sequence_length': sequence_length,
                'total_size_mb': (n_sequences * sequence_length) / (1024 * 1024),
                'generated_at': datetime.now().isoformat()
            }
        }
    
    def save_test_data(self, output_dir: str = './test_data'):
        """Save various test datasets to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate small test genome
        print("Generating test genome...")
        test_genome = self.generate_genome(50, avg_traits_per_gene=2.5)
        with open(f'{output_dir}/test_genome.json', 'w') as f:
            json.dump(test_genome, f, indent=2)
        
        # Generate frequency table
        print("Generating frequency table...")
        freq_table = self.generate_frequency_table(test_genome)
        with open(f'{output_dir}/frequency_table.json', 'w') as f:
            json.dump(freq_table, f, indent=2)
        
        # Generate trial data
        print("Generating trial data...")
        trials = self.generate_trial_data(200, test_genome)
        with open(f'{output_dir}/trial_data.json', 'w') as f:
            json.dump(trials, f, indent=2)
        
        # Generate FASTA file
        print("Generating FASTA file...")
        with open(f'{output_dir}/test_sequences.fasta', 'w') as f:
            for gene in test_genome[:20]:  # First 20 genes
                f.write(f">{gene['id']} {' '.join(gene['annotations']['traits'])}\n")
                # Write sequence in 80-character lines
                seq = gene['sequence']
                for i in range(0, len(seq), 80):
                    f.write(seq[i:i+80] + '\n')
        
        print(f"Test data saved to {output_dir}/")


class MockDataGenerator:
    """Generate mock data for unit tests."""
    
    @staticmethod
    def mock_statistical_data(n_samples: int = 100, n_traits: int = 5) -> Dict:
        """Generate mock data for statistical tests."""
        np.random.seed(42)
        
        # Create correlated trait data
        base_data = np.random.randn(n_samples, n_traits)
        
        # Add correlations
        correlation_matrix = np.eye(n_traits)
        correlation_matrix[0, 1] = correlation_matrix[1, 0] = 0.7
        correlation_matrix[2, 3] = correlation_matrix[3, 2] = -0.5
        
        # Apply correlations
        L = np.linalg.cholesky(correlation_matrix)
        correlated_data = base_data @ L.T
        
        trait_names = [f'trait_{i}' for i in range(n_traits)]
        trait_df = pd.DataFrame(correlated_data, columns=trait_names)
        
        # Gene expression data
        gene_expression = pd.Series(
            np.random.lognormal(2, 0.5, n_samples),
            name='gene_expression'
        )
        
        # Gene-trait associations
        gene_trait_map = {
            f'gene_{i}': random.sample(trait_names, k=random.randint(1, 3))
            for i in range(20)
        }
        
        return {
            'trait_data': trait_df,
            'gene_expression': gene_expression,
            'gene_trait_associations': gene_trait_map
        }


if __name__ == '__main__':
    # Generate test data
    generator = TestDataGenerator()
    generator.save_test_data()
    
    # Example: Generate specific test case
    test_gene = generator.generate_gene(
        'test_ftsZ',
        ['growth_rate', 'cell_division'],
        length=1200
    )
    print(f"Generated test gene: {test_gene['id']}")
    print(f"Traits: {test_gene['annotations']['traits']}")
    print(f"Sequence length: {len(test_gene['sequence'])}")