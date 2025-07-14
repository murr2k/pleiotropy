#!/usr/bin/env python3
"""
Genome Utilities for the 20 Genomes Experiment
Provides validation, trait definition generation, and metadata management
"""

import os
import json
import hashlib
from typing import Dict, List, Optional
from collections import defaultdict

class GenomeValidator:
    """Validate downloaded genome files"""
    
    @staticmethod
    def validate_fasta(filepath: str) -> Dict[str, any]:
        """Validate FASTA file format and return statistics"""
        stats = {
            'valid': False,
            'num_sequences': 0,
            'total_length': 0,
            'gc_content': 0.0,
            'errors': [],
            'checksum': ''
        }
        
        try:
            # Calculate file checksum
            with open(filepath, 'rb') as f:
                stats['checksum'] = hashlib.md5(f.read()).hexdigest()
            
            # Read and validate FASTA
            sequences = []
            current_seq = ''
            header_count = 0
            
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    if line.startswith('>'):
                        if current_seq:
                            sequences.append(current_seq)
                        current_seq = ''
                        header_count += 1
                    else:
                        # Validate sequence characters
                        valid_chars = set('ACGTNacgtn')
                        invalid = set(line) - valid_chars
                        if invalid:
                            stats['errors'].append(f"Line {line_num}: Invalid characters {invalid}")
                        current_seq += line.upper()
                
                if current_seq:
                    sequences.append(current_seq)
            
            if header_count == 0:
                stats['errors'].append("No FASTA headers found")
                return stats
            
            # Calculate statistics
            stats['num_sequences'] = len(sequences)
            total_seq = ''.join(sequences)
            stats['total_length'] = len(total_seq)
            
            # Calculate GC content
            gc_count = total_seq.count('G') + total_seq.count('C')
            if stats['total_length'] > 0:
                stats['gc_content'] = (gc_count / stats['total_length']) * 100
            
            stats['valid'] = len(stats['errors']) == 0
            
        except Exception as e:
            stats['errors'].append(f"Error reading file: {str(e)}")
        
        return stats


class TraitDefinitionGenerator:
    """Generate trait definitions based on organism characteristics"""
    
    # Organism-specific trait patterns
    ORGANISM_TRAITS = {
        'mycobacterium_tuberculosis': {
            'pathogenesis': {
                'name': 'Pathogenesis',
                'description': 'Virulence and disease-causing mechanisms',
                'keywords': ['virulence', 'toxin', 'invasion', 'evasion'],
                'codon_bias': {'GCG': 0.35, 'GCC': 0.30, 'TGC': 0.25}
            },
            'dormancy': {
                'name': 'Dormancy',
                'description': 'Latent infection and persistence',
                'keywords': ['dormancy', 'persistence', 'stress'],
                'codon_bias': {'ATC': 0.28, 'CTG': 0.32, 'GAC': 0.25}
            },
            'cell_wall': {
                'name': 'Cell Wall Synthesis',
                'description': 'Mycolic acid and cell wall components',
                'keywords': ['mycolic', 'cell wall', 'lipid'],
                'codon_bias': {'CGC': 0.40, 'GAG': 0.35, 'TTC': 0.20}
            }
        },
        'lactobacillus_acidophilus': {
            'probiotic': {
                'name': 'Probiotic Function',
                'description': 'Beneficial host interactions',
                'keywords': ['adhesion', 'colonization', 'beneficial'],
                'codon_bias': {'AAA': 0.35, 'GAA': 0.30, 'TTA': 0.25}
            },
            'acid_tolerance': {
                'name': 'Acid Tolerance',
                'description': 'Survival in acidic environments',
                'keywords': ['acid', 'pH', 'tolerance', 'stress'],
                'codon_bias': {'CAA': 0.32, 'GGA': 0.28, 'AGA': 0.22}
            },
            'lactose_metabolism': {
                'name': 'Lactose Metabolism',
                'description': 'Fermentation and sugar utilization',
                'keywords': ['lactose', 'fermentation', 'sugar'],
                'codon_bias': {'CTT': 0.30, 'ATT': 0.35, 'GGT': 0.25}
            }
        },
        'streptococcus_pneumoniae': {
            'capsule': {
                'name': 'Capsule Production',
                'description': 'Polysaccharide capsule synthesis',
                'keywords': ['capsule', 'polysaccharide', 'cps'],
                'codon_bias': {'AAG': 0.38, 'TTG': 0.32, 'CCT': 0.22}
            },
            'competence': {
                'name': 'Natural Competence',
                'description': 'DNA uptake and transformation',
                'keywords': ['competence', 'transformation', 'com'],
                'codon_bias': {'GAT': 0.35, 'CAT': 0.30, 'ACT': 0.25}
            },
            'virulence': {
                'name': 'Virulence Factors',
                'description': 'Disease-causing mechanisms',
                'keywords': ['pneumolysin', 'autolysin', 'virulence'],
                'codon_bias': {'GCT': 0.33, 'TCT': 0.28, 'AGT': 0.24}
            }
        },
        'bacillus_subtilis': {
            'sporulation': {
                'name': 'Sporulation',
                'description': 'Endospore formation',
                'keywords': ['spore', 'sporulation', 'spo'],
                'codon_bias': {'GAG': 0.40, 'CTG': 0.35, 'AAG': 0.20}
            },
            'competence': {
                'name': 'Competence',
                'description': 'Natural transformation ability',
                'keywords': ['competence', 'com', 'transformation'],
                'codon_bias': {'CGT': 0.32, 'GGC': 0.30, 'TGG': 0.25}
            },
            'motility': {
                'name': 'Motility',
                'description': 'Flagellar movement and chemotaxis',
                'keywords': ['flagella', 'motility', 'chemotaxis'],
                'codon_bias': {'ATC': 0.35, 'GAC': 0.32, 'TCC': 0.23}
            }
        },
        'helicobacter_pylori': {
            'urease': {
                'name': 'Urease Activity',
                'description': 'Acid neutralization in stomach',
                'keywords': ['urease', 'ure', 'acid'],
                'codon_bias': {'AAA': 0.45, 'TTT': 0.35, 'AGA': 0.15}
            },
            'adhesion': {
                'name': 'Adhesion',
                'description': 'Gastric epithelial attachment',
                'keywords': ['adhesin', 'bab', 'sab'],
                'codon_bias': {'CTT': 0.38, 'ATT': 0.32, 'GTT': 0.22}
            },
            'cag_pai': {
                'name': 'CagA Pathogenicity',
                'description': 'Type IV secretion and CagA',
                'keywords': ['cag', 'T4SS', 'pathogenicity'],
                'codon_bias': {'GAT': 0.36, 'CAT': 0.34, 'TAT': 0.20}
            }
        }
    }
    
    @classmethod
    def generate_trait_definitions(cls, organism_name: str) -> Dict:
        """Generate trait definitions for a specific organism"""
        # Normalize organism name
        safe_name = organism_name.lower().replace(' ', '_').replace('.', '')
        
        # Default traits for all bacteria
        default_traits = {
            'central_metabolism': {
                'name': 'Central Metabolism',
                'description': 'Core metabolic pathways',
                'keywords': ['glycolysis', 'citrate', 'metabolism'],
                'codon_bias': {'GCG': 0.30, 'GAA': 0.35, 'CTG': 0.25}
            },
            'stress_response': {
                'name': 'Stress Response',
                'description': 'Environmental stress adaptation',
                'keywords': ['heat shock', 'stress', 'chaperone'],
                'codon_bias': {'AAG': 0.32, 'GAG': 0.33, 'CGC': 0.25}
            },
            'protein_synthesis': {
                'name': 'Protein Synthesis',
                'description': 'Ribosomal and translation machinery',
                'keywords': ['ribosome', 'translation', 'tRNA'],
                'codon_bias': {'AAA': 0.28, 'GAA': 0.38, 'CGA': 0.24}
            }
        }
        
        # Get organism-specific traits if available
        specific_traits = {}
        for key in cls.ORGANISM_TRAITS:
            if key in safe_name:
                specific_traits = cls.ORGANISM_TRAITS[key]
                break
        
        # Combine default and specific traits
        all_traits = {**default_traits, **specific_traits}
        
        # Format for the analysis pipeline
        trait_definitions = {
            'organism': organism_name,
            'traits': {}
        }
        
        for trait_id, trait_data in all_traits.items():
            trait_definitions['traits'][trait_id] = {
                'name': trait_data['name'],
                'description': trait_data['description'],
                'detection_keywords': trait_data['keywords'],
                'expected_codon_bias': trait_data['codon_bias'],
                'confidence_threshold': 0.6
            }
        
        return trait_definitions


def create_experiment_config(genome_metadata: Dict, output_dir: str) -> Dict:
    """Create experiment configuration for a genome"""
    config = {
        'genome_file': genome_metadata['file_path'],
        'organism': genome_metadata['organism'],
        'strain': genome_metadata['strain'],
        'accession': genome_metadata['accession'],
        'genome_size': genome_metadata['size'],
        'analysis_parameters': {
            'window_size': 300,
            'window_overlap': 50,
            'min_gene_length': 100,
            'confidence_threshold': 0.5
        },
        'output_directory': os.path.join(output_dir, 'results')
    }
    
    return config


def main():
    """Test utilities with downloaded genomes"""
    genomes_dir = "genomes"
    
    # Load metadata
    metadata_file = os.path.join(genomes_dir, "genome_metadata.json")
    if not os.path.exists(metadata_file):
        print("No genome metadata found. Run genome_acquisition.py first.")
        return
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Validate genomes and generate trait definitions
    print("Validating downloaded genomes...\n")
    
    for genome_id, genome_info in metadata.items():
        print(f"Processing {genome_info['organism']}...")
        
        # Validate FASTA
        validator = GenomeValidator()
        validation = validator.validate_fasta(genome_info['file_path'])
        
        print(f"  - Valid: {validation['valid']}")
        print(f"  - Size: {validation['total_length']:,} bp")
        print(f"  - GC Content: {validation['gc_content']:.1f}%")
        print(f"  - Checksum: {validation['checksum']}")
        
        if validation['errors']:
            print(f"  - Errors: {validation['errors']}")
        
        # Generate trait definitions
        generator = TraitDefinitionGenerator()
        traits = generator.generate_trait_definitions(genome_id)
        
        # Save trait definitions
        trait_file = os.path.join(genomes_dir, f"{genome_id}_traits.json")
        with open(trait_file, 'w') as f:
            json.dump(traits, f, indent=2)
        
        print(f"  - Generated {len(traits['traits'])} trait definitions")
        print(f"  - Saved to {trait_file}\n")
        
        # Create experiment config
        config = create_experiment_config(genome_info, genomes_dir)
        config_file = os.path.join(genomes_dir, f"{genome_id}_config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()