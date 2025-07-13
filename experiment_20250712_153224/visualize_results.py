#!/usr/bin/env python3
"""Visualize pleiotropy analysis results"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import numpy as np

def load_results():
    """Load analysis results"""
    with open('results/analysis_results.json', 'r') as f:
        analysis = json.load(f)
    
    with open('results/pleiotropic_genes.json', 'r') as f:
        pleiotropic = json.load(f)
    
    return analysis, pleiotropic

def create_summary_report(analysis, pleiotropic):
    """Create a summary report"""
    report = []
    report.append("# Pleiotropy Experiment Results\n")
    report.append(f"**Date**: January 12, 2025\n")
    report.append(f"**Genome**: E. coli K-12 MG1655 (4.64 Mb)\n")
    report.append(f"**Analysis Time**: ~1 second\n\n")
    
    report.append("## Summary Statistics\n")
    report.append(f"- Total sequences analyzed: {analysis['sequences']}\n")
    report.append(f"- Pleiotropic genes found: {len(pleiotropic)}\n")
    report.append(f"- Total traits identified: {len(set(t['trait_names'][0] for t in analysis['identified_traits']))}\n\n")
    
    report.append("## Pleiotropic Genes\n")
    for gene in pleiotropic:
        report.append(f"\n### Gene: {gene['gene_id']}\n")
        report.append(f"- **Traits**: {', '.join(gene['traits'])}\n")
        report.append(f"- **Confidence**: {gene['confidence']:.2f}\n")
    
    report.append("\n## Trait Analysis\n")
    trait_counts = defaultdict(int)
    trait_confidences = defaultdict(list)
    
    for trait in analysis['identified_traits']:
        trait_name = trait['trait_names'][0]
        trait_counts[trait_name] += 1
        trait_confidences[trait_name].append(trait['confidence_score'])
    
    for trait_name in sorted(trait_counts.keys()):
        avg_conf = np.mean(trait_confidences[trait_name])
        report.append(f"\n### {trait_name}\n")
        report.append(f"- Regions identified: {trait_counts[trait_name]}\n")
        report.append(f"- Average confidence: {avg_conf:.2f}\n")
        
        # Find associated genes
        for t in analysis['identified_traits']:
            if t['trait_names'][0] == trait_name:
                report.append(f"- Associated genes: {', '.join(t['associated_genes'])}\n")
                break
    
    # Save report
    with open('experiment_report.md', 'w') as f:
        f.write(''.join(report))
    
    return ''.join(report)

def visualize_genome_traits(analysis):
    """Create a genome visualization showing trait regions"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Genome length (from the analysis results)
    genome_length = 4641652
    
    # Draw genome backbone
    ax.add_patch(patches.Rectangle((0, 0.4), genome_length, 0.2, 
                                   facecolor='lightgray', edgecolor='black'))
    
    # Color map for traits
    colors = {
        'regulatory': '#FF6B6B',
        'stress_response': '#4ECDC4',
        'carbon_metabolism': '#45B7D1',
        'motility': '#96CEB4',
        'dna_processing': '#DDA0DD',
        'structural': '#F4A460'
    }
    
    # Plot trait regions
    y_positions = {'regulatory': 0.7, 'stress_response': 0.1}
    
    for trait in analysis['identified_traits']:
        trait_name = trait['trait_names'][0]
        color = colors.get(trait_name, 'gray')
        y_pos = y_positions.get(trait_name, 0.5)
        
        for region in trait['contributing_regions']:
            start, end = region
            width = end - start
            
            # Draw trait region
            rect = patches.Rectangle((start, y_pos), width, 0.2,
                                   facecolor=color, alpha=0.7,
                                   edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
    
    # Add trait labels
    for trait_name, y_pos in y_positions.items():
        ax.text(-200000, y_pos + 0.1, trait_name, 
                horizontalalignment='right', verticalalignment='center',
                fontsize=10, fontweight='bold')
    
    # Add gene annotations
    ax.text(genome_length/2, 0.5, 'E. coli K-12 Genome', 
            horizontalalignment='center', verticalalignment='center',
            fontsize=12, fontweight='bold')
    
    # Format plot
    ax.set_xlim(-300000, genome_length + 100000)
    ax.set_ylim(-0.1, 1.0)
    ax.set_xlabel('Genome Position (bp)', fontsize=12)
    ax.set_title('Pleiotropic Trait Distribution in E. coli K-12', fontsize=14, fontweight='bold')
    
    # Remove y-axis
    ax.set_yticks([])
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    # Format x-axis
    ax.ticklabel_format(style='plain', axis='x')
    
    plt.tight_layout()
    plt.savefig('genome_trait_map.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_confidence_plot(analysis):
    """Create confidence score distribution plot"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Extract confidence scores by trait
    trait_scores = defaultdict(list)
    
    for trait in analysis['identified_traits']:
        trait_name = trait['trait_names'][0]
        trait_scores[trait_name].append(trait['confidence_score'])
    
    # Create bar plot
    traits = list(trait_scores.keys())
    avg_scores = [np.mean(trait_scores[t]) for t in traits]
    
    bars = ax.bar(traits, avg_scores, color=['#FF6B6B', '#4ECDC4'])
    
    # Add value labels on bars
    for bar, score in zip(bars, avg_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}', ha='center', va='bottom')
    
    # Formatting
    ax.set_ylabel('Average Confidence Score', fontsize=12)
    ax.set_title('Trait Detection Confidence Scores', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add threshold line
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='High Confidence Threshold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('confidence_scores.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main visualization function"""
    print("Loading results...")
    analysis, pleiotropic = load_results()
    
    print("Creating summary report...")
    report = create_summary_report(analysis, pleiotropic)
    print("\nReport saved to: experiment_report.md")
    
    print("Creating genome trait map...")
    visualize_genome_traits(analysis)
    print("Visualization saved to: genome_trait_map.png")
    
    print("Creating confidence plot...")
    create_confidence_plot(analysis)
    print("Plot saved to: confidence_scores.png")
    
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(report)

if __name__ == "__main__":
    main()