#!/usr/bin/env python3
"""
Statistical Visualizations for Pleiotropic Gene Analysis
========================================================

Creates comprehensive visualizations of statistical analysis results.

MEMORY NAMESPACE: swarm-pleiotropy-analysis-1752302124/statistical-analyzer/visualizations
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_results(file_path="reports/comprehensive_statistical_analysis.json"):
    """Load statistical analysis results."""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_codon_bias_visualization(codon_data, output_dir="reports/visualizations"):
    """Create comprehensive codon usage bias visualizations."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract data
    amino_acids = list(codon_data['amino_acid_bias'].keys())
    p_values = [codon_data['amino_acid_bias'][aa]['p_value'] for aa in amino_acids]
    p_adjusted = [codon_data['amino_acid_bias'][aa]['p_adjusted'] for aa in amino_acids]
    bias_strengths = [codon_data['amino_acid_bias'][aa]['bias_strength'] for aa in amino_acids]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Codon Usage Bias Analysis', fontsize=16, fontweight='bold')
    
    # 1. P-values comparison (raw vs adjusted)
    x_pos = np.arange(len(amino_acids))
    width = 0.35
    
    ax1.bar(x_pos - width/2, [-np.log10(p) for p in p_values], width, 
            label='Raw p-values', alpha=0.7, color='skyblue')
    ax1.bar(x_pos + width/2, [-np.log10(p) for p in p_adjusted], width,
            label='Adjusted p-values', alpha=0.7, color='orange')
    
    ax1.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='α = 0.05')
    ax1.set_xlabel('Amino Acid')
    ax1.set_ylabel('-log₁₀(p-value)')
    ax1.set_title('Statistical Significance of Codon Usage Bias')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(amino_acids, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Bias strength distribution
    ax2.bar(amino_acids, bias_strengths, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Amino Acid')
    ax2.set_ylabel('Bias Strength (Standard Deviation)')
    ax2.set_title('Codon Usage Bias Strength by Amino Acid')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter plot: Bias strength vs significance
    significant = [codon_data['amino_acid_bias'][aa]['significant_after_correction'] for aa in amino_acids]
    colors = ['red' if sig else 'blue' for sig in significant]
    
    ax3.scatter(bias_strengths, [-np.log10(p) for p in p_adjusted], c=colors, alpha=0.7, s=100)
    ax3.set_xlabel('Bias Strength')
    ax3.set_ylabel('-log₁₀(Adjusted p-value)')
    ax3.set_title('Bias Strength vs Statistical Significance')
    ax3.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7)
    
    # Add amino acid labels
    for i, aa in enumerate(amino_acids):
        ax3.annotate(aa, (bias_strengths[i], -np.log10(p_adjusted[i])), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax3.grid(True, alpha=0.3)
    
    # 4. Chi-squared statistics
    chi2_stats = [codon_data['amino_acid_bias'][aa]['chi2_statistic'] for aa in amino_acids]
    ax4.bar(amino_acids, chi2_stats, color='lightgreen', alpha=0.7)
    ax4.set_xlabel('Amino Acid')
    ax4.set_ylabel('Chi-squared Statistic')
    ax4.set_title('Chi-squared Test Statistics')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/codon_usage_bias_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Codon bias visualization saved to {output_dir}/codon_usage_bias_analysis.png")

def create_trait_correlation_heatmap(trait_data, output_dir="reports/visualizations"):
    """Create trait correlation matrix heatmap."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert correlation matrix to DataFrame
    corr_matrix = pd.DataFrame(trait_data['correlation_matrix'])
    p_matrix = pd.DataFrame(trait_data['p_value_matrix'])
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Correlation heatmap
    mask = np.triu(np.ones_like(corr_matrix))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, ax=ax1, cbar_kws={"shrink": .8})
    ax1.set_title('Trait Correlation Matrix')
    
    # Significance heatmap
    significance_mask = (p_matrix < 0.05).astype(int)
    significance_mask = np.where(mask, np.nan, significance_mask)
    
    sns.heatmap(p_matrix, mask=mask, annot=True, cmap='YlOrRd_r', 
                square=True, ax=ax2, cbar_kws={"shrink": .8})
    ax2.set_title('P-value Matrix')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/trait_correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Significant correlations bar plot
    sig_corr = trait_data['significant_correlations']
    if sig_corr:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        trait_pairs = [f"{sc['trait1']} - {sc['trait2']}" for sc in sig_corr]
        correlations = [sc['correlation'] for sc in sig_corr]
        p_values = [sc['p_value'] for sc in sig_corr]
        
        bars = ax.bar(range(len(trait_pairs)), correlations, 
                     color=['red' if p < 0.001 else 'orange' if p < 0.01 else 'yellow' 
                           for p in p_values], alpha=0.7)
        
        ax.set_xlabel('Trait Pairs')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title('Significant Trait Correlations')
        ax.set_xticks(range(len(trait_pairs)))
        ax.set_xticklabels(trait_pairs, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/significant_trait_correlations.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Trait correlation visualizations saved to {output_dir}/")

def create_pca_visualization(pca_data, output_dir="reports/visualizations"):
    """Create PCA analysis visualizations."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    explained_var = pca_data['explained_variance_ratio']
    cumulative_var = pca_data['cumulative_variance_ratio']
    trait_names = pca_data['trait_names']
    loadings = np.array(pca_data['loadings'])
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Principal Component Analysis of Traits', fontsize=16, fontweight='bold')
    
    # 1. Scree plot
    components = range(1, len(explained_var) + 1)
    ax1.bar(components, explained_var, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Scree Plot - Individual Component Variance')
    ax1.grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, var in enumerate(explained_var):
        ax1.text(i + 1, var + 0.005, f'{var:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Cumulative variance
    ax2.plot(components, cumulative_var, 'o-', color='red', linewidth=2, markersize=6)
    ax2.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='90% Variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Variance Explained')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add percentage labels
    for i, var in enumerate(cumulative_var):
        if i % 2 == 0:  # Show every other label to avoid crowding
            ax2.text(i + 1, var + 0.02, f'{var:.1%}', ha='center', va='bottom', fontsize=8)
    
    # 3. PC1 vs PC2 loadings
    if len(loadings) >= 2:
        ax3.scatter(loadings[0], loadings[1], s=100, alpha=0.7, color='purple')
        
        for i, trait in enumerate(trait_names):
            ax3.annotate(trait, (loadings[0][i], loadings[1][i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax3.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
        ax3.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
        ax3.set_title('PC1 vs PC2 Loadings')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 4. Loadings heatmap for first 3 components
    n_components_show = min(3, len(loadings))
    loadings_df = pd.DataFrame(
        loadings[:n_components_show].T, 
        columns=[f'PC{i+1}' for i in range(n_components_show)],
        index=trait_names
    )
    
    sns.heatmap(loadings_df, annot=True, cmap='RdBu_r', center=0, ax=ax4,
                cbar_kws={"shrink": .8})
    ax4.set_title('Component Loadings Matrix')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PCA visualization saved to {output_dir}/pca_analysis.png")

def create_mutual_information_visualization(mi_data, output_dir="reports/visualizations"):
    """Create mutual information analysis visualizations."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert MI matrix to DataFrame
    mi_matrix = pd.DataFrame(mi_data['mutual_information_matrix']).T
    high_mi_pairs = mi_data['high_mi_pairs']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Mutual information heatmap
    sns.heatmap(mi_matrix, annot=True, cmap='viridis', ax=ax1,
                cbar_kws={"shrink": .8})
    ax1.set_title('Gene-Trait Mutual Information Matrix')
    ax1.set_xlabel('Traits')
    ax1.set_ylabel('Genes')
    
    # 2. High MI pairs bar plot
    if high_mi_pairs:
        pair_labels = [f"{pair['gene']} - {pair['trait']}" for pair in high_mi_pairs]
        mi_values = [pair['mutual_information'] for pair in high_mi_pairs]
        
        bars = ax2.barh(range(len(pair_labels)), mi_values, color='orange', alpha=0.7)
        ax2.set_yticks(range(len(pair_labels)))
        ax2.set_yticklabels(pair_labels)
        ax2.set_xlabel('Mutual Information')
        ax2.set_title('High Mutual Information Gene-Trait Pairs')
        ax2.axvline(x=mi_data['threshold'], color='red', linestyle='--', alpha=0.7, 
                   label=f'Threshold = {mi_data["threshold"]}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, mi_values)):
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                    f'{value:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mutual_information_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Mutual information visualization saved to {output_dir}/mutual_information_analysis.png")

def create_validation_summary(validation_data, output_dir="reports/visualizations"):
    """Create validation summary visualization."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Validation metrics
    metrics = ['Precision', 'Recall', 'F1 Score']
    values = [validation_data['precision'], validation_data['recall'], validation_data['f1_score']]
    
    # Confusion matrix data
    tp = validation_data['true_positives']
    fp = validation_data['false_positives'] 
    fn = validation_data['false_negatives']
    # Note: true negatives not available in current data
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Performance metrics
    bars = ax1.bar(metrics, values, color=['steelblue', 'orange', 'green'], alpha=0.7)
    ax1.set_ylabel('Score')
    ax1.set_title('Validation Performance Metrics')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Known vs Predicted genes
    known_genes = validation_data['known_pleiotropic_genes']
    predicted_genes = validation_data['predicted_genes']
    missed_genes = validation_data['missed_genes']
    
    # Simple bar chart showing counts
    categories = ['Known Genes', 'Predicted Genes', 'True Positives', 'False Negatives']
    counts = [len(known_genes), len(predicted_genes), tp, fn]
    colors = ['blue', 'orange', 'green', 'red']
    
    bars = ax2.bar(categories, counts, color=colors, alpha=0.7)
    ax2.set_ylabel('Count')
    ax2.set_title('Gene Detection Summary')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/validation_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed gene comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # List known genes that were missed
    y_pos = np.arange(len(known_genes))
    colors = ['red' if gene in missed_genes else 'green' for gene in known_genes]
    
    ax.barh(y_pos, [1] * len(known_genes), color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(known_genes)
    ax.set_xlabel('Detection Status')
    ax.set_title('Known Pleiotropic Genes Detection Status')
    ax.set_xlim(0, 1.2)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='Detected'),
                      Patch(facecolor='red', alpha=0.7, label='Missed')]
    ax.legend(handles=legend_elements)
    
    # Add status labels
    for i, gene in enumerate(known_genes):
        status = 'MISSED' if gene in missed_genes else 'DETECTED'
        ax.text(0.5, i, status, ha='center', va='center', fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gene_detection_status.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Validation visualizations saved to {output_dir}/")

def create_comprehensive_dashboard(results, output_dir="reports/visualizations"):
    """Create a comprehensive dashboard of all results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create summary statistics figure
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(6, 4, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('COMPREHENSIVE STATISTICAL ANALYSIS DASHBOARD\nGenomic Pleiotropy Cryptanalysis', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Overall summary (top row)
    ax1 = fig.add_subplot(gs[0, :2])
    summary = results['summary']
    
    # Summary metrics
    summary_text = f"""
ANALYSIS SUMMARY
• Total Analyses Performed: {summary['total_analyses_performed']}
• Significant Findings: {len(summary['significant_findings'])}
• Known Pleiotropic Genes: {len(results['validation']['known_pleiotropic_genes'])}
• Detected Genes: {len(results['validation']['predicted_genes'])}

KEY FINDINGS:
{chr(10).join('• ' + finding for finding in summary['significant_findings'])}
    """
    
    ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # 2. Performance metrics (top right)
    ax2 = fig.add_subplot(gs[0, 2:])
    validation = results['validation']
    metrics = ['Precision', 'Recall', 'F1 Score']
    values = [validation['precision'], validation['recall'], validation['f1_score']]
    
    bars = ax2.bar(metrics, values, color=['steelblue', 'orange', 'green'], alpha=0.7)
    ax2.set_ylabel('Score')
    ax2.set_title('Validation Performance Metrics', fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Codon bias strength (second row left)
    ax3 = fig.add_subplot(gs[1, :2])
    codon_data = results['codon_usage_bias']
    amino_acids = list(codon_data['amino_acid_bias'].keys())
    bias_strengths = [codon_data['amino_acid_bias'][aa]['bias_strength'] for aa in amino_acids]
    
    ax3.bar(amino_acids, bias_strengths, color='lightcoral', alpha=0.7)
    ax3.set_xlabel('Amino Acid')
    ax3.set_ylabel('Bias Strength')
    ax3.set_title('Codon Usage Bias Strength by Amino Acid', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. PCA variance (second row right)
    ax4 = fig.add_subplot(gs[1, 2:])
    pca_data = results['pca_analysis']
    explained_var = pca_data['explained_variance_ratio']
    components = range(1, len(explained_var) + 1)
    
    ax4.bar(components, explained_var, alpha=0.7, color='steelblue')
    ax4.set_xlabel('Principal Component')
    ax4.set_ylabel('Explained Variance Ratio')
    ax4.set_title('PCA - Component Variance Explained', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Trait correlation heatmap (third row)
    ax5 = fig.add_subplot(gs[2, :])
    trait_data = results['trait_correlations']
    corr_matrix = pd.DataFrame(trait_data['correlation_matrix'])
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, ax=ax5, cbar_kws={"shrink": .5})
    ax5.set_title('Trait Correlation Matrix', fontweight='bold')
    
    # 6. Bootstrap confidence interval (fourth row left)
    ax6 = fig.add_subplot(gs[3, :2])
    bootstrap_data = results['bootstrap_confidence']
    mean_val = bootstrap_data['mean']
    ci_lower, ci_upper = bootstrap_data['confidence_interval']
    
    ax6.bar(['Codon Frequencies'], [mean_val], yerr=[[mean_val - ci_lower], [ci_upper - mean_val]], 
           capsize=10, color='gold', alpha=0.7)
    ax6.set_ylabel('Mean Frequency')
    ax6.set_title('Bootstrap Confidence Interval (95%)', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Add CI text
    ax6.text(0, mean_val + (ci_upper - ci_lower) * 0.3, 
            f'CI: [{ci_lower:.4f}, {ci_upper:.4f}]',
            ha='center', va='bottom', fontweight='bold')
    
    # 7. Known genes status (fourth row right)
    ax7 = fig.add_subplot(gs[3, 2:])
    known_genes = validation['known_pleiotropic_genes']
    missed_genes = validation['missed_genes']
    
    y_pos = np.arange(len(known_genes))
    colors = ['red' if gene in missed_genes else 'green' for gene in known_genes]
    
    ax7.barh(y_pos, [1] * len(known_genes), color=colors, alpha=0.7)
    ax7.set_yticks(y_pos)
    ax7.set_yticklabels(known_genes)
    ax7.set_xlabel('Detection Status')
    ax7.set_title('Known Pleiotropic Genes Status', fontweight='bold')
    ax7.set_xlim(0, 1.2)
    
    # 8. Statistical significance levels (fifth row)
    ax8 = fig.add_subplot(gs[4, :])
    
    # Collect all p-values from different analyses
    p_val_categories = []
    p_values = []
    
    # Codon bias p-values
    for aa in amino_acids:
        p_val_categories.append(f'Codon-{aa}')
        p_values.append(codon_data['amino_acid_bias'][aa]['p_adjusted'])
    
    # Trait correlation p-values
    for sc in trait_data['significant_correlations']:
        p_val_categories.append(f'Trait-{sc["trait1"][:3]}-{sc["trait2"][:3]}')
        p_values.append(sc['p_value'])
    
    # Plot significance levels
    log_p_values = [-np.log10(p) for p in p_values]
    colors = ['red' if p < 0.001 else 'orange' if p < 0.01 else 'yellow' for p in p_values]
    
    ax8.bar(range(len(log_p_values)), log_p_values, color=colors, alpha=0.7)
    ax8.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='α = 0.05')
    ax8.axhline(y=-np.log10(0.01), color='orange', linestyle='--', alpha=0.7, label='α = 0.01')
    ax8.axhline(y=-np.log10(0.001), color='green', linestyle='--', alpha=0.7, label='α = 0.001')
    
    ax8.set_xlabel('Analysis')
    ax8.set_ylabel('-log₁₀(p-value)')
    ax8.set_title('Statistical Significance Across All Analyses', fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Analysis timestamp and metadata (bottom)
    ax9 = fig.add_subplot(gs[5, :])
    metadata = results['metadata']
    
    metadata_text = f"""
ANALYSIS METADATA
• Timestamp: {metadata['analysis_timestamp']}
• Memory Namespace: {metadata['memory_namespace']}
• Statistical Parameters: α = {metadata['statistical_parameters']['alpha']}, 
  Bootstrap Iterations = {metadata['statistical_parameters']['bootstrap_iterations']},
  Multiple Testing Correction = {metadata['statistical_parameters']['multiple_testing_method']}

VALIDATION NOTES:
• Analysis detected no pleiotropic genes (all 5 known genes missed)
• Codon usage bias analysis shows significant deviations from expected frequencies
• Strong trait correlations found between related biological processes
• PCA analysis explains >90% variance with 5 components
    """
    
    ax9.text(0.05, 0.95, metadata_text, transform=ax9.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    
    plt.savefig(f"{output_dir}/comprehensive_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive dashboard saved to {output_dir}/comprehensive_dashboard.png")

def main():
    """Main visualization generation function."""
    print("Creating comprehensive statistical visualizations...")
    
    # Load results
    results = load_results()
    
    # Create all visualizations
    create_codon_bias_visualization(results['codon_usage_bias'])
    create_trait_correlation_heatmap(results['trait_correlations'])
    create_pca_visualization(results['pca_analysis'])
    create_mutual_information_visualization(results['mutual_information'])
    create_validation_summary(results['validation'])
    create_comprehensive_dashboard(results)
    
    print("\nAll visualizations created successfully!")
    print("Files saved to reports/visualizations/")

if __name__ == "__main__":
    main()