"""
Trait Visualizer Module for Genomic Pleiotropy Analysis

This module provides visualization tools for exploring trait-gene relationships,
including heatmaps, network graphs, and correlation matrices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TraitVisualizer:
    """Main class for visualizing genomic trait relationships."""
    
    def __init__(self, style: str = 'seaborn'):
        """Initialize the visualizer with a specific style."""
        plt.style.use(style)
        self.default_figsize = (12, 8)
        self.colormap = 'coolwarm'
        
    def plot_trait_correlation_heatmap(self, 
                                     trait_data: pd.DataFrame,
                                     method: str = 'pearson',
                                     figsize: Optional[Tuple[int, int]] = None,
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a heatmap showing correlations between different traits.
        
        Args:
            trait_data: DataFrame with traits as columns and samples as rows
            method: Correlation method ('pearson', 'spearman', 'kendall')
            figsize: Figure size (width, height)
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        figsize = figsize or self.default_figsize
        
        # Calculate correlation matrix
        corr_matrix = trait_data.corr(method=method)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   cmap=self.colormap,
                   center=0,
                   annot=True,
                   fmt='.2f',
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title(f'Trait Correlation Matrix ({method.capitalize()})', 
                    fontsize=16, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_gene_trait_network(self,
                              gene_trait_associations: Dict[str, List[str]],
                              layout: str = 'spring',
                              node_size_factor: int = 300,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a network graph showing gene-trait associations.
        
        Args:
            gene_trait_associations: Dict mapping genes to list of associated traits
            layout: Network layout algorithm ('spring', 'circular', 'kamada_kawai')
            node_size_factor: Factor for scaling node sizes
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        # Create graph
        G = nx.Graph()
        
        # Add nodes and edges
        for gene, traits in gene_trait_associations.items():
            G.add_node(gene, node_type='gene')
            for trait in traits:
                G.add_node(trait, node_type='trait')
                G.add_edge(gene, trait)
        
        # Set layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Separate genes and traits
        gene_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'gene']
        trait_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'trait']
        
        # Calculate node sizes based on degree
        node_sizes = {n: G.degree(n) * node_size_factor for n in G.nodes()}
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                             nodelist=gene_nodes,
                             node_color='lightblue',
                             node_size=[node_sizes[n] for n in gene_nodes],
                             alpha=0.8,
                             ax=ax,
                             label='Genes')
        
        nx.draw_networkx_nodes(G, pos,
                             nodelist=trait_nodes,
                             node_color='lightcoral',
                             node_size=[node_sizes[n] for n in trait_nodes],
                             alpha=0.8,
                             ax=ax,
                             label='Traits')
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.5, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        
        ax.set_title('Gene-Trait Association Network', fontsize=18, pad=20)
        ax.legend(loc='upper right', fontsize=12)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_trait_distribution(self,
                              trait_data: pd.DataFrame,
                              trait_name: str,
                              plot_type: str = 'both',
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of a specific trait.
        
        Args:
            trait_data: DataFrame containing trait values
            trait_name: Name of the trait to plot
            plot_type: Type of plot ('hist', 'kde', 'both')
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        trait_values = trait_data[trait_name].dropna()
        
        if plot_type in ['hist', 'both']:
            ax.hist(trait_values, bins=30, alpha=0.7, color='skyblue', 
                   edgecolor='black', density=True, label='Histogram')
        
        if plot_type in ['kde', 'both']:
            trait_values.plot.kde(ax=ax, linewidth=2, color='red', label='KDE')
        
        ax.set_xlabel(trait_name, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Distribution of {trait_name}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = trait_values.mean()
        median_val = trait_values.median()
        ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def create_interactive_heatmap(self,
                                 trait_data: pd.DataFrame,
                                 method: str = 'pearson') -> go.Figure:
        """
        Create an interactive heatmap using Plotly.
        
        Args:
            trait_data: DataFrame with traits as columns
            method: Correlation method
            
        Returns:
            Plotly Figure object
        """
        # Calculate correlation matrix
        corr_matrix = trait_data.corr(method=method)
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f'Interactive Trait Correlation Matrix ({method.capitalize()})',
            xaxis_title='Traits',
            yaxis_title='Traits',
            width=800,
            height=800,
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'}
        )
        
        return fig
    
    def plot_pleiotropy_score_distribution(self,
                                         gene_scores: Dict[str, float],
                                         threshold: Optional[float] = None,
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of pleiotropy scores across genes.
        
        Args:
            gene_scores: Dictionary mapping gene names to pleiotropy scores
            threshold: Optional threshold to highlight
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Convert to sorted lists
        genes = list(gene_scores.keys())
        scores = list(gene_scores.values())
        sorted_indices = np.argsort(scores)[::-1]
        sorted_genes = [genes[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        # Plot 1: Bar plot of top genes
        top_n = min(20, len(sorted_genes))
        ax1.bar(range(top_n), sorted_scores[:top_n], color='steelblue')
        ax1.set_xticks(range(top_n))
        ax1.set_xticklabels(sorted_genes[:top_n], rotation=45, ha='right')
        ax1.set_ylabel('Pleiotropy Score', fontsize=12)
        ax1.set_title(f'Top {top_n} Genes by Pleiotropy Score', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')
        
        if threshold:
            ax1.axhline(threshold, color='red', linestyle='--', 
                       label=f'Threshold: {threshold}')
            ax1.legend()
        
        # Plot 2: Distribution histogram
        ax2.hist(scores, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Pleiotropy Score', fontsize=12)
        ax2.set_ylabel('Number of Genes', fontsize=12)
        ax2.set_title('Distribution of Pleiotropy Scores', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        if threshold:
            ax2.axvline(threshold, color='red', linestyle='--', 
                       label=f'Threshold: {threshold}')
            ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def create_trait_gene_sankey(self,
                               gene_trait_associations: Dict[str, List[str]],
                               top_n_genes: Optional[int] = 10) -> go.Figure:
        """
        Create a Sankey diagram showing gene-trait relationships.
        
        Args:
            gene_trait_associations: Dict mapping genes to traits
            top_n_genes: Number of top genes to include
            
        Returns:
            Plotly Figure object
        """
        # Prepare data for Sankey
        if top_n_genes:
            # Sort genes by number of associated traits
            sorted_genes = sorted(gene_trait_associations.items(), 
                                key=lambda x: len(x[1]), reverse=True)
            gene_trait_associations = dict(sorted_genes[:top_n_genes])
        
        # Create node lists
        genes = list(gene_trait_associations.keys())
        traits = list(set(trait for traits in gene_trait_associations.values() 
                         for trait in traits))
        
        # Create node labels and indices
        labels = genes + traits
        gene_indices = {gene: i for i, gene in enumerate(genes)}
        trait_indices = {trait: i + len(genes) for i, trait in enumerate(traits)}
        
        # Create links
        sources = []
        targets = []
        values = []
        
        for gene, associated_traits in gene_trait_associations.items():
            for trait in associated_traits:
                sources.append(gene_indices[gene])
                targets.append(trait_indices[trait])
                values.append(1)
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=["lightblue"] * len(genes) + ["lightcoral"] * len(traits)
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values
            )
        )])
        
        fig.update_layout(
            title="Gene-Trait Associations (Sankey Diagram)",
            font_size=12,
            height=600
        )
        
        return fig