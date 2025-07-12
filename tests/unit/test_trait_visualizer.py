"""
Unit tests for the TraitVisualizer module.

Tests cover all visualization functionality including:
- Correlation heatmaps
- Network graphs
- Distribution plots
- Interactive visualizations
- Sankey diagrams
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Use non-interactive backend for testing
matplotlib.use('Agg')

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from python_analysis.trait_visualizer import TraitVisualizer


class TestTraitVisualizer:
    """Test suite for TraitVisualizer class."""
    
    @pytest.fixture
    def visualizer(self):
        """Create a TraitVisualizer instance for testing."""
        return TraitVisualizer(style='seaborn')
    
    @pytest.fixture
    def sample_trait_data(self):
        """Create sample trait data for testing."""
        np.random.seed(42)
        n_samples = 100
        data = {
            'trait1': np.random.normal(0, 1, n_samples),
            'trait2': np.random.normal(0, 1, n_samples),
            'trait3': np.random.normal(0, 1, n_samples),
            'trait4': np.random.normal(0, 1, n_samples)
        }
        # Add correlation between trait1 and trait2
        data['trait2'] = data['trait1'] * 0.7 + np.random.normal(0, 0.5, n_samples)
        return pd.DataFrame(data)
    
    @pytest.fixture
    def gene_trait_associations(self):
        """Create sample gene-trait associations."""
        return {
            'gene1': ['trait1', 'trait2', 'trait3'],
            'gene2': ['trait2', 'trait4'],
            'gene3': ['trait1', 'trait3', 'trait4'],
            'gene4': ['trait1'],
            'gene5': ['trait2', 'trait3', 'trait4']
        }
    
    @pytest.fixture
    def gene_scores(self):
        """Create sample pleiotropy scores."""
        return {
            'gene1': 3.5,
            'gene2': 2.1,
            'gene3': 4.2,
            'gene4': 1.0,
            'gene5': 3.8,
            'gene6': 2.5,
            'gene7': 1.5,
            'gene8': 3.0
        }
    
    def test_init(self):
        """Test TraitVisualizer initialization."""
        viz = TraitVisualizer()
        assert viz.default_figsize == (12, 8)
        assert viz.colormap == 'coolwarm'
        
        # Test with custom style
        with patch('matplotlib.pyplot.style.use') as mock_style:
            viz = TraitVisualizer(style='ggplot')
            mock_style.assert_called_with('ggplot')
    
    def test_plot_trait_correlation_heatmap(self, visualizer, sample_trait_data):
        """Test correlation heatmap plotting."""
        fig = visualizer.plot_trait_correlation_heatmap(sample_trait_data)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0  # Should have at least one axes
        
        # Check that correlation was calculated
        ax = fig.axes[0]
        assert ax.get_title() == 'Trait Correlation Matrix (Pearson)'
        
        plt.close(fig)
    
    def test_plot_trait_correlation_heatmap_custom_params(self, visualizer, sample_trait_data, tmp_path):
        """Test correlation heatmap with custom parameters."""
        save_path = tmp_path / "correlation_heatmap.png"
        
        fig = visualizer.plot_trait_correlation_heatmap(
            sample_trait_data,
            method='spearman',
            figsize=(10, 10),
            save_path=str(save_path)
        )
        
        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 10
        assert save_path.exists()
        
        plt.close(fig)
    
    def test_plot_gene_trait_network(self, visualizer, gene_trait_associations):
        """Test gene-trait network plotting."""
        fig = visualizer.plot_gene_trait_network(gene_trait_associations)
        
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert ax.get_title() == 'Gene-Trait Association Network'
        
        # Verify legend exists
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 2  # Genes and Traits
        
        plt.close(fig)
    
    def test_plot_gene_trait_network_layouts(self, visualizer, gene_trait_associations):
        """Test different network layouts."""
        layouts = ['spring', 'circular', 'kamada_kawai']
        
        for layout in layouts:
            fig = visualizer.plot_gene_trait_network(
                gene_trait_associations,
                layout=layout
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_plot_gene_trait_network_save(self, visualizer, gene_trait_associations, tmp_path):
        """Test saving network plot."""
        save_path = tmp_path / "network.png"
        
        fig = visualizer.plot_gene_trait_network(
            gene_trait_associations,
            save_path=str(save_path)
        )
        
        assert save_path.exists()
        plt.close(fig)
    
    def test_plot_trait_distribution(self, visualizer, sample_trait_data):
        """Test trait distribution plotting."""
        fig = visualizer.plot_trait_distribution(
            sample_trait_data,
            'trait1',
            plot_type='both'
        )
        
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert ax.get_title() == 'Distribution of trait1'
        assert ax.get_xlabel() == 'trait1'
        assert ax.get_ylabel() == 'Density'
        
        # Check for mean and median lines
        lines = ax.get_lines()
        assert len(lines) >= 3  # KDE line + mean + median
        
        plt.close(fig)
    
    def test_plot_trait_distribution_types(self, visualizer, sample_trait_data):
        """Test different distribution plot types."""
        for plot_type in ['hist', 'kde', 'both']:
            fig = visualizer.plot_trait_distribution(
                sample_trait_data,
                'trait1',
                plot_type=plot_type
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_plot_trait_distribution_with_nan(self, visualizer):
        """Test distribution plot with NaN values."""
        data = pd.DataFrame({
            'trait1': [1, 2, 3, np.nan, 5, 6, np.nan, 8, 9, 10]
        })
        
        fig = visualizer.plot_trait_distribution(data, 'trait1')
        assert isinstance(fig, plt.Figure)
        
        # Should handle NaN values gracefully
        ax = fig.axes[0]
        # Check that mean/median lines exist
        assert len(ax.get_lines()) >= 3
        
        plt.close(fig)
    
    def test_create_interactive_heatmap(self, visualizer, sample_trait_data):
        """Test interactive heatmap creation."""
        fig = visualizer.create_interactive_heatmap(sample_trait_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)
        
        # Check layout
        assert 'Interactive Trait Correlation Matrix' in fig.layout.title.text
        assert fig.layout.xaxis.title.text == 'Traits'
        assert fig.layout.yaxis.title.text == 'Traits'
    
    def test_create_interactive_heatmap_methods(self, visualizer, sample_trait_data):
        """Test interactive heatmap with different correlation methods."""
        for method in ['pearson', 'spearman', 'kendall']:
            fig = visualizer.create_interactive_heatmap(
                sample_trait_data,
                method=method
            )
            assert isinstance(fig, go.Figure)
            assert method.capitalize() in fig.layout.title.text
    
    def test_plot_pleiotropy_score_distribution(self, visualizer, gene_scores):
        """Test pleiotropy score distribution plotting."""
        fig = visualizer.plot_pleiotropy_score_distribution(gene_scores)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Two subplots
        
        # Check first subplot (bar plot)
        ax1 = fig.axes[0]
        assert 'Top' in ax1.get_title()
        assert ax1.get_ylabel() == 'Pleiotropy Score'
        
        # Check second subplot (histogram)
        ax2 = fig.axes[1]
        assert ax2.get_title() == 'Distribution of Pleiotropy Scores'
        assert ax2.get_xlabel() == 'Pleiotropy Score'
        assert ax2.get_ylabel() == 'Number of Genes'
        
        plt.close(fig)
    
    def test_plot_pleiotropy_score_distribution_with_threshold(self, visualizer, gene_scores):
        """Test pleiotropy score distribution with threshold."""
        fig = visualizer.plot_pleiotropy_score_distribution(
            gene_scores,
            threshold=3.0
        )
        
        # Check that threshold lines exist
        for ax in fig.axes:
            lines = [line for line in ax.get_lines() if line.get_linestyle() == '--']
            if lines:  # At least one subplot should have threshold line
                assert any('Threshold' in line.get_label() for line in lines)
        
        plt.close(fig)
    
    def test_plot_pleiotropy_score_distribution_many_genes(self, visualizer):
        """Test with many genes to ensure top N selection works."""
        # Create scores for 50 genes
        many_scores = {f'gene{i}': np.random.uniform(0, 5) for i in range(50)}
        
        fig = visualizer.plot_pleiotropy_score_distribution(many_scores)
        
        # First subplot should show only top 20
        ax1 = fig.axes[0]
        assert len(ax1.patches) == 20  # 20 bars
        
        plt.close(fig)
    
    def test_create_trait_gene_sankey(self, visualizer, gene_trait_associations):
        """Test Sankey diagram creation."""
        fig = visualizer.create_trait_gene_sankey(gene_trait_associations)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Sankey)
        
        # Check layout
        assert 'Gene-Trait Associations' in fig.layout.title.text
        
        # Check nodes and links
        sankey = fig.data[0]
        assert len(sankey.node.label) > 0
        assert len(sankey.link.source) > 0
        assert len(sankey.link.target) > 0
    
    def test_create_trait_gene_sankey_top_n(self, visualizer):
        """Test Sankey diagram with top N genes selection."""
        # Create many gene-trait associations
        large_associations = {
            f'gene{i}': [f'trait{j}' for j in range(i % 5 + 1)]
            for i in range(20)
        }
        
        fig = visualizer.create_trait_gene_sankey(
            large_associations,
            top_n_genes=5
        )
        
        # Should only include top 5 genes
        sankey = fig.data[0]
        gene_labels = [label for label in sankey.node.label if label.startswith('gene')]
        assert len(gene_labels) == 5
    
    def test_create_trait_gene_sankey_empty(self, visualizer):
        """Test Sankey diagram with empty associations."""
        fig = visualizer.create_trait_gene_sankey({})
        
        assert isinstance(fig, go.Figure)
        # Should handle empty data gracefully
        sankey = fig.data[0]
        assert len(sankey.node.label) == 0


class TestTraitVisualizerEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def visualizer(self):
        return TraitVisualizer()
    
    def test_empty_dataframe(self, visualizer):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        # Should handle empty data gracefully
        fig = visualizer.plot_trait_correlation_heatmap(empty_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_single_trait(self, visualizer):
        """Test with single trait."""
        single_trait = pd.DataFrame({'trait1': [1, 2, 3, 4, 5]})
        
        fig = visualizer.plot_trait_correlation_heatmap(single_trait)
        assert isinstance(fig, plt.Figure)
        
        # Correlation matrix should be 1x1
        ax = fig.axes[0]
        assert ax.get_xlim()[1] < 2  # Only one column
        
        plt.close(fig)
    
    def test_all_nan_trait(self, visualizer):
        """Test with all NaN trait values."""
        data = pd.DataFrame({
            'trait1': [1, 2, 3, 4, 5],
            'trait2': [np.nan] * 5
        })
        
        fig = visualizer.plot_trait_distribution(data, 'trait2')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_network_with_isolated_nodes(self, visualizer):
        """Test network plot with isolated nodes."""
        associations = {
            'gene1': ['trait1'],
            'gene2': [],  # Isolated gene
            'gene3': ['trait2']
        }
        
        fig = visualizer.plot_gene_trait_network(associations)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_invalid_trait_name(self, visualizer):
        """Test distribution plot with invalid trait name."""
        data = pd.DataFrame({'trait1': [1, 2, 3]})
        
        with pytest.raises(KeyError):
            visualizer.plot_trait_distribution(data, 'nonexistent_trait')


class TestTraitVisualizerIntegration:
    """Integration tests combining multiple visualizations."""
    
    @pytest.fixture
    def visualizer(self):
        return TraitVisualizer()
    
    @pytest.fixture
    def complete_dataset(self):
        """Create a complete dataset for integration testing."""
        np.random.seed(42)
        n_samples = 200
        
        # Create correlated traits
        trait_data = pd.DataFrame({
            'growth_rate': np.random.normal(1.0, 0.2, n_samples),
            'metabolic_rate': np.random.normal(2.0, 0.5, n_samples),
            'stress_response': np.random.normal(0.5, 0.1, n_samples),
            'motility': np.random.binomial(1, 0.7, n_samples),
            'virulence': np.random.exponential(0.5, n_samples)
        })
        
        # Add correlations
        trait_data['metabolic_rate'] += trait_data['growth_rate'] * 0.5
        trait_data['stress_response'] -= trait_data['virulence'] * 0.3
        
        # Gene associations
        gene_associations = {
            'ftsZ': ['growth_rate', 'stress_response'],
            'rpoS': ['stress_response', 'virulence', 'metabolic_rate'],
            'flhD': ['motility', 'virulence'],
            'crp': ['metabolic_rate', 'growth_rate', 'motility'],
            'lon': ['stress_response', 'growth_rate']
        }
        
        # Pleiotropy scores
        scores = {gene: len(traits) * (1 + np.random.uniform(-0.2, 0.2)) 
                 for gene, traits in gene_associations.items()}
        
        return trait_data, gene_associations, scores
    
    def test_complete_visualization_workflow(self, visualizer, complete_dataset, tmp_path):
        """Test complete visualization workflow."""
        trait_data, gene_associations, scores = complete_dataset
        
        # 1. Create correlation heatmap
        fig1 = visualizer.plot_trait_correlation_heatmap(
            trait_data,
            save_path=str(tmp_path / "correlation.png")
        )
        assert (tmp_path / "correlation.png").exists()
        plt.close(fig1)
        
        # 2. Create network visualization
        fig2 = visualizer.plot_gene_trait_network(
            gene_associations,
            save_path=str(tmp_path / "network.png")
        )
        assert (tmp_path / "network.png").exists()
        plt.close(fig2)
        
        # 3. Create trait distributions
        for trait in ['growth_rate', 'virulence']:
            fig = visualizer.plot_trait_distribution(
                trait_data,
                trait,
                save_path=str(tmp_path / f"{trait}_dist.png")
            )
            assert (tmp_path / f"{trait}_dist.png").exists()
            plt.close(fig)
        
        # 4. Create pleiotropy score distribution
        fig4 = visualizer.plot_pleiotropy_score_distribution(
            scores,
            threshold=2.5,
            save_path=str(tmp_path / "pleiotropy_scores.png")
        )
        assert (tmp_path / "pleiotropy_scores.png").exists()
        plt.close(fig4)
        
        # 5. Create interactive visualizations
        fig5 = visualizer.create_interactive_heatmap(trait_data)
        assert isinstance(fig5, go.Figure)
        
        fig6 = visualizer.create_trait_gene_sankey(gene_associations)
        assert isinstance(fig6, go.Figure)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])