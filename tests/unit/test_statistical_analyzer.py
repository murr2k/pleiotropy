"""
Unit tests for the StatisticalAnalyzer module.

Tests cover all major functionality including:
- Trait correlation calculations
- Gene-trait association testing
- Pleiotropy scoring
- PCA analysis
- Clustering
- Enrichment analysis
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from python_analysis.statistical_analyzer import StatisticalAnalyzer


class TestStatisticalAnalyzer:
    """Test suite for StatisticalAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a StatisticalAnalyzer instance for testing."""
        return StatisticalAnalyzer(multiple_testing_method='fdr_bh')
    
    @pytest.fixture
    def sample_trait_data(self):
        """Create sample trait data for testing."""
        np.random.seed(42)
        n_samples = 100
        data = {
            'trait1': np.random.normal(0, 1, n_samples),
            'trait2': np.random.normal(0, 1, n_samples),
            'trait3': np.random.normal(0, 1, n_samples),
        }
        # Add some correlation between trait1 and trait2
        data['trait2'] = data['trait1'] * 0.7 + np.random.normal(0, 0.5, n_samples)
        # Add some missing values
        data['trait3'][90:] = np.nan
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_gene_expression(self):
        """Create sample gene expression data."""
        np.random.seed(42)
        return pd.Series(np.random.normal(5, 2, 100), name='gene1')
    
    def test_init(self):
        """Test StatisticalAnalyzer initialization."""
        analyzer = StatisticalAnalyzer()
        assert analyzer.multiple_testing_method == 'fdr_bh'
        
        analyzer = StatisticalAnalyzer(multiple_testing_method='bonferroni')
        assert analyzer.multiple_testing_method == 'bonferroni'
    
    def test_calculate_trait_correlations_pearson(self, analyzer, sample_trait_data):
        """Test Pearson correlation calculation."""
        corr_matrix, p_matrix = analyzer.calculate_trait_correlations(
            sample_trait_data, method='pearson'
        )
        
        # Check matrix properties
        assert corr_matrix.shape == (3, 3)
        assert p_matrix.shape == (3, 3)
        assert np.allclose(corr_matrix.values.diagonal(), 1.0)
        assert corr_matrix.equals(corr_matrix.T)
        
        # Check correlation between trait1 and trait2 is strong
        assert corr_matrix.loc['trait1', 'trait2'] > 0.7
        assert p_matrix.loc['trait1', 'trait2'] < 0.001
    
    def test_calculate_trait_correlations_spearman(self, analyzer, sample_trait_data):
        """Test Spearman correlation calculation."""
        corr_matrix, p_matrix = analyzer.calculate_trait_correlations(
            sample_trait_data, method='spearman'
        )
        
        assert corr_matrix.shape == (3, 3)
        assert p_matrix.shape == (3, 3)
        assert np.allclose(corr_matrix.values.diagonal(), 1.0)
    
    def test_calculate_trait_correlations_min_samples(self, analyzer):
        """Test correlation with minimum samples requirement."""
        # Create data with very few samples
        data = pd.DataFrame({
            'trait1': [1, 2, 3],
            'trait2': [2, 3, 4]
        })
        
        corr_matrix, p_matrix = analyzer.calculate_trait_correlations(
            data, min_samples=5
        )
        
        # Should return NaN for correlations with insufficient samples
        assert np.isnan(corr_matrix.loc['trait1', 'trait2'])
        assert np.isnan(p_matrix.loc['trait1', 'trait2'])
    
    def test_calculate_trait_correlations_invalid_method(self, analyzer, sample_trait_data):
        """Test with invalid correlation method."""
        with pytest.raises(ValueError, match="Unknown correlation method"):
            analyzer.calculate_trait_correlations(sample_trait_data, method='invalid')
    
    def test_test_gene_trait_association_continuous(self, analyzer, sample_gene_expression):
        """Test gene-trait association for continuous traits."""
        # Create correlated trait
        trait = sample_gene_expression * 0.5 + pd.Series(np.random.normal(0, 1, 100))
        
        results = analyzer.test_gene_trait_association(
            sample_gene_expression, trait, test_type='continuous'
        )
        
        assert results['test'] == 'pearson_correlation'
        assert 'correlation' in results
        assert 'p_value' in results
        assert results['n_samples'] == 100
        assert results['p_value'] < 0.05  # Should be significant
    
    def test_test_gene_trait_association_categorical_binary(self, analyzer, sample_gene_expression):
        """Test gene-trait association for binary categorical traits."""
        # Create binary trait with difference in gene expression
        trait = pd.Series(['A'] * 50 + ['B'] * 50)
        gene_expr = pd.concat([
            pd.Series(np.random.normal(5, 1, 50)),
            pd.Series(np.random.normal(7, 1, 50))
        ])
        
        results = analyzer.test_gene_trait_association(
            gene_expr, trait, test_type='categorical'
        )
        
        assert results['test'] == 't_test'
        assert 't_statistic' in results
        assert 'p_value' in results
        assert 'group1_mean' in results
        assert 'group2_mean' in results
    
    def test_test_gene_trait_association_categorical_multiclass(self, analyzer):
        """Test gene-trait association for multi-class categorical traits."""
        # Create multi-class trait
        trait = pd.Series(['A'] * 30 + ['B'] * 30 + ['C'] * 40)
        gene_expr = pd.concat([
            pd.Series(np.random.normal(5, 1, 30)),
            pd.Series(np.random.normal(6, 1, 30)),
            pd.Series(np.random.normal(7, 1, 40))
        ])
        
        results = analyzer.test_gene_trait_association(
            gene_expr, trait, test_type='categorical'
        )
        
        assert results['test'] == 'anova'
        assert 'f_statistic' in results
        assert 'p_value' in results
        assert results['n_groups'] == 3
    
    def test_test_gene_trait_association_auto_detection(self, analyzer, sample_gene_expression):
        """Test automatic detection of test type."""
        # Continuous trait
        continuous_trait = pd.Series(np.random.normal(0, 1, 100))
        results = analyzer.test_gene_trait_association(
            sample_gene_expression, continuous_trait, test_type='auto'
        )
        assert results['test'] == 'pearson_correlation'
        
        # Categorical trait
        categorical_trait = pd.Series(['A', 'B', 'C'] * 33 + ['A'])
        results = analyzer.test_gene_trait_association(
            sample_gene_expression, categorical_trait, test_type='auto'
        )
        assert results['test'] == 'anova'
    
    def test_test_gene_trait_association_insufficient_data(self, analyzer):
        """Test with insufficient data."""
        gene_expr = pd.Series([1, 2])
        trait = pd.Series([1, 2])
        
        results = analyzer.test_gene_trait_association(gene_expr, trait)
        assert results['test'] == 'insufficient_data'
        assert np.isnan(results['p_value'])
    
    def test_calculate_pleiotropy_score_count(self, analyzer):
        """Test simple count-based pleiotropy scoring."""
        gene_trait_associations = {
            'gene1': ['trait1', 'trait2', 'trait3'],
            'gene2': ['trait1'],
            'gene3': ['trait1', 'trait2', 'trait3', 'trait4', 'trait5']
        }
        
        scores = analyzer.calculate_pleiotropy_score(
            gene_trait_associations, method='count'
        )
        
        assert scores['gene1'] == 3
        assert scores['gene2'] == 1
        assert scores['gene3'] == 5
    
    def test_calculate_pleiotropy_score_weighted(self, analyzer):
        """Test correlation-weighted pleiotropy scoring."""
        gene_trait_associations = {
            'gene1': ['trait1', 'trait2'],
            'gene2': ['trait1', 'trait3'],
            'gene3': ['trait1']
        }
        
        # Create correlation matrix
        trait_corr = pd.DataFrame({
            'trait1': [1.0, 0.8, 0.2],
            'trait2': [0.8, 1.0, 0.1],
            'trait3': [0.2, 0.1, 1.0]
        }, index=['trait1', 'trait2', 'trait3'])
        
        scores = analyzer.calculate_pleiotropy_score(
            gene_trait_associations, trait_corr, method='count_weighted'
        )
        
        # gene1 has highly correlated traits, so lower weighted score
        assert scores['gene1'] < 2
        # gene2 has less correlated traits, so higher weighted score
        assert scores['gene2'] > scores['gene1']
        # gene3 has single trait
        assert scores['gene3'] == 1
    
    def test_calculate_pleiotropy_score_entropy(self, analyzer):
        """Test entropy-based pleiotropy scoring."""
        gene_trait_associations = {
            'gene1': ['trait1', 'trait2'],
            'gene2': ['trait1', 'trait2', 'trait3', 'trait4'],
            'gene3': []
        }
        
        scores = analyzer.calculate_pleiotropy_score(
            gene_trait_associations, method='entropy'
        )
        
        assert scores['gene1'] > 0
        assert scores['gene2'] > scores['gene1']  # More traits = higher entropy
        assert scores['gene3'] == 0  # No traits = 0 entropy
    
    def test_calculate_pleiotropy_score_invalid_method(self, analyzer):
        """Test with invalid scoring method."""
        gene_trait_associations = {'gene1': ['trait1']}
        
        with pytest.raises(ValueError, match="Unknown scoring method"):
            analyzer.calculate_pleiotropy_score(
                gene_trait_associations, method='invalid'
            )
    
    def test_perform_pca_analysis(self, analyzer, sample_trait_data):
        """Test PCA analysis."""
        results = analyzer.perform_pca_analysis(sample_trait_data, n_components=2)
        
        assert 'transformed_data' in results
        assert 'loadings' in results
        assert 'explained_variance' in results
        assert 'explained_variance_ratio' in results
        assert 'cumulative_variance_ratio' in results
        
        # Check dimensions
        assert results['transformed_data'].shape[1] == 2
        assert results['loadings'].shape == (3, 2)
        assert len(results['explained_variance']) == 2
        
        # Check variance properties
        assert np.all(results['explained_variance_ratio'] >= 0)
        assert np.all(results['explained_variance_ratio'] <= 1)
        assert results['cumulative_variance_ratio'][-1] <= 1.0
    
    def test_perform_pca_analysis_no_standardization(self, analyzer, sample_trait_data):
        """Test PCA without standardization."""
        results = analyzer.perform_pca_analysis(
            sample_trait_data, standardize=False
        )
        
        assert 'transformed_data' in results
        assert results['n_samples'] == 90  # 10 samples have NaN in trait3
    
    def test_cluster_traits_kmeans(self, analyzer, sample_trait_data):
        """Test k-means clustering of traits."""
        results = analyzer.cluster_traits(
            sample_trait_data, method='kmeans', n_clusters=2
        )
        
        assert 'labels' in results
        assert 'cluster_centers' in results
        assert 'inertia' in results
        assert 'trait_clusters' in results
        
        assert len(results['labels']) == 3  # 3 traits
        assert results['n_clusters'] == 2
        assert len(results['trait_clusters']) == 3
    
    def test_cluster_traits_kmeans_auto_k(self, analyzer, sample_trait_data):
        """Test k-means with automatic k selection."""
        results = analyzer.cluster_traits(
            sample_trait_data, method='kmeans', n_clusters=None
        )
        
        assert 'n_clusters' in results
        assert results['n_clusters'] >= 2
        assert results['n_clusters'] <= 10
    
    def test_cluster_traits_dbscan(self, analyzer, sample_trait_data):
        """Test DBSCAN clustering of traits."""
        results = analyzer.cluster_traits(
            sample_trait_data, method='dbscan', eps=1.5, min_samples=1
        )
        
        assert 'labels' in results
        assert 'n_clusters' in results
        assert 'n_noise' in results
        assert 'trait_clusters' in results
    
    def test_find_optimal_k(self, analyzer):
        """Test optimal k finding for clustering."""
        # Create simple clusterable data
        np.random.seed(42)
        data = np.vstack([
            np.random.normal(0, 0.5, (20, 2)),
            np.random.normal(5, 0.5, (20, 2)),
            np.random.normal(10, 0.5, (20, 2))
        ])
        
        optimal_k = analyzer._find_optimal_k(data, max_k=6)
        assert 2 <= optimal_k <= 4  # Should find approximately 3 clusters
    
    def test_test_enrichment(self, analyzer):
        """Test pathway enrichment analysis."""
        gene_set = ['gene1', 'gene2', 'gene3', 'gene4']
        background_genes = ['gene' + str(i) for i in range(1, 101)]
        pathway_genes = {
            'pathway1': ['gene1', 'gene2', 'gene5', 'gene6'],
            'pathway2': ['gene10', 'gene11', 'gene12'],
            'pathway3': ['gene1', 'gene2', 'gene3', 'gene4', 'gene5']
        }
        
        results = analyzer.test_enrichment(gene_set, background_genes, pathway_genes)
        
        assert isinstance(results, pd.DataFrame)
        assert 'pathway' in results.columns
        assert 'p_value' in results.columns
        assert 'p_adjusted' in results.columns
        assert 'odds_ratio' in results.columns
        
        # pathway3 should be most enriched (4/5 genes in set)
        assert results.iloc[0]['pathway'] == 'pathway3'
        assert results.iloc[0]['n_overlap'] == 4
    
    def test_test_enrichment_empty_pathways(self, analyzer):
        """Test enrichment with empty pathways."""
        gene_set = ['gene1', 'gene2']
        background_genes = ['gene1', 'gene2', 'gene3']
        pathway_genes = {}
        
        results = analyzer.test_enrichment(gene_set, background_genes, pathway_genes)
        assert len(results) == 0
    
    def test_calculate_trait_heritability(self, analyzer, sample_trait_data):
        """Test heritability calculation."""
        heritability = analyzer.calculate_trait_heritability(sample_trait_data)
        
        assert isinstance(heritability, dict)
        assert len(heritability) == 3
        assert all(0 <= h2 <= 1 for h2 in heritability.values())
    
    def test_calculate_trait_heritability_with_kinship(self, analyzer, sample_trait_data):
        """Test heritability calculation with kinship matrix."""
        n_samples = len(sample_trait_data)
        # Create mock kinship matrix
        kinship = np.eye(n_samples) * 0.5 + np.random.rand(n_samples, n_samples) * 0.1
        kinship = (kinship + kinship.T) / 2  # Make symmetric
        
        with pytest.warns(UserWarning, match="simplified heritability estimation"):
            heritability = analyzer.calculate_trait_heritability(
                sample_trait_data, kinship_matrix=kinship
            )
        
        assert isinstance(heritability, dict)
        assert all(0.2 <= h2 <= 0.8 for h2 in heritability.values())


class TestStatisticalAnalyzerEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def analyzer(self):
        return StatisticalAnalyzer()
    
    def test_correlations_with_all_nan(self, analyzer):
        """Test correlation calculation with all NaN column."""
        data = pd.DataFrame({
            'trait1': [1, 2, 3, 4, 5],
            'trait2': [np.nan] * 5
        })
        
        corr_matrix, p_matrix = analyzer.calculate_trait_correlations(data)
        assert np.isnan(corr_matrix.loc['trait1', 'trait2'])
        assert np.isnan(p_matrix.loc['trait1', 'trait2'])
    
    def test_correlations_with_single_value(self, analyzer):
        """Test correlation with constant values."""
        data = pd.DataFrame({
            'trait1': [1, 1, 1, 1, 1],
            'trait2': [1, 2, 3, 4, 5]
        })
        
        # This should handle the case where standard deviation is 0
        corr_matrix, p_matrix = analyzer.calculate_trait_correlations(data)
        # The result might be NaN or raise a warning, both are acceptable
    
    def test_pca_with_insufficient_samples(self, analyzer):
        """Test PCA with fewer samples than features."""
        data = pd.DataFrame({
            'trait1': [1, 2],
            'trait2': [2, 3],
            'trait3': [3, 4],
            'trait4': [4, 5]
        })
        
        results = analyzer.perform_pca_analysis(data)
        # Should handle gracefully, returning fewer components
        assert results['n_components'] <= 2
    
    def test_clustering_with_single_trait(self, analyzer):
        """Test clustering with only one trait."""
        data = pd.DataFrame({'trait1': [1, 2, 3, 4, 5]})
        
        # Should handle single feature clustering
        results = analyzer.cluster_traits(data, method='kmeans', n_clusters=2)
        assert len(results['labels']) == 1


class TestStatisticalAnalyzerPerformance:
    """Performance tests for large datasets."""
    
    @pytest.fixture
    def analyzer(self):
        return StatisticalAnalyzer()
    
    @pytest.fixture
    def large_trait_data(self):
        """Create large dataset for performance testing."""
        np.random.seed(42)
        n_samples = 1000
        n_traits = 50
        data = {}
        for i in range(n_traits):
            data[f'trait{i}'] = np.random.normal(0, 1, n_samples)
        return pd.DataFrame(data)
    
    @pytest.mark.slow
    def test_correlation_performance(self, analyzer, large_trait_data, benchmark):
        """Benchmark correlation calculation on large dataset."""
        result = benchmark(analyzer.calculate_trait_correlations, large_trait_data)
        corr_matrix, p_matrix = result
        assert corr_matrix.shape == (50, 50)
    
    @pytest.mark.slow
    def test_pca_performance(self, analyzer, large_trait_data, benchmark):
        """Benchmark PCA on large dataset."""
        result = benchmark(analyzer.perform_pca_analysis, large_trait_data)
        assert result['n_components'] <= 50
    
    @pytest.mark.slow
    def test_clustering_performance(self, analyzer, large_trait_data, benchmark):
        """Benchmark clustering on large dataset."""
        result = benchmark(analyzer.cluster_traits, large_trait_data, 
                         method='kmeans', n_clusters=5)
        assert result['n_clusters'] == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])