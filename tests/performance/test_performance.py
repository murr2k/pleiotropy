"""
Performance tests for the Genomic Pleiotropy Cryptanalysis system.

Tests system performance with large datasets and measures:
- Processing speed
- Memory usage
- Scalability
- Database query performance
"""

import pytest
import time
import psutil
import numpy as np
import pandas as pd
from memory_profiler import profile
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from python_analysis.statistical_analyzer import StatisticalAnalyzer
from python_analysis.trait_visualizer import TraitVisualizer
from tests.fixtures.test_data_generator import TestDataGenerator


class TestPerformanceStatisticalAnalyzer:
    """Performance tests for StatisticalAnalyzer."""
    
    @pytest.fixture
    def large_trait_data(self):
        """Generate large trait dataset."""
        np.random.seed(42)
        n_samples = 10000
        n_traits = 100
        
        data = {}
        for i in range(n_traits):
            data[f'trait_{i}'] = np.random.normal(0, 1, n_samples)
            # Add some missing values
            if i % 5 == 0:
                data[f'trait_{i}'][np.random.choice(n_samples, 100, replace=False)] = np.nan
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def analyzer(self):
        return StatisticalAnalyzer()
    
    @pytest.mark.benchmark(group="correlation")
    def test_correlation_performance(self, analyzer, large_trait_data, benchmark):
        """Benchmark correlation calculation on large dataset."""
        def calculate_correlations():
            return analyzer.calculate_trait_correlations(large_trait_data)
        
        result = benchmark(calculate_correlations)
        corr_matrix, p_matrix = result
        
        assert corr_matrix.shape == (100, 100)
        assert p_matrix.shape == (100, 100)
    
    @pytest.mark.benchmark(group="pca")
    def test_pca_performance(self, analyzer, large_trait_data, benchmark):
        """Benchmark PCA on large dataset."""
        def perform_pca():
            return analyzer.perform_pca_analysis(large_trait_data, n_components=10)
        
        result = benchmark(perform_pca)
        assert result['n_components'] == 10
    
    @pytest.mark.benchmark(group="clustering")
    def test_clustering_performance(self, analyzer, benchmark):
        """Benchmark trait clustering."""
        # Use smaller dataset for clustering (transpose operation)
        n_samples = 1000
        n_traits = 50
        data = pd.DataFrame(
            np.random.randn(n_samples, n_traits),
            columns=[f'trait_{i}' for i in range(n_traits)]
        )
        
        def cluster_traits():
            return analyzer.cluster_traits(data, method='kmeans', n_clusters=10)
        
        result = benchmark(cluster_traits)
        assert result['n_clusters'] == 10
    
    @pytest.mark.benchmark(group="enrichment")
    def test_enrichment_performance(self, analyzer, benchmark):
        """Benchmark enrichment analysis."""
        # Generate test data
        all_genes = [f'gene_{i}' for i in range(10000)]
        gene_set = all_genes[:500]  # 500 genes in set
        
        # Generate 100 pathways
        pathway_genes = {}
        for i in range(100):
            pathway_size = np.random.randint(20, 200)
            pathway_genes[f'pathway_{i}'] = np.random.choice(
                all_genes, pathway_size, replace=False
            ).tolist()
        
        def test_enrichment():
            return analyzer.test_enrichment(gene_set, all_genes, pathway_genes)
        
        result = benchmark(test_enrichment)
        assert len(result) == 100
    
    @pytest.mark.slow
    def test_memory_usage_correlation(self, analyzer):
        """Test memory usage during correlation calculation."""
        # Create very large dataset
        n_samples = 5000
        n_traits = 200
        
        data = pd.DataFrame(
            np.random.randn(n_samples, n_traits),
            columns=[f'trait_{i}' for i in range(n_traits)]
        )
        
        # Monitor memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run analysis
        corr_matrix, p_matrix = analyzer.calculate_trait_correlations(data)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        print(f"Memory increase: {memory_increase:.2f} MB")
        assert memory_increase < 1000  # Should use less than 1GB
    
    @pytest.mark.slow
    def test_scalability_traits(self, analyzer, tmp_path):
        """Test scalability with increasing number of traits."""
        results = []
        n_samples = 1000
        
        for n_traits in [10, 50, 100, 200, 500]:
            data = pd.DataFrame(
                np.random.randn(n_samples, n_traits),
                columns=[f'trait_{i}' for i in range(n_traits)]
            )
            
            start_time = time.time()
            analyzer.calculate_trait_correlations(data)
            elapsed = time.time() - start_time
            
            results.append({
                'n_traits': n_traits,
                'time': elapsed,
                'time_per_pair': elapsed / (n_traits * (n_traits - 1) / 2)
            })
            
            print(f"Traits: {n_traits}, Time: {elapsed:.2f}s")
        
        # Save results
        with open(tmp_path / 'scalability_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Check that time scales appropriately (roughly O(n²))
        assert results[-1]['time'] < results[0]['time'] * 100


class TestPerformanceVisualization:
    """Performance tests for visualization components."""
    
    @pytest.fixture
    def visualizer(self):
        return TraitVisualizer()
    
    @pytest.fixture
    def large_gene_network(self):
        """Generate large gene-trait network."""
        associations = {}
        n_genes = 1000
        n_traits = 50
        
        all_traits = [f'trait_{i}' for i in range(n_traits)]
        
        for i in range(n_genes):
            n_associated = np.random.poisson(3) + 1
            associations[f'gene_{i}'] = np.random.choice(
                all_traits, 
                min(n_associated, n_traits),
                replace=False
            ).tolist()
        
        return associations
    
    @pytest.mark.benchmark(group="visualization")
    def test_heatmap_performance(self, visualizer, benchmark):
        """Benchmark heatmap generation."""
        # Medium-sized dataset
        data = pd.DataFrame(
            np.random.randn(500, 50),
            columns=[f'trait_{i}' for i in range(50)]
        )
        
        def create_heatmap():
            fig = visualizer.plot_trait_correlation_heatmap(data)
            plt.close(fig)
            return fig
        
        benchmark(create_heatmap)
    
    @pytest.mark.benchmark(group="visualization")
    def test_network_performance(self, visualizer, large_gene_network, benchmark):
        """Benchmark network visualization."""
        # Use subset for visualization performance
        subset = dict(list(large_gene_network.items())[:100])
        
        def create_network():
            fig = visualizer.plot_gene_trait_network(subset)
            plt.close(fig)
            return fig
        
        benchmark(create_network)
    
    @pytest.mark.slow
    def test_interactive_visualization_performance(self, visualizer):
        """Test performance of interactive visualizations."""
        # Large dataset
        data = pd.DataFrame(
            np.random.randn(1000, 100),
            columns=[f'trait_{i}' for i in range(100)]
        )
        
        start_time = time.time()
        fig = visualizer.create_interactive_heatmap(data)
        elapsed = time.time() - start_time
        
        print(f"Interactive heatmap creation: {elapsed:.2f}s")
        assert elapsed < 10  # Should complete within 10 seconds
        
        # Check figure size
        assert len(fig.data) == 1
        assert fig.data[0].z.shape == (100, 100)


class TestPerformanceIntegration:
    """Integration performance tests."""
    
    @pytest.fixture
    def data_generator(self):
        return TestDataGenerator(seed=42)
    
    @pytest.mark.slow
    @pytest.mark.benchmark(group="integration")
    def test_full_pipeline_performance(self, data_generator, benchmark, tmp_path):
        """Test full analysis pipeline performance."""
        # Generate test genome
        genome = data_generator.generate_genome(n_genes=100)
        
        # Save to FASTA
        fasta_path = tmp_path / "test_genome.fasta"
        with open(fasta_path, 'w') as f:
            for gene in genome:
                f.write(f">{gene['id']}\n{gene['sequence']}\n")
        
        def run_pipeline():
            # 1. Generate frequency table
            freq_table = data_generator.generate_frequency_table(genome)
            
            # 2. Create trait data
            trait_data = pd.DataFrame({
                trait: np.random.randn(100)
                for trait in ['growth', 'stress', 'motility']
            })
            
            # 3. Run statistical analysis
            analyzer = StatisticalAnalyzer()
            corr_matrix, _ = analyzer.calculate_trait_correlations(trait_data)
            pca_results = analyzer.perform_pca_analysis(trait_data)
            
            # 4. Create visualizations
            visualizer = TraitVisualizer()
            fig1 = visualizer.plot_trait_correlation_heatmap(trait_data)
            plt.close(fig1)
            
            return {
                'freq_table': freq_table,
                'correlations': corr_matrix,
                'pca': pca_results
            }
        
        result = benchmark(run_pipeline)
        assert 'freq_table' in result
        assert 'correlations' in result
        assert 'pca' in result
    
    @pytest.mark.slow
    def test_database_query_performance(self, data_generator, tmp_path):
        """Test database query performance with large dataset."""
        # Generate large trial dataset
        trials = data_generator.generate_trial_data(n_trials=10000)
        
        # Simulate database queries
        start_time = time.time()
        
        # Query 1: Filter by organism
        organism_trials = [t for t in trials if t['organism'] == 'E. coli K-12']
        
        # Query 2: Filter by date range
        from datetime import datetime, timedelta
        cutoff_date = (datetime.now() - timedelta(days=180)).isoformat()
        recent_trials = [t for t in trials if t['experiment_date'] > cutoff_date]
        
        # Query 3: Complex query - high confidence trials with specific traits
        complex_results = []
        for trial in trials:
            if trial['status'] == 'validated':
                for trait, score in trial['confidence_scores'].items():
                    if score > 0.9:
                        complex_results.append(trial)
                        break
        
        elapsed = time.time() - start_time
        
        print(f"Database query simulation time: {elapsed:.2f}s")
        print(f"Organism filter: {len(organism_trials)} results")
        print(f"Date filter: {len(recent_trials)} results")
        print(f"Complex query: {len(complex_results)} results")
        
        assert elapsed < 5  # Should complete within 5 seconds
    
    @pytest.mark.slow
    def test_concurrent_analysis(self):
        """Test concurrent analysis of multiple datasets."""
        import concurrent.futures
        
        def analyze_dataset(dataset_id):
            """Analyze a single dataset."""
            np.random.seed(dataset_id)
            
            # Generate data
            data = pd.DataFrame(
                np.random.randn(1000, 20),
                columns=[f'trait_{i}' for i in range(20)]
            )
            
            # Run analysis
            analyzer = StatisticalAnalyzer()
            corr_matrix, _ = analyzer.calculate_trait_correlations(data)
            pca_results = analyzer.perform_pca_analysis(data, n_components=5)
            
            return {
                'dataset_id': dataset_id,
                'max_correlation': corr_matrix.abs().max().max(),
                'explained_variance': pca_results['explained_variance_ratio'].sum()
            }
        
        # Run concurrent analyses
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(analyze_dataset, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        elapsed = time.time() - start_time
        
        print(f"Concurrent analysis time: {elapsed:.2f}s for 20 datasets")
        assert len(results) == 20
        assert elapsed < 30  # Should scale well with concurrency


class TestPerformanceMemoryProfiling:
    """Detailed memory profiling tests."""
    
    @pytest.mark.slow
    @profile
    def test_memory_profile_full_analysis(self):
        """Profile memory usage during full analysis."""
        # Generate large dataset
        n_samples = 5000
        n_traits = 100
        
        data = pd.DataFrame(
            np.random.randn(n_samples, n_traits),
            columns=[f'trait_{i}' for i in range(n_traits)]
        )
        
        analyzer = StatisticalAnalyzer()
        
        # Step 1: Correlation analysis
        corr_matrix, p_matrix = analyzer.calculate_trait_correlations(data)
        
        # Step 2: PCA
        pca_results = analyzer.perform_pca_analysis(data, n_components=20)
        
        # Step 3: Clustering
        cluster_results = analyzer.cluster_traits(data, method='kmeans', n_clusters=10)
        
        # Clean up
        del corr_matrix, p_matrix, pca_results, cluster_results
        
        print("Memory profiling complete")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--benchmark-only'])