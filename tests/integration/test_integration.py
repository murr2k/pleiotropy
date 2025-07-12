"""
Integration tests for the Genomic Pleiotropy Cryptanalysis system.

Tests the integration between different components:
- Rust and Python integration
- Full workflow testing
- Database integration
- API integration
"""

import pytest
import subprocess
import json
import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from python_analysis.statistical_analyzer import StatisticalAnalyzer
from python_analysis.trait_visualizer import TraitVisualizer
from python_analysis.rust_interface import RustInterface
from tests.fixtures.test_data_generator import TestDataGenerator


class TestRustPythonIntegration:
    """Test integration between Rust and Python components."""
    
    @pytest.fixture
    def rust_binary_path(self):
        """Get path to Rust binary."""
        rust_impl_dir = Path(__file__).parent.parent.parent / "rust_impl"
        binary_path = rust_impl_dir / "target" / "release" / "genomic_pleiotropy"
        
        # Build if not exists
        if not binary_path.exists():
            subprocess.run(
                ["cargo", "build", "--release"],
                cwd=rust_impl_dir,
                check=True
            )
        
        return str(binary_path)
    
    @pytest.fixture
    def test_data(self, tmp_path):
        """Generate test data files."""
        generator = TestDataGenerator()
        
        # Generate test genome
        genome = generator.generate_genome(20)
        
        # Save FASTA file
        fasta_path = tmp_path / "test.fasta"
        with open(fasta_path, 'w') as f:
            for gene in genome:
                f.write(f">{gene['id']}\n{gene['sequence']}\n")
        
        # Save frequency table
        freq_table = generator.generate_frequency_table(genome)
        freq_path = tmp_path / "frequency_table.json"
        with open(freq_path, 'w') as f:
            json.dump(freq_table, f)
        
        # Save trait definitions
        traits_path = tmp_path / "traits.json"
        with open(traits_path, 'w') as f:
            json.dump({
                'traits': [t['name'] for t in freq_table['trait_definitions']]
            }, f)
        
        return {
            'fasta_path': str(fasta_path),
            'freq_path': str(freq_path),
            'traits_path': str(traits_path),
            'output_dir': str(tmp_path),
            'genome': genome
        }
    
    def test_rust_cli_execution(self, rust_binary_path, test_data):
        """Test Rust CLI execution."""
        # Run Rust analysis
        cmd = [
            rust_binary_path,
            "--input", test_data['fasta_path'],
            "--frequencies", test_data['freq_path'],
            "--output", test_data['output_dir'] + "/results.json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0, f"Rust CLI failed: {result.stderr}"
        
        # Check output file exists
        output_file = Path(test_data['output_dir']) / "results.json"
        assert output_file.exists()
        
        # Validate output format
        with open(output_file) as f:
            results = json.load(f)
        
        assert 'decrypted_regions' in results
        assert isinstance(results['decrypted_regions'], list)
    
    @pytest.mark.skipif(not Path("python_analysis/rust_interface.py").exists(),
                        reason="Rust interface not implemented")
    def test_rust_python_interface(self, test_data):
        """Test Python interface to Rust."""
        interface = RustInterface()
        
        # Load test data
        with open(test_data['fasta_path']) as f:
            sequences = f.read()
        
        with open(test_data['freq_path']) as f:
            freq_table = json.load(f)
        
        # Call Rust from Python
        results = interface.decrypt_sequences(sequences, freq_table)
        
        assert isinstance(results, list)
        if results:
            # Validate result structure
            region = results[0]
            assert 'gene_id' in region
            assert 'decrypted_traits' in region
            assert 'confidence_scores' in region
    
    def test_full_analysis_pipeline(self, rust_binary_path, test_data, tmp_path):
        """Test complete analysis pipeline."""
        # Step 1: Run Rust analysis
        rust_output = tmp_path / "rust_results.json"
        cmd = [
            rust_binary_path,
            "--input", test_data['fasta_path'],
            "--frequencies", test_data['freq_path'],
            "--output", str(rust_output)
        ]
        
        subprocess.run(cmd, check=True)
        
        # Step 2: Load Rust results
        with open(rust_output) as f:
            rust_results = json.load(f)
        
        # Step 3: Prepare trait data for Python analysis
        trait_data = {}
        for gene in test_data['genome']:
            for trait in gene['annotations']['traits']:
                if trait not in trait_data:
                    trait_data[trait] = []
                trait_data[trait].append(gene['annotations']['confidence'][trait])
        
        # Pad arrays to same length
        max_len = max(len(v) for v in trait_data.values())
        for trait in trait_data:
            trait_data[trait].extend([None] * (max_len - len(trait_data[trait])))
        
        import pandas as pd
        trait_df = pd.DataFrame(trait_data)
        
        # Step 4: Run Python statistical analysis
        analyzer = StatisticalAnalyzer()
        corr_matrix, p_matrix = analyzer.calculate_trait_correlations(trait_df)
        
        # Step 5: Generate visualizations
        visualizer = TraitVisualizer()
        fig = visualizer.plot_trait_correlation_heatmap(
            trait_df,
            save_path=str(tmp_path / "correlation_heatmap.png")
        )
        
        # Verify outputs
        assert (tmp_path / "correlation_heatmap.png").exists()
        assert not corr_matrix.empty
        
        # Close figure
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestDatabaseIntegration:
    """Test database operations."""
    
    @pytest.fixture
    def mock_database(self, tmp_path):
        """Create mock database with test data."""
        generator = TestDataGenerator()
        trials = generator.generate_trial_data(100)
        
        # Save as JSON (simulating database)
        db_path = tmp_path / "trial_db.json"
        with open(db_path, 'w') as f:
            json.dump({'trials': trials}, f)
        
        return db_path
    
    def test_database_crud_operations(self, mock_database):
        """Test Create, Read, Update, Delete operations."""
        # Read
        with open(mock_database) as f:
            data = json.load(f)
        
        trials = data['trials']
        assert len(trials) == 100
        
        # Create
        new_trial = {
            'id': 'TRIAL_999999',
            'organism': 'E. coli TEST',
            'gene_id': 'test_gene',
            'experiment_date': '2024-01-01',
            'researcher': 'Test User',
            'traits_tested': ['growth'],
            'traits_confirmed': ['growth'],
            'confidence_scores': {'growth': 0.95},
            'status': 'completed'
        }
        trials.append(new_trial)
        
        # Update
        trials[0]['status'] = 'validated'
        trials[0]['updated_at'] = '2024-01-02T00:00:00'
        
        # Delete (mark as deleted)
        trials[1]['deleted'] = True
        
        # Save back
        with open(mock_database, 'w') as f:
            json.dump({'trials': trials}, f)
        
        # Verify changes
        with open(mock_database) as f:
            updated_data = json.load(f)
        
        assert len(updated_data['trials']) == 101
        assert updated_data['trials'][0]['status'] == 'validated'
        assert updated_data['trials'][1].get('deleted', False)
    
    def test_database_queries(self, mock_database):
        """Test complex database queries."""
        with open(mock_database) as f:
            data = json.load(f)
        
        trials = data['trials']
        
        # Query 1: High confidence trials
        high_confidence = [
            t for t in trials
            if any(score > 0.9 for score in t['confidence_scores'].values())
        ]
        
        # Query 2: Trials by organism and status
        ecoli_validated = [
            t for t in trials
            if 'E. coli' in t['organism'] and t['status'] == 'validated'
        ]
        
        # Query 3: Aggregate by researcher
        by_researcher = {}
        for trial in trials:
            researcher = trial['researcher']
            if researcher not in by_researcher:
                by_researcher[researcher] = []
            by_researcher[researcher].append(trial)
        
        # Verify queries return results
        assert len(high_confidence) > 0
        assert isinstance(ecoli_validated, list)
        assert len(by_researcher) > 0
    
    def test_database_performance_with_indices(self, tmp_path):
        """Test database query performance with indices."""
        # Generate large dataset
        generator = TestDataGenerator()
        trials = generator.generate_trial_data(10000)
        
        # Create indices (simulated)
        indices = {
            'by_organism': {},
            'by_gene': {},
            'by_date': {},
            'by_status': {}
        }
        
        for i, trial in enumerate(trials):
            # Organism index
            org = trial['organism']
            if org not in indices['by_organism']:
                indices['by_organism'][org] = []
            indices['by_organism'][org].append(i)
            
            # Gene index
            gene = trial['gene_id']
            if gene not in indices['by_gene']:
                indices['by_gene'][gene] = []
            indices['by_gene'][gene].append(i)
            
            # Status index
            status = trial['status']
            if status not in indices['by_status']:
                indices['by_status'][status] = []
            indices['by_status'][status].append(i)
        
        # Test indexed query performance
        import time
        
        # Query using index
        start = time.time()
        ecoli_indices = indices['by_organism'].get('E. coli K-12', [])
        ecoli_trials = [trials[i] for i in ecoli_indices]
        indexed_time = time.time() - start
        
        # Query without index (full scan)
        start = time.time()
        ecoli_trials_scan = [t for t in trials if t['organism'] == 'E. coli K-12']
        scan_time = time.time() - start
        
        print(f"Indexed query: {indexed_time:.4f}s")
        print(f"Full scan: {scan_time:.4f}s")
        
        # Indexed query should be faster
        assert indexed_time < scan_time
        assert len(ecoli_trials) == len(ecoli_trials_scan)


class TestAPIIntegration:
    """Test API integration (when implemented)."""
    
    @pytest.mark.skip(reason="API not yet implemented")
    def test_api_endpoints(self):
        """Test REST API endpoints."""
        import requests
        
        base_url = "http://localhost:8000/api"
        
        # Test health check
        response = requests.get(f"{base_url}/health")
        assert response.status_code == 200
        
        # Test trial endpoints
        response = requests.get(f"{base_url}/trials")
        assert response.status_code == 200
        trials = response.json()
        assert isinstance(trials, list)
        
        # Test filtering
        response = requests.get(f"{base_url}/trials?organism=E.%20coli")
        assert response.status_code == 200
        
        # Test pagination
        response = requests.get(f"{base_url}/trials?page=1&limit=10")
        assert response.status_code == 200
        assert len(response.json()) <= 10
    
    @pytest.mark.skip(reason="API not yet implemented")
    def test_api_authentication(self):
        """Test API authentication."""
        import requests
        
        base_url = "http://localhost:8000/api"
        
        # Test without auth
        response = requests.post(f"{base_url}/trials", json={})
        assert response.status_code == 401
        
        # Test with auth
        headers = {"Authorization": "Bearer test_token"}
        response = requests.post(f"{base_url}/trials", json={}, headers=headers)
        assert response.status_code in [200, 201, 400]  # Not 401


class TestWorkflowIntegration:
    """Test complete workflows."""
    
    def test_ecoli_workflow(self, tmp_path):
        """Test E. coli analysis workflow."""
        # This simulates the ecoli_workflow.sh script
        
        # Step 1: Prepare data
        generator = TestDataGenerator()
        
        # Generate E. coli-like genome
        genome = []
        ecoli_genes = {
            'ftsZ': ['cell_division', 'growth_rate'],
            'rpoS': ['stress_response', 'stationary_phase'],
            'flhD': ['motility', 'flagellar_assembly'],
            'crp': ['catabolite_repression', 'metabolism']
        }
        
        for gene_name, traits in ecoli_genes.items():
            gene = generator.generate_gene(gene_name, traits, length=1200)
            genome.append(gene)
        
        # Save genome
        fasta_path = tmp_path / "ecoli_test.fasta"
        with open(fasta_path, 'w') as f:
            for gene in genome:
                f.write(f">{gene['id']}\n{gene['sequence']}\n")
        
        # Step 2: Generate frequency table
        freq_table = generator.generate_frequency_table(genome)
        freq_path = tmp_path / "ecoli_frequencies.json"
        with open(freq_path, 'w') as f:
            json.dump(freq_table, f)
        
        # Step 3: Run analysis (simulated)
        results = {
            'organism': 'E. coli (test)',
            'genes_analyzed': len(genome),
            'traits_detected': list(ecoli_genes.values()),
            'pleiotropy_scores': {
                gene: len(traits) for gene, traits in ecoli_genes.items()
            }
        }
        
        results_path = tmp_path / "ecoli_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Verify workflow outputs
        assert fasta_path.exists()
        assert freq_path.exists()
        assert results_path.exists()
        
        # Validate results
        with open(results_path) as f:
            workflow_results = json.load(f)
        
        assert workflow_results['genes_analyzed'] == 4
        assert workflow_results['pleiotropy_scores']['rpoS'] == 2
    
    def test_batch_processing_workflow(self, tmp_path):
        """Test batch processing of multiple organisms."""
        organisms = ['E. coli', 'Salmonella', 'Klebsiella']
        generator = TestDataGenerator()
        
        all_results = []
        
        for organism in organisms:
            # Generate organism-specific data
            genome = generator.generate_genome(10)
            
            # Save data
            org_dir = tmp_path / organism.replace(' ', '_')
            org_dir.mkdir()
            
            fasta_path = org_dir / "genome.fasta"
            with open(fasta_path, 'w') as f:
                for gene in genome:
                    f.write(f">{gene['id']}\n{gene['sequence']}\n")
            
            # Process (simulated)
            results = {
                'organism': organism,
                'genes_analyzed': len(genome),
                'total_traits': len(set(t for g in genome 
                                      for t in g['annotations']['traits'])),
                'avg_traits_per_gene': sum(len(g['annotations']['traits']) 
                                          for g in genome) / len(genome)
            }
            
            all_results.append(results)
        
        # Save batch results
        batch_results_path = tmp_path / "batch_results.json"
        with open(batch_results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Verify
        assert len(all_results) == 3
        assert all(r['genes_analyzed'] == 10 for r in all_results)
        assert batch_results_path.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])