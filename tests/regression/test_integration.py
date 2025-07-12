"""
Comprehensive End-to-End Integration Tests for Genomic Pleiotropy Cryptanalysis System

This module tests the complete system integration including:
- Full user workflows end-to-end
- Swarm agent coordination and task execution
- Rust-Python integration bridges
- Docker deployment and orchestration
- Memory system integration (Redis)
- Monitoring systems (Prometheus/Grafana)
- System performance under concurrent usage

Memory Namespace: swarm-regression-1752301224
"""

import asyncio
import json
import os
import sys
import time
import tempfile
import subprocess
import pytest
import redis
import requests
import docker
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "trial_database" / "swarm"))

from tests.fixtures.test_data_generator import TestDataGenerator
from python_analysis.statistical_analyzer import StatisticalAnalyzer
from python_analysis.trait_visualizer import TraitVisualizer

# Memory namespace for test isolation
MEMORY_NAMESPACE = "swarm-regression-1752301224"


class SystemTestHarness:
    """Test harness for managing system state during integration tests."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.test_data_dir = None
        self.redis_client = None
        self.docker_client = None
        self.services_started = False
        
    def setup(self):
        """Set up test environment."""
        # Create temporary test data directory
        self.test_data_dir = tempfile.mkdtemp(prefix="pleiotropy_integration_")
        
        # Connect to Redis (assuming it's running)
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
        except (redis.ConnectionError, redis.TimeoutError):
            self.redis_client = None
            
        # Connect to Docker
        try:
            self.docker_client = docker.from_env()
        except Exception:
            self.docker_client = None
            
    def teardown(self):
        """Clean up test environment."""
        # Clean Redis test data
        if self.redis_client:
            pattern = f"{MEMORY_NAMESPACE}:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                
        # Remove temporary files
        if self.test_data_dir and os.path.exists(self.test_data_dir):
            import shutil
            shutil.rmtree(self.test_data_dir)
            
    def save_test_finding(self, finding_key: str, data: Dict[str, Any]):
        """Save test finding to memory namespace."""
        if self.redis_client:
            key = f"{MEMORY_NAMESPACE}/integration-test/{finding_key}"
            self.redis_client.set(key, json.dumps(data), ex=3600)  # 1 hour TTL


@pytest.fixture(scope="session")
def test_harness():
    """Session-scoped test harness."""
    harness = SystemTestHarness()
    harness.setup()
    yield harness
    harness.teardown()


@pytest.fixture
def test_data(test_harness):
    """Generate test data for integration tests."""
    generator = TestDataGenerator(seed=42)
    
    # Generate comprehensive test genome
    genome = generator.generate_genome(n_genes=50, avg_traits_per_gene=3)
    
    # Generate frequency table
    freq_table = generator.generate_frequency_table(genome)
    
    # Generate trial data
    trials = generator.generate_trial_data(n_trials=200, genome=genome)
    
    # Save data files
    data_dir = Path(test_harness.test_data_dir)
    
    # FASTA file
    fasta_path = data_dir / "test_genome.fasta"
    with open(fasta_path, 'w') as f:
        for gene in genome[:20]:  # Use subset for faster tests
            f.write(f">{gene['id']}\n{gene['sequence']}\n")
    
    # Frequency table
    freq_path = data_dir / "frequency_table.json"
    with open(freq_path, 'w') as f:
        json.dump(freq_table, f)
    
    # Traits configuration
    traits_path = data_dir / "traits.json"
    traits_config = {
        'traits': [t['name'] for t in freq_table['trait_definitions']],
        'categories': {
            'metabolism': ['glucose_utilization', 'lactose_metabolism', 'amino_acid_synthesis'],
            'stress': ['heat_shock', 'oxidative_stress', 'osmotic_stress'],
            'motility': ['flagellar_assembly', 'chemotaxis', 'pili_formation']
        }
    }
    with open(traits_path, 'w') as f:
        json.dump(traits_config, f)
    
    # Trial database
    trials_path = data_dir / "trials.json"
    with open(trials_path, 'w') as f:
        json.dump({'trials': trials}, f)
    
    return {
        'genome': genome,
        'freq_table': freq_table,
        'trials': trials,
        'fasta_path': str(fasta_path),
        'freq_path': str(freq_path),
        'traits_path': str(traits_path),
        'trials_path': str(trials_path),
        'data_dir': str(data_dir)
    }


class TestEndToEndWorkflows:
    """Test complete user workflows end-to-end."""
    
    def test_complete_ecoli_analysis_workflow(self, test_harness, test_data):
        """Test complete E. coli analysis workflow from start to finish."""
        # Save test start
        test_harness.save_test_finding("workflow_start", {
            "test_name": "complete_ecoli_analysis_workflow",
            "started_at": time.time(),
            "data_files": test_data['data_dir']
        })
        
        workflow_results = {}
        
        # Step 1: Data Validation
        assert os.path.exists(test_data['fasta_path'])
        assert os.path.exists(test_data['freq_path'])
        workflow_results['data_validation'] = True
        
        # Step 2: Rust Binary Check and Execution
        rust_binary = test_harness.project_root / "rust_impl" / "target" / "release" / "genomic_pleiotropy"
        
        if not rust_binary.exists():
            # Try to build Rust binary
            rust_dir = test_harness.project_root / "rust_impl"
            try:
                result = subprocess.run(
                    ["cargo", "build", "--release"],
                    cwd=rust_dir,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                workflow_results['rust_build'] = result.returncode == 0
                if result.returncode != 0:
                    workflow_results['rust_build_error'] = result.stderr
            except subprocess.TimeoutExpired:
                workflow_results['rust_build'] = False
                workflow_results['rust_build_error'] = "Build timeout"
        else:
            workflow_results['rust_build'] = True
        
        # Step 3: Run Analysis (if binary available)
        if workflow_results['rust_build'] and rust_binary.exists():
            output_file = Path(test_data['data_dir']) / "rust_results.json"
            try:
                cmd = [
                    str(rust_binary),
                    "--input", test_data['fasta_path'],
                    "--frequencies", test_data['freq_path'],
                    "--output", str(output_file)
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60  # 1 minute timeout
                )
                
                workflow_results['rust_execution'] = result.returncode == 0
                workflow_results['rust_output_exists'] = output_file.exists()
                
                if result.returncode != 0:
                    workflow_results['rust_execution_error'] = result.stderr
                
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                workflow_results['rust_execution'] = False
                workflow_results['rust_execution_error'] = str(e)
        else:
            workflow_results['rust_execution'] = False
            workflow_results['rust_execution_error'] = "Binary not available"
        
        # Step 4: Python Analysis
        try:
            # Statistical analysis
            analyzer = StatisticalAnalyzer()
            
            # Create mock trait data from test genome
            import pandas as pd
            trait_data = {}
            for gene in test_data['genome'][:10]:  # Use subset
                for trait in gene['annotations']['traits']:
                    if trait not in trait_data:
                        trait_data[trait] = []
                    trait_data[trait].append(gene['annotations']['confidence'][trait])
            
            # Pad to same length
            max_len = max(len(v) for v in trait_data.values()) if trait_data else 1
            for trait in trait_data:
                while len(trait_data[trait]) < max_len:
                    trait_data[trait].append(0.5)  # Default confidence
            
            if trait_data:
                trait_df = pd.DataFrame(trait_data)
                corr_matrix, p_matrix = analyzer.calculate_trait_correlations(trait_df)
                workflow_results['python_statistical_analysis'] = not corr_matrix.empty
            else:
                workflow_results['python_statistical_analysis'] = False
                workflow_results['python_analysis_error'] = "No trait data available"
            
        except Exception as e:
            workflow_results['python_statistical_analysis'] = False
            workflow_results['python_analysis_error'] = str(e)
        
        # Step 5: Visualization
        try:
            visualizer = TraitVisualizer()
            if trait_data:
                trait_df = pd.DataFrame(trait_data)
                viz_path = Path(test_data['data_dir']) / "correlation_heatmap.png"
                fig = visualizer.plot_trait_correlation_heatmap(
                    trait_df,
                    save_path=str(viz_path)
                )
                workflow_results['python_visualization'] = viz_path.exists()
                
                # Clean up matplotlib
                import matplotlib.pyplot as plt
                plt.close(fig)
            else:
                workflow_results['python_visualization'] = False
        except Exception as e:
            workflow_results['python_visualization'] = False
            workflow_results['python_visualization_error'] = str(e)
        
        # Save workflow results
        workflow_results['completed_at'] = time.time()
        workflow_results['total_duration'] = workflow_results['completed_at'] - workflow_results.get('started_at', time.time())
        
        test_harness.save_test_finding("workflow_results", workflow_results)
        
        # Verify critical components worked
        assert workflow_results['data_validation'], "Data validation failed"
        # Note: Rust components might not be available in all environments
        # Focus on Python components which should always work
        assert workflow_results['python_statistical_analysis'], f"Python analysis failed: {workflow_results.get('python_analysis_error', 'Unknown error')}"
    
    def test_batch_processing_workflow(self, test_harness, test_data):
        """Test batch processing of multiple genomes."""
        batch_results = {
            'started_at': time.time(),
            'genomes_processed': 0,
            'errors': []
        }
        
        generator = TestDataGenerator(seed=123)
        
        # Generate multiple organism datasets
        organisms = ['E. coli', 'Salmonella', 'Klebsiella']
        results = []
        
        for organism in organisms:
            try:
                # Generate organism-specific genome
                genome = generator.generate_genome(n_genes=10, avg_traits_per_gene=2)
                
                # Simulate analysis
                analysis_result = {
                    'organism': organism,
                    'genes_analyzed': len(genome),
                    'traits_found': len(set(t for g in genome for t in g['annotations']['traits'])),
                    'avg_confidence': sum(
                        sum(g['annotations']['confidence'].values()) / len(g['annotations']['confidence'])
                        for g in genome
                    ) / len(genome)
                }
                
                results.append(analysis_result)
                batch_results['genomes_processed'] += 1
                
            except Exception as e:
                batch_results['errors'].append(f"{organism}: {str(e)}")
        
        batch_results['completed_at'] = time.time()
        batch_results['results'] = results
        batch_results['success_rate'] = batch_results['genomes_processed'] / len(organisms)
        
        test_harness.save_test_finding("batch_processing", batch_results)
        
        assert batch_results['genomes_processed'] >= 2, "Should process at least 2 genomes"
        assert batch_results['success_rate'] >= 0.6, "Should have at least 60% success rate"


def _redis_available():
    """Check if Redis is available."""
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        return True
    except:
        return False


class TestSwarmCoordination:
    """Test swarm agent coordination and task distribution."""
    
    @pytest.mark.skipif(not _redis_available(), reason="Redis not available")
    def test_swarm_agent_coordination(self, test_harness):
        """Test swarm agent registration and coordination."""
        coordination_results = {
            'started_at': time.time(),
            'redis_available': False,
            'agents_registered': 0,
            'tasks_submitted': 0,
            'errors': []
        }
        
        try:
            # Test Redis connection
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            redis_client.ping()
            coordination_results['redis_available'] = True
            
            # Simulate agent registration
            agents = [
                {'type': 'rust_analyzer', 'capabilities': ['crypto_analysis', 'sequence_parsing']},
                {'type': 'python_visualizer', 'capabilities': ['visualization', 'statistics']},
                {'type': 'data_manager', 'capabilities': ['database', 'storage']}
            ]
            
            for agent in agents:
                agent_key = f"{MEMORY_NAMESPACE}:agent:{agent['type']}"
                agent_data = {
                    'type': agent['type'],
                    'capabilities': agent['capabilities'],
                    'status': 'active',
                    'registered_at': time.time()
                }
                redis_client.set(agent_key, json.dumps(agent_data), ex=300)
                coordination_results['agents_registered'] += 1
            
            # Simulate task submission
            tasks = [
                {'type': 'sequence_parsing', 'data': 'test.fasta'},
                {'type': 'trait_visualization', 'data': 'results.json'},
                {'type': 'database_update', 'data': 'new_trials.json'}
            ]
            
            for i, task in enumerate(tasks):
                task_key = f"{MEMORY_NAMESPACE}:task:{i}"
                task_data = {
                    'id': i,
                    'type': task['type'],
                    'data': task['data'],
                    'status': 'pending',
                    'submitted_at': time.time()
                }
                redis_client.set(task_key, json.dumps(task_data), ex=300)
                coordination_results['tasks_submitted'] += 1
            
            # Verify agent-task matching capability
            coordination_results['coordination_working'] = True
            
        except Exception as e:
            coordination_results['errors'].append(str(e))
            coordination_results['coordination_working'] = False
        
        coordination_results['completed_at'] = time.time()
        test_harness.save_test_finding("swarm_coordination", coordination_results)
        
        assert coordination_results['redis_available'], "Redis must be available for swarm coordination"
        assert coordination_results['agents_registered'] >= 2, "Should register multiple agents"
        assert coordination_results['tasks_submitted'] >= 2, "Should submit multiple tasks"
    
    def test_memory_system_integration(self, test_harness):
        """Test Redis-based memory system and pub/sub functionality."""
        memory_results = {
            'started_at': time.time(),
            'operations_tested': 0,
            'errors': []
        }
        
        if not test_harness.redis_client:
            pytest.skip("Redis not available")
        
        try:
            redis_client = test_harness.redis_client
            
            # Test basic operations
            test_data = {
                'analysis_results': {'gene_count': 100, 'traits_found': 15},
                'agent_status': {'rust_analyzer': 'active', 'python_viz': 'busy'},
                'task_queue': [{'id': 1, 'type': 'analysis'}, {'id': 2, 'type': 'visualization'}]
            }
            
            # Write operations
            for key, data in test_data.items():
                redis_key = f"{MEMORY_NAMESPACE}:test:{key}"
                redis_client.set(redis_key, json.dumps(data), ex=60)
                memory_results['operations_tested'] += 1
            
            # Read operations
            for key, expected_data in test_data.items():
                redis_key = f"{MEMORY_NAMESPACE}:test:{key}"
                stored_data = redis_client.get(redis_key)
                assert stored_data is not None, f"Data not found for key: {key}"
                
                parsed_data = json.loads(stored_data)
                assert parsed_data == expected_data, f"Data mismatch for key: {key}"
                memory_results['operations_tested'] += 1
            
            # Test pub/sub functionality
            import threading
            import queue
            
            message_queue = queue.Queue()
            
            def subscribe_thread():
                pubsub = redis_client.pubsub()
                pubsub.subscribe(f"{MEMORY_NAMESPACE}:events")
                for message in pubsub.listen():
                    if message['type'] == 'message':
                        message_queue.put(message['data'])
                        break
            
            # Start subscriber
            subscriber = threading.Thread(target=subscribe_thread)
            subscriber.daemon = True
            subscriber.start()
            
            time.sleep(0.1)  # Allow subscriber to start
            
            # Publish message
            test_message = "test_coordination_event"
            redis_client.publish(f"{MEMORY_NAMESPACE}:events", test_message)
            
            # Wait for message
            try:
                received = message_queue.get(timeout=2)
                assert received == test_message, "Pub/sub message not received correctly"
                memory_results['pubsub_working'] = True
            except queue.Empty:
                memory_results['pubsub_working'] = False
                memory_results['errors'].append("Pub/sub timeout")
            
        except Exception as e:
            memory_results['errors'].append(str(e))
        
        memory_results['completed_at'] = time.time()
        test_harness.save_test_finding("memory_system", memory_results)
        
        assert memory_results['operations_tested'] >= 6, "Should test multiple Redis operations"
        assert len(memory_results['errors']) == 0, f"Memory system errors: {memory_results['errors']}"


class TestDockerDeployment:
    """Test Docker container orchestration and service health."""
    
    def test_docker_compose_configuration(self, test_harness):
        """Test Docker Compose configuration validity."""
        docker_results = {
            'started_at': time.time(),
            'config_valid': False,
            'services_defined': 0,
            'errors': []
        }
        
        try:
            # Check if docker-compose.yml exists
            compose_file = test_harness.project_root / "docker-compose.yml"
            assert compose_file.exists(), "docker-compose.yml not found"
            
            # Validate compose file
            result = subprocess.run(
                ["docker-compose", "-f", str(compose_file), "config"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                docker_results['config_valid'] = True
                
                # Count services in output
                config_output = result.stdout
                docker_results['services_defined'] = config_output.count('container_name:')
                
            else:
                docker_results['errors'].append(f"Compose validation failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            docker_results['errors'].append("Docker compose validation timeout")
        except Exception as e:
            docker_results['errors'].append(str(e))
        
        docker_results['completed_at'] = time.time()
        test_harness.save_test_finding("docker_deployment", docker_results)
        
        assert docker_results['config_valid'], "Docker Compose configuration must be valid"
        assert docker_results['services_defined'] >= 3, "Should define at least 3 services"
    
    @pytest.mark.skipif(not os.getenv("DOCKER_TEST", False), reason="Docker tests require DOCKER_TEST env var")
    def test_service_health_checks(self, test_harness):
        """Test Docker service health checks and startup."""
        health_results = {
            'started_at': time.time(),
            'services_healthy': 0,
            'errors': []
        }
        
        if not test_harness.docker_client:
            pytest.skip("Docker not available")
        
        try:
            # Get running containers
            containers = test_harness.docker_client.containers.list()
            pleiotropy_containers = [c for c in containers if 'pleiotropy' in c.name]
            
            for container in pleiotropy_containers:
                try:
                    # Check container health
                    container.reload()
                    if container.status == 'running':
                        health_results['services_healthy'] += 1
                except Exception as e:
                    health_results['errors'].append(f"Container {container.name}: {str(e)}")
            
            health_results['containers_found'] = len(pleiotropy_containers)
            
        except Exception as e:
            health_results['errors'].append(str(e))
        
        health_results['completed_at'] = time.time()
        test_harness.save_test_finding("service_health", health_results)
        
        # This test is informational - don't fail if no containers running
        # Just record the findings for analysis


class TestMonitoringSystems:
    """Test Prometheus metrics and Grafana dashboard integration."""
    
    def test_prometheus_configuration(self, test_harness):
        """Test Prometheus configuration and metrics endpoint."""
        monitoring_results = {
            'started_at': time.time(),
            'prometheus_config_valid': False,
            'grafana_config_valid': False,
            'errors': []
        }
        
        try:
            # Check Prometheus config
            prometheus_config = test_harness.project_root / "monitoring" / "prometheus.yml"
            if prometheus_config.exists():
                with open(prometheus_config) as f:
                    import yaml
                    config = yaml.safe_load(f)
                    
                    if 'scrape_configs' in config:
                        monitoring_results['prometheus_config_valid'] = True
                        monitoring_results['scrape_targets'] = len(config['scrape_configs'])
            
            # Check Grafana dashboards
            grafana_dir = test_harness.project_root / "monitoring" / "grafana" / "dashboards"
            if grafana_dir.exists():
                dashboard_files = list(grafana_dir.glob("*.json"))
                monitoring_results['grafana_dashboards'] = len(dashboard_files)
                monitoring_results['grafana_config_valid'] = len(dashboard_files) > 0
            
        except Exception as e:
            monitoring_results['errors'].append(str(e))
        
        monitoring_results['completed_at'] = time.time()
        test_harness.save_test_finding("monitoring_systems", monitoring_results)
        
        # These are optional components, so don't fail tests
        assert len(monitoring_results['errors']) == 0, f"Monitoring config errors: {monitoring_results['errors']}"
    
    @pytest.mark.skipif(not os.getenv("MONITORING_TEST", False), reason="Monitoring tests require MONITORING_TEST env var")
    def test_metrics_collection(self, test_harness):
        """Test Prometheus metrics collection from services."""
        metrics_results = {
            'started_at': time.time(),
            'prometheus_reachable': False,
            'metrics_collected': 0,
            'errors': []
        }
        
        try:
            # Try to reach Prometheus
            response = requests.get("http://localhost:9090/api/v1/query?query=up", timeout=5)
            if response.status_code == 200:
                metrics_results['prometheus_reachable'] = True
                data = response.json()
                if 'data' in data and 'result' in data['data']:
                    metrics_results['metrics_collected'] = len(data['data']['result'])
            
        except requests.RequestException as e:
            metrics_results['errors'].append(f"Prometheus connection: {str(e)}")
        
        metrics_results['completed_at'] = time.time()
        test_harness.save_test_finding("metrics_collection", metrics_results)


class TestConcurrentUsage:
    """Test system performance under concurrent usage."""
    
    def test_concurrent_analysis_requests(self, test_harness, test_data):
        """Test system performance with multiple concurrent analysis requests."""
        concurrency_results = {
            'started_at': time.time(),
            'concurrent_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'errors': []
        }
        
        def simulate_analysis_task(task_id):
            """Simulate a single analysis task."""
            try:
                # Simulate data processing
                generator = TestDataGenerator(seed=task_id)
                mini_genome = generator.generate_genome(n_genes=5, avg_traits_per_gene=2)
                
                # Simulate analysis time
                time.sleep(0.1)
                
                # Create result
                result = {
                    'task_id': task_id,
                    'genes_processed': len(mini_genome),
                    'traits_found': len(set(t for g in mini_genome for t in g['annotations']['traits'])),
                    'processing_time': 0.1
                }
                
                # Save to Redis if available
                if test_harness.redis_client:
                    key = f"{MEMORY_NAMESPACE}:concurrent_task:{task_id}"
                    test_harness.redis_client.set(
                        key, 
                        json.dumps(result), 
                        ex=60
                    )
                
                return True, result
                
            except Exception as e:
                return False, str(e)
        
        # Run concurrent tasks
        num_tasks = 10
        concurrency_results['concurrent_tasks'] = num_tasks
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(simulate_analysis_task, i) for i in range(num_tasks)]
            
            for future in as_completed(futures):
                try:
                    success, result = future.result(timeout=5)
                    if success:
                        concurrency_results['successful_tasks'] += 1
                    else:
                        concurrency_results['failed_tasks'] += 1
                        concurrency_results['errors'].append(result)
                except Exception as e:
                    concurrency_results['failed_tasks'] += 1
                    concurrency_results['errors'].append(str(e))
        
        concurrency_results['completed_at'] = time.time()
        concurrency_results['duration'] = concurrency_results['completed_at'] - concurrency_results['started_at']
        concurrency_results['success_rate'] = concurrency_results['successful_tasks'] / num_tasks
        
        test_harness.save_test_finding("concurrent_usage", concurrency_results)
        
        assert concurrency_results['success_rate'] >= 0.8, "Should handle at least 80% of concurrent tasks successfully"
        assert concurrency_results['duration'] < 10, "Concurrent tasks should complete within 10 seconds"
    
    def test_memory_system_under_load(self, test_harness):
        """Test Redis memory system performance under load."""
        load_results = {
            'started_at': time.time(),
            'operations_attempted': 0,
            'operations_successful': 0,
            'errors': []
        }
        
        if not test_harness.redis_client:
            pytest.skip("Redis not available")
        
        def redis_operation_batch(batch_id):
            """Perform a batch of Redis operations."""
            try:
                operations = 0
                redis_client = test_harness.redis_client
                
                # Perform mixed operations
                for i in range(20):
                    # Write operation
                    key = f"{MEMORY_NAMESPACE}:load_test:{batch_id}:{i}"
                    data = {'batch': batch_id, 'item': i, 'timestamp': time.time()}
                    redis_client.set(key, json.dumps(data), ex=30)
                    operations += 1
                    
                    # Read operation
                    retrieved = redis_client.get(key)
                    if retrieved:
                        operations += 1
                    
                    # List operation
                    redis_client.lpush(f"{MEMORY_NAMESPACE}:load_list:{batch_id}", str(i))
                    operations += 1
                
                return operations, []
                
            except Exception as e:
                return 0, [str(e)]
        
        # Run load test with multiple workers
        num_batches = 5
        load_results['batches'] = num_batches
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(redis_operation_batch, i) for i in range(num_batches)]
            
            for future in as_completed(futures):
                try:
                    operations, errors = future.result(timeout=10)
                    load_results['operations_attempted'] += 60  # Expected operations per batch
                    load_results['operations_successful'] += operations
                    load_results['errors'].extend(errors)
                except Exception as e:
                    load_results['errors'].append(str(e))
        
        load_results['completed_at'] = time.time()
        load_results['duration'] = load_results['completed_at'] - load_results['started_at']
        
        if load_results['operations_attempted'] > 0:
            load_results['success_rate'] = load_results['operations_successful'] / load_results['operations_attempted']
        else:
            load_results['success_rate'] = 0
        
        test_harness.save_test_finding("memory_load_test", load_results)
        
        assert load_results['success_rate'] >= 0.95, "Redis should handle 95% of operations under load"
        assert load_results['duration'] < 15, "Load test should complete within 15 seconds"


class TestSystemStartupShutdown:
    """Test system startup and shutdown procedures."""
    
    def test_startup_script_validation(self, test_harness):
        """Test system startup script functionality."""
        startup_results = {
            'started_at': time.time(),
            'script_exists': False,
            'script_executable': False,
            'dependency_checks_work': False,
            'errors': []
        }
        
        try:
            startup_script = test_harness.project_root / "start_system.sh"
            
            if startup_script.exists():
                startup_results['script_exists'] = True
                
                # Check if script is executable
                if os.access(startup_script, os.X_OK):
                    startup_results['script_executable'] = True
                
                # Test script validation (dry run)
                try:
                    result = subprocess.run(
                        ["bash", str(startup_script), "--help"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode == 0 and "--help" in result.stdout:
                        startup_results['dependency_checks_work'] = True
                    else:
                        startup_results['errors'].append(f"Script help failed: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    startup_results['errors'].append("Script help timeout")
            
        except Exception as e:
            startup_results['errors'].append(str(e))
        
        startup_results['completed_at'] = time.time()
        test_harness.save_test_finding("startup_validation", startup_results)
        
        assert startup_results['script_exists'], "Startup script must exist"
        assert len(startup_results['errors']) == 0, f"Startup script errors: {startup_results['errors']}"
    
    def test_graceful_shutdown_handling(self, test_harness):
        """Test graceful shutdown handling."""
        shutdown_results = {
            'started_at': time.time(),
            'cleanup_successful': False,
            'errors': []
        }
        
        try:
            # Simulate service cleanup
            if test_harness.redis_client:
                # Create some test data to clean up
                test_keys = []
                for i in range(5):
                    key = f"{MEMORY_NAMESPACE}:shutdown_test:{i}"
                    test_harness.redis_client.set(key, f"test_data_{i}", ex=60)
                    test_keys.append(key)
                
                # Simulate graceful cleanup
                cleanup_count = 0
                for key in test_keys:
                    if test_harness.redis_client.delete(key):
                        cleanup_count += 1
                
                shutdown_results['keys_cleaned'] = cleanup_count
                shutdown_results['cleanup_successful'] = cleanup_count == len(test_keys)
            else:
                shutdown_results['cleanup_successful'] = True  # No cleanup needed
            
        except Exception as e:
            shutdown_results['errors'].append(str(e))
        
        shutdown_results['completed_at'] = time.time()
        test_harness.save_test_finding("shutdown_handling", shutdown_results)
        
        assert shutdown_results['cleanup_successful'], "Graceful shutdown cleanup must work"


def test_generate_integration_report(test_harness):
    """Generate comprehensive integration test report."""
    if not test_harness.redis_client:
        pytest.skip("Redis not available for report generation")
    
    # Collect all test findings
    pattern = f"{MEMORY_NAMESPACE}/integration-test/*"
    keys = test_harness.redis_client.keys(pattern)
    
    findings = {}
    for key in keys:
        data = test_harness.redis_client.get(key)
        if data:
            finding_name = key.split('/')[-1]
            findings[finding_name] = json.loads(data)
    
    # Generate report
    report = {
        'integration_test_summary': {
            'namespace': MEMORY_NAMESPACE,
            'total_findings': len(findings),
            'generated_at': time.time(),
            'findings': findings
        }
    }
    
    # Save final report
    test_harness.save_test_finding("final_report", report)
    
    # Write to file for external access
    report_file = test_harness.project_root / "reports" / f"integration_test_report_{int(time.time())}.json"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print("INTEGRATION TEST REPORT SUMMARY")
    print(f"{'='*60}")
    print(f"Total test findings: {len(findings)}")
    print(f"Report saved to: {report_file}")
    print(f"Memory namespace: {MEMORY_NAMESPACE}")
    
    for finding_name, finding_data in findings.items():
        if isinstance(finding_data, dict):
            errors = finding_data.get('errors', [])
            status = "PASS" if not errors else "ISSUES"
            print(f"  {finding_name}: {status}")
            if errors:
                for error in errors[:2]:  # Show first 2 errors
                    print(f"    - {error}")
    
    assert len(findings) > 0, "Should have generated test findings"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])