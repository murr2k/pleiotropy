"""
Integration Testing Suite for Swarm System
"""

import asyncio
import json
import os
import sys
import pytest
import redis
from pathlib import Path
from typing import Dict, Any

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from coordinator import SwarmCoordinator, TaskStatus
from rust_analyzer_agent import RustAnalyzerAgent
from python_visualizer_agent import PythonVisualizerAgent
from integration import SwarmIntegration


class TestSwarmIntegration:
    """Test suite for swarm integration"""
    
    @pytest.fixture
    async def coordinator(self):
        """Create coordinator instance"""
        coord = SwarmCoordinator(redis_host='localhost')
        # Don't start the full coordinator, just use it for testing
        yield coord
        # Cleanup
        coord.redis_client.flushdb()
    
    @pytest.fixture
    async def integration(self):
        """Create integration instance"""
        integ = SwarmIntegration()
        yield integ
        await integ.shutdown()
    
    @pytest.mark.asyncio
    async def test_coordinator_task_submission(self, coordinator):
        """Test task submission and retrieval"""
        # Submit a task
        task_id = await coordinator.submit_task(
            task_type="test_task",
            payload={"test": "data"},
            priority=5
        )
        
        assert task_id is not None
        assert task_id.startswith("task_")
        
        # Check task status
        status = coordinator.get_task_status(task_id)
        assert status is not None
        assert status['status'] == 'pending'
        assert status['type'] == 'test_task'
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, coordinator):
        """Test agent registration"""
        # Register an agent
        agent_id = coordinator.register_agent(
            agent_type="test_agent",
            capabilities=["test_cap1", "test_cap2"]
        )
        
        assert agent_id is not None
        assert agent_id.startswith("test_agent_")
        
        # Check agent status
        status = coordinator.get_agent_status(agent_id)
        assert status is not None
        assert status['type'] == 'test_agent'
        assert 'test_cap1' in status['capabilities']
    
    @pytest.mark.asyncio
    async def test_memory_operations(self, coordinator):
        """Test memory save/load operations"""
        test_data = {"key": "value", "number": 42}
        key = "test_memory_key"
        
        # Save to memory
        coordinator.redis_client.set(
            f"{coordinator.memory_namespace}:{key}",
            json.dumps(test_data)
        )
        
        # Load from memory
        loaded = coordinator.redis_client.get(f"{coordinator.memory_namespace}:{key}")
        assert loaded is not None
        
        loaded_data = json.loads(loaded)
        assert loaded_data == test_data
    
    @pytest.mark.asyncio
    async def test_rust_agent_initialization(self):
        """Test Rust analyzer agent initialization"""
        agent = RustAnalyzerAgent()
        
        assert agent.agent_type == "rust_analyzer"
        assert "crypto_analysis" in agent.capabilities
        assert "sequence_parsing" in agent.capabilities
    
    @pytest.mark.asyncio
    async def test_python_agent_initialization(self):
        """Test Python visualizer agent initialization"""
        agent = PythonVisualizerAgent()
        
        assert agent.agent_type == "python_visualizer"
        assert "visualization" in agent.capabilities
        assert "statistics" in agent.capabilities
    
    @pytest.mark.asyncio
    async def test_task_execution_flow(self, coordinator):
        """Test complete task execution flow"""
        # This is a simplified test - in real scenario, agents would be running
        
        # Submit task
        task_id = await coordinator.submit_task(
            task_type="sequence_parsing",
            payload={"file": "test.fasta"}
        )
        
        # Simulate task assignment
        task = coordinator.tasks[task_id]
        task.status = TaskStatus.ASSIGNED
        task.assigned_to = "test_agent_123"
        
        # Simulate task completion
        task.status = TaskStatus.COMPLETED
        task.result = {"parsed": True, "sequences": 10}
        
        # Check final status
        status = coordinator.get_task_status(task_id)
        assert status['status'] == 'completed'
        assert status['result']['parsed'] is True


class TestIntegrationBridges:
    """Test integration bridges"""
    
    @pytest.mark.asyncio
    async def test_rust_bridge_functionality(self):
        """Test Rust interface bridge"""
        from integration import RustInterfaceBridge
        
        # Create mock coordinator
        coordinator = SwarmCoordinator()
        bridge = RustInterfaceBridge(coordinator)
        
        # Test that bridge methods exist
        assert hasattr(bridge, 'analyze_genome')
    
    @pytest.mark.asyncio
    async def test_python_bridge_functionality(self):
        """Test Python analysis bridge"""
        from integration import PythonAnalysisBridge
        
        # Create mock coordinator
        coordinator = SwarmCoordinator()
        bridge = PythonAnalysisBridge(coordinator)
        
        # Test that bridge methods exist
        assert hasattr(bridge, 'create_visualization')


class TestBackwardCompatibility:
    """Test backward compatibility functions"""
    
    def test_backward_compatible_imports(self):
        """Test that backward compatible functions can be imported"""
        from integration import run_rust_analysis, create_trait_visualization
        
        assert callable(run_rust_analysis)
        assert callable(create_trait_visualization)


# Performance tests
class TestPerformance:
    """Performance testing"""
    
    @pytest.mark.asyncio
    async def test_task_throughput(self, coordinator):
        """Test task submission throughput"""
        import time
        
        start_time = time.time()
        num_tasks = 100
        
        task_ids = []
        for i in range(num_tasks):
            task_id = await coordinator.submit_task(
                task_type="perf_test",
                payload={"index": i}
            )
            task_ids.append(task_id)
        
        elapsed = time.time() - start_time
        throughput = num_tasks / elapsed
        
        print(f"Task submission throughput: {throughput:.2f} tasks/second")
        assert throughput > 50  # Should handle at least 50 tasks/second
    
    @pytest.mark.asyncio
    async def test_memory_performance(self, coordinator):
        """Test memory operations performance"""
        import time
        
        num_operations = 1000
        test_data = {"test": "data" * 100}  # ~1KB of data
        
        # Write performance
        start_time = time.time()
        for i in range(num_operations):
            coordinator.redis_client.set(
                f"{coordinator.memory_namespace}:perf_test_{i}",
                json.dumps(test_data),
                ex=60
            )
        write_time = time.time() - start_time
        
        # Read performance
        start_time = time.time()
        for i in range(num_operations):
            data = coordinator.redis_client.get(f"{coordinator.memory_namespace}:perf_test_{i}")
        read_time = time.time() - start_time
        
        print(f"Memory write throughput: {num_operations/write_time:.2f} ops/second")
        print(f"Memory read throughput: {num_operations/read_time:.2f} ops/second")
        
        assert num_operations/write_time > 100  # At least 100 writes/second
        assert num_operations/read_time > 500   # At least 500 reads/second


# Integration scenarios
class TestScenarios:
    """Test real-world scenarios"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_ecoli_workflow_components(self, integration):
        """Test E. coli workflow components"""
        # This test checks that all components needed for the workflow exist
        
        # Check paths
        genome_file = integration.project_root / "genome_research" / "ecoli_k12.fasta"
        traits_file = integration.project_root / "genome_research" / "ecoli_pleiotropic_genes.json"
        
        # Note: In a real test, we'd create test data files
        # For now, just check the structure is correct
        assert integration.project_root.exists()
        assert (integration.project_root / "rust_impl").exists()
        assert (integration.project_root / "python_analysis").exists()


def run_tests():
    """Run all tests"""
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])


if __name__ == "__main__":
    run_tests()