"""
Integration Module - Connects swarm system with existing Rust and Python components
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any, List, Optional
import subprocess
from pathlib import Path
import logging

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "python_analysis"))

from coordinator import SwarmCoordinator
from rust_analyzer_agent import RustAnalyzerAgent
from python_visualizer_agent import PythonVisualizerAgent

# Import existing Python modules
try:
    from python_analysis.rust_interface import RustInterface
    from python_analysis.statistical_analyzer import StatisticalAnalyzer
    from python_analysis.trait_visualizer import TraitVisualizer
except ImportError:
    logging.warning("Could not import existing Python modules")

logger = logging.getLogger(__name__)


class SwarmIntegration:
    """Integrates swarm system with existing codebase"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or "/home/murr2k/projects/agentic/pleiotropy")
        self.coordinator = SwarmCoordinator()
        self.agents = {}
        self.memory_namespace = "swarm-auto-centralized-1752300927219"
    
    async def initialize(self):
        """Initialize the integration system"""
        logger.info("Initializing Swarm Integration...")
        
        # Start coordinator in background
        asyncio.create_task(self.coordinator.start())
        
        # Initialize agents
        await self._initialize_agents()
        
        # Setup integration bridges
        await self._setup_bridges()
        
        logger.info("Swarm Integration initialized successfully")
    
    async def _initialize_agents(self):
        """Initialize and register agents"""
        # Rust Analyzer Agent
        rust_agent = RustAnalyzerAgent(
            rust_impl_path=str(self.project_root / "rust_impl")
        )
        rust_task = asyncio.create_task(rust_agent.start())
        self.agents['rust_analyzer'] = {'agent': rust_agent, 'task': rust_task}
        
        # Python Visualizer Agent
        python_agent = PythonVisualizerAgent()
        python_task = asyncio.create_task(python_agent.start())
        self.agents['python_visualizer'] = {'agent': python_agent, 'task': python_task}
        
        # Wait for agents to register
        await asyncio.sleep(2)
        
        logger.info(f"Initialized {len(self.agents)} agents")
    
    async def _setup_bridges(self):
        """Setup bridges to existing code"""
        # Create adapter for existing Rust interface
        self.rust_bridge = RustInterfaceBridge(self.coordinator)
        
        # Create adapter for existing Python modules
        self.python_bridge = PythonAnalysisBridge(self.coordinator)
    
    async def run_ecoli_workflow(self) -> Dict[str, Any]:
        """Run the E. coli workflow using swarm system"""
        logger.info("Starting E. coli workflow via swarm...")
        
        # Load E. coli data paths
        genome_file = self.project_root / "genome_research" / "ecoli_k12.fasta"
        traits_file = self.project_root / "genome_research" / "ecoli_pleiotropic_genes.json"
        
        # Step 1: Parse sequence
        parse_task_id = await self.coordinator.submit_task(
            task_type="sequence_parsing",
            payload={
                "file": str(genome_file),
                "format": "fasta"
            },
            priority=10
        )
        
        # Wait for parsing to complete
        parse_result = await self._wait_for_task(parse_task_id)
        
        # Step 2: Run cryptographic analysis
        crypto_task_id = await self.coordinator.submit_task(
            task_type="crypto_analysis",
            payload={
                "genome_file": str(genome_file),
                "traits_file": str(traits_file),
                "window_size": 1000,
                "overlap": 100
            },
            priority=9
        )
        
        crypto_result = await self._wait_for_task(crypto_task_id)
        
        # Step 3: Generate visualizations
        viz_tasks = []
        
        # Frequency heatmap
        heatmap_task = await self.coordinator.submit_task(
            task_type="visualization",
            payload={
                "viz_type": "heatmap",
                "data_key": f"analysis:{crypto_task_id}",
                "title": "E. coli Codon Frequency Analysis"
            },
            priority=5
        )
        viz_tasks.append(heatmap_task)
        
        # Trait network
        network_task = await self.coordinator.submit_task(
            task_type="visualization",
            payload={
                "viz_type": "trait_network",
                "data_key": f"analysis:{crypto_task_id}",
                "title": "E. coli Pleiotropic Trait Network"
            },
            priority=5
        )
        viz_tasks.append(network_task)
        
        # Wait for visualizations
        viz_results = await asyncio.gather(
            *[self._wait_for_task(task_id) for task_id in viz_tasks]
        )
        
        # Step 4: Generate report
        report_task = await self.coordinator.submit_task(
            task_type="report_generation",
            payload={
                "analysis_keys": [
                    f"analysis:{crypto_task_id}",
                    f"viz:{heatmap_task}",
                    f"viz:{network_task}"
                ]
            },
            priority=3
        )
        
        report_result = await self._wait_for_task(report_task)
        
        return {
            'status': 'success',
            'workflow': 'ecoli_analysis',
            'results': {
                'parsing': parse_result,
                'crypto_analysis': crypto_result,
                'visualizations': viz_results,
                'report': report_result
            }
        }
    
    async def _wait_for_task(self, task_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for a task to complete"""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            task_status = self.coordinator.get_task_status(task_id)
            
            if task_status and task_status['status'] == 'completed':
                return task_status.get('result', {})
            elif task_status and task_status['status'] == 'failed':
                raise RuntimeError(f"Task {task_id} failed: {task_status.get('error')}")
            
            await asyncio.sleep(1)
        
        raise TimeoutError(f"Task {task_id} timed out after {timeout} seconds")
    
    async def shutdown(self):
        """Shutdown the integration system"""
        logger.info("Shutting down Swarm Integration...")
        
        # Stop agents
        for agent_info in self.agents.values():
            agent_info['agent'].stop()
            agent_info['task'].cancel()
        
        # Stop coordinator
        await self.coordinator.stop()
        
        logger.info("Swarm Integration shutdown complete")


class RustInterfaceBridge:
    """Bridge between swarm system and existing Rust interface"""
    
    def __init__(self, coordinator: SwarmCoordinator):
        self.coordinator = coordinator
    
    async def analyze_genome(self, genome_path: str, traits_path: str, **kwargs) -> Dict[str, Any]:
        """Analyze genome using swarm system"""
        task_id = await self.coordinator.submit_task(
            task_type="crypto_analysis",
            payload={
                "genome_file": genome_path,
                "traits_file": traits_path,
                **kwargs
            }
        )
        
        # Wait for completion
        while True:
            status = self.coordinator.get_task_status(task_id)
            if status['status'] == 'completed':
                return status['result']
            elif status['status'] == 'failed':
                raise RuntimeError(f"Analysis failed: {status.get('error')}")
            await asyncio.sleep(1)


class PythonAnalysisBridge:
    """Bridge between swarm system and existing Python analysis"""
    
    def __init__(self, coordinator: SwarmCoordinator):
        self.coordinator = coordinator
    
    async def create_visualization(self, data: Any, viz_type: str = "heatmap", **kwargs) -> str:
        """Create visualization using swarm system"""
        # Store data in Redis
        import uuid
        data_key = f"bridge_data_{uuid.uuid4().hex}"
        
        # Use coordinator's Redis client
        self.coordinator.redis_client.set(
            f"{self.coordinator.memory_namespace}:{data_key}",
            json.dumps(data),
            ex=3600
        )
        
        # Submit visualization task
        task_id = await self.coordinator.submit_task(
            task_type="visualization",
            payload={
                "viz_type": viz_type,
                "data_key": data_key,
                **kwargs
            }
        )
        
        # Wait for completion
        while True:
            status = self.coordinator.get_task_status(task_id)
            if status['status'] == 'completed':
                return status['result']['visualization']['cache_key']
            elif status['status'] == 'failed':
                raise RuntimeError(f"Visualization failed: {status.get('error')}")
            await asyncio.sleep(1)


# Backward compatibility functions
def run_rust_analysis(genome_file: str, traits_file: str) -> Dict[str, Any]:
    """Backward compatible function for running Rust analysis"""
    async def _run():
        integration = SwarmIntegration()
        await integration.initialize()
        bridge = RustInterfaceBridge(integration.coordinator)
        result = await bridge.analyze_genome(genome_file, traits_file)
        await integration.shutdown()
        return result
    
    return asyncio.run(_run())


def create_trait_visualization(data: Dict[str, Any], output_file: str):
    """Backward compatible function for creating visualizations"""
    async def _run():
        integration = SwarmIntegration()
        await integration.initialize()
        bridge = PythonAnalysisBridge(integration.coordinator)
        cache_key = await bridge.create_visualization(data)
        await integration.shutdown()
        return cache_key
    
    return asyncio.run(_run())


# Example usage
async def main():
    integration = SwarmIntegration()
    await integration.initialize()
    
    # Run E. coli workflow
    results = await integration.run_ecoli_workflow()
    print(json.dumps(results, indent=2))
    
    await integration.shutdown()


if __name__ == "__main__":
    asyncio.run(main())