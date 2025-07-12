"""
Rust Analyzer Agent - Handles cryptographic analysis tasks
"""

import asyncio
import subprocess
import json
import os
from typing import Dict, Any
from base_agent import BaseSwarmAgent
import logging

logger = logging.getLogger(__name__)


class RustAnalyzerAgent(BaseSwarmAgent):
    """Agent for running Rust-based cryptographic analysis"""
    
    def __init__(self, rust_impl_path: str = None):
        super().__init__(
            agent_type="rust_analyzer",
            capabilities=["crypto_analysis", "sequence_parsing", "frequency_analysis", "trait_extraction"]
        )
        self.rust_impl_path = rust_impl_path or "/home/murr2k/projects/agentic/pleiotropy/rust_impl"
        self.analysis_cache = {}
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Rust analysis task"""
        task_type = task_data.get('type')
        payload = task_data.get('payload', {})
        
        if task_type == "crypto_analysis":
            return await self._run_crypto_analysis(payload)
        elif task_type == "sequence_parsing":
            return await self._parse_sequence(payload)
        elif task_type == "frequency_analysis":
            return await self._analyze_frequencies(payload)
        elif task_type == "trait_extraction":
            return await self._extract_traits(payload)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _run_crypto_analysis(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run full cryptographic analysis"""
        genome_file = payload.get('genome_file')
        traits_file = payload.get('traits_file')
        window_size = payload.get('window_size', 1000)
        overlap = payload.get('overlap', 100)
        
        # Build Rust command
        cmd = [
            "cargo", "run", "--release", "--",
            "--genome", genome_file,
            "--traits", traits_file,
            "--window-size", str(window_size),
            "--overlap", str(overlap),
            "--output", "json"
        ]
        
        # Run analysis
        logger.info(f"Running Rust analysis: {' '.join(cmd)}")
        
        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.rust_impl_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                raise RuntimeError(f"Rust analysis failed: {stderr.decode()}")
            
            # Parse JSON output
            analysis_result = json.loads(stdout.decode())
            
            # Cache result
            cache_key = f"{genome_file}:{window_size}:{overlap}"
            self.analysis_cache[cache_key] = analysis_result
            
            # Save to memory for other agents
            self.save_to_memory(f"analysis:{task_data['id']}", analysis_result)
            
            return {
                'status': 'success',
                'analysis': analysis_result,
                'cache_key': cache_key
            }
            
        except Exception as e:
            logger.error(f"Crypto analysis error: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _parse_sequence(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse genomic sequence"""
        sequence_file = payload.get('file')
        format_type = payload.get('format', 'fasta')
        
        # Use Rust sequence parser
        cmd = [
            "cargo", "run", "--bin", "sequence_parser", "--",
            "--input", sequence_file,
            "--format", format_type
        ]
        
        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.rust_impl_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                raise RuntimeError(f"Sequence parsing failed: {stderr.decode()}")
            
            parsed_data = json.loads(stdout.decode())
            
            return {
                'status': 'success',
                'sequence_data': parsed_data,
                'num_sequences': len(parsed_data.get('sequences', []))
            }
            
        except Exception as e:
            logger.error(f"Sequence parsing error: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _analyze_frequencies(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze codon frequencies"""
        sequence_data = payload.get('sequence_data')
        
        if isinstance(sequence_data, str):
            # Load from memory if string key provided
            sequence_data = self.load_from_memory(sequence_data)
        
        # Run frequency analysis
        cmd = [
            "cargo", "run", "--bin", "frequency_analyzer", "--"
        ]
        
        try:
            # Pass sequence data via stdin
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.rust_impl_path,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate(
                input=json.dumps(sequence_data).encode()
            )
            
            if process.returncode != 0:
                raise RuntimeError(f"Frequency analysis failed: {stderr.decode()}")
            
            frequency_data = json.loads(stdout.decode())
            
            # Calculate statistics
            total_codons = sum(frequency_data.values())
            unique_codons = len(frequency_data)
            
            return {
                'status': 'success',
                'frequencies': frequency_data,
                'statistics': {
                    'total_codons': total_codons,
                    'unique_codons': unique_codons,
                    'shannon_entropy': self._calculate_entropy(frequency_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Frequency analysis error: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _extract_traits(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract traits from analysis"""
        analysis_data = payload.get('analysis_data')
        confidence_threshold = payload.get('confidence_threshold', 0.7)
        
        if isinstance(analysis_data, str):
            # Load from memory if string key provided
            analysis_data = self.load_from_memory(analysis_data)
        
        # Run trait extraction
        cmd = [
            "cargo", "run", "--bin", "trait_extractor", "--",
            "--threshold", str(confidence_threshold)
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.rust_impl_path,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate(
                input=json.dumps(analysis_data).encode()
            )
            
            if process.returncode != 0:
                raise RuntimeError(f"Trait extraction failed: {stderr.decode()}")
            
            traits = json.loads(stdout.decode())
            
            # Publish trait discovery event
            self.publish_event('traits_discovered', {
                'traits': traits,
                'confidence_threshold': confidence_threshold
            })
            
            return {
                'status': 'success',
                'traits': traits,
                'num_traits': len(traits)
            }
            
        except Exception as e:
            logger.error(f"Trait extraction error: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _calculate_entropy(self, frequencies: Dict[str, int]) -> float:
        """Calculate Shannon entropy"""
        import math
        total = sum(frequencies.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in frequencies.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect Rust analyzer metrics"""
        # Check Rust process health
        try:
            result = await asyncio.create_subprocess_exec(
                "cargo", "--version",
                cwd=self.rust_impl_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            await result.communicate()
            rust_healthy = result.returncode == 0
        except:
            rust_healthy = False
        
        return {
            'rust_healthy': rust_healthy,
            'cache_size': len(self.analysis_cache),
            'cached_analyses': list(self.analysis_cache.keys())
        }


# Run agent if executed directly
async def main():
    agent = RustAnalyzerAgent()
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())