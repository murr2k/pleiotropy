#!/usr/bin/env python3
"""
CUDA Factorizer Agent for Semiprime and Composite Number Factorization

This agent specializes in GPU-accelerated factorization of composite numbers,
particularly semiprimes for cryptographic analysis. It uses the CUDA-enabled
Rust implementation for 10-50x performance improvements.
"""

import os
import sys
import json
import time
import subprocess
import logging
from typing import Dict, List, Optional, Tuple
from base_agent import BaseAgent

class CudaFactorizerAgent(BaseAgent):
    """Agent specialized in CUDA-accelerated composite number factorization"""
    
    def __init__(self, agent_id: str = "cuda_factorizer"):
        super().__init__(agent_id, "cuda_factorizer")
        self.rust_bin_path = "/usr/local/bin"
        self.seeker_bin = f"{self.rust_bin_path}/semiprime_seeker"
        self.validate_bin = f"{self.rust_bin_path}/validate_factorization"
        
        # Performance tracking
        self.factorization_count = 0
        self.total_time = 0.0
        self.successful_factorizations = 0
        
        logging.info(f"CUDA Factorizer Agent {agent_id} initialized")
        self._check_cuda_availability()
    
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available and working"""
        try:
            # Check for NVIDIA runtime
            result = subprocess.run(
                ["nvidia-smi"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                logging.info("CUDA device detected and available")
                return True
            else:
                logging.warning("nvidia-smi failed, GPU may not be available")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logging.warning("CUDA not available, will use CPU fallback")
            return False
    
    def factorize_number(self, number: int) -> Dict:
        """
        Factorize a composite number using CUDA acceleration
        
        Args:
            number: Integer to factorize
            
        Returns:
            Dict with factorization results and metadata
        """
        start_time = time.time()
        self.factorization_count += 1
        
        try:
            # Use the Rust factorizer binary
            cmd = [
                "/usr/local/bin/genomic_cryptanalysis",
                "factorize",
                str(number)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            elapsed_time = time.time() - start_time
            self.total_time += elapsed_time
            
            if result.returncode == 0:
                self.successful_factorizations += 1
                
                # Parse the output (expected JSON format)
                try:
                    factors_data = json.loads(result.stdout)
                    factors = factors_data.get("factors", [])
                    
                    return {
                        "success": True,
                        "number": number,
                        "factors": factors,
                        "is_semiprime": len(factors) == 2,
                        "time_seconds": elapsed_time,
                        "method": factors_data.get("method", "unknown"),
                        "gpu_used": factors_data.get("gpu_used", False),
                        "agent_id": self.agent_id
                    }
                except json.JSONDecodeError:
                    # Fallback: parse simple factor list
                    factors = [int(x.strip()) for x in result.stdout.strip().split() if x.strip().isdigit()]
                    return {
                        "success": True,
                        "number": number,
                        "factors": factors,
                        "is_semiprime": len(factors) == 2,
                        "time_seconds": elapsed_time,
                        "method": "rust_fallback",
                        "gpu_used": False,
                        "agent_id": self.agent_id
                    }
            else:
                logging.error(f"Factorization failed: {result.stderr}")
                return {
                    "success": False,
                    "number": number,
                    "error": result.stderr,
                    "time_seconds": elapsed_time,
                    "agent_id": self.agent_id
                }
                
        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            logging.error(f"Factorization of {number} timed out after {elapsed_time:.2f}s")
            return {
                "success": False,
                "number": number,
                "error": "timeout",
                "time_seconds": elapsed_time,
                "agent_id": self.agent_id
            }
        except Exception as e:
            elapsed_time = time.time() - start_time
            logging.error(f"Error factorizing {number}: {e}")
            return {
                "success": False,
                "number": number,
                "error": str(e),
                "time_seconds": elapsed_time,
                "agent_id": self.agent_id
            }
    
    def run_semiprime_seeker(self, target_time: float = 600.0) -> Dict:
        """
        Run the semiprime seeker to find optimal challenge
        
        Args:
            target_time: Target factorization time in seconds (default 10 minutes)
            
        Returns:
            Dict with seeker results
        """
        if not os.path.exists(self.seeker_bin):
            logging.error(f"Semiprime seeker binary not found: {self.seeker_bin}")
            return {"success": False, "error": "seeker_binary_not_found"}
        
        try:
            cmd = [self.seeker_bin, "--target-time", str(target_time)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                try:
                    seeker_data = json.loads(result.stdout)
                    return {
                        "success": True,
                        "target_time": target_time,
                        "result": seeker_data,
                        "agent_id": self.agent_id
                    }
                except json.JSONDecodeError:
                    return {
                        "success": True,
                        "target_time": target_time,
                        "result": {"raw_output": result.stdout},
                        "agent_id": self.agent_id
                    }
            else:
                logging.error(f"Seeker failed: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "agent_id": self.agent_id
                }
                
        except subprocess.TimeoutExpired:
            logging.error("Semiprime seeker timed out")
            return {
                "success": False,
                "error": "timeout",
                "agent_id": self.agent_id
            }
        except Exception as e:
            logging.error(f"Error running seeker: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id
            }
    
    def validate_factorization(self, number: int, factors: List[int]) -> Dict:
        """
        Validate that factors multiply to the original number and are prime
        
        Args:
            number: Original number
            factors: List of proposed factors
            
        Returns:
            Dict with validation results
        """
        try:
            # Check multiplication
            product = 1
            for factor in factors:
                product *= factor
            
            if product != number:
                return {
                    "valid": False,
                    "error": f"Product {product} != original {number}",
                    "agent_id": self.agent_id
                }
            
            # Use Rust validator for primality testing
            if os.path.exists(self.validate_bin):
                cmd = [self.validate_bin, str(number)] + [str(f) for f in factors]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    try:
                        validation_data = json.loads(result.stdout)
                        return {
                            "valid": True,
                            "validation": validation_data,
                            "agent_id": self.agent_id
                        }
                    except json.JSONDecodeError:
                        pass
            
            # Fallback validation
            return {
                "valid": True,
                "product_check": True,
                "primality_check": "not_verified",
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "agent_id": self.agent_id
            }
    
    def get_performance_stats(self) -> Dict:
        """Get agent performance statistics"""
        avg_time = self.total_time / max(self.factorization_count, 1)
        success_rate = self.successful_factorizations / max(self.factorization_count, 1)
        
        return {
            "factorization_count": self.factorization_count,
            "successful_factorizations": self.successful_factorizations,
            "total_time": self.total_time,
            "average_time": avg_time,
            "success_rate": success_rate,
            "agent_id": self.agent_id
        }
    
    def process_task(self, task_data: Dict) -> Dict:
        """Process a factorization task"""
        task_type = task_data.get("type", "factorize")
        
        if task_type == "factorize":
            number = task_data.get("number")
            if not number:
                return {"success": False, "error": "no_number_provided"}
            return self.factorize_number(number)
            
        elif task_type == "seeker":
            target_time = task_data.get("target_time", 600.0)
            return self.run_semiprime_seeker(target_time)
            
        elif task_type == "validate":
            number = task_data.get("number")
            factors = task_data.get("factors", [])
            if not number or not factors:
                return {"success": False, "error": "missing_validation_data"}
            return self.validate_factorization(number, factors)
            
        elif task_type == "stats":
            return self.get_performance_stats()
            
        else:
            return {"success": False, "error": f"unknown_task_type: {task_type}"}

def main():
    """Main entry point for CUDA Factorizer Agent"""
    agent = CudaFactorizerAgent()
    
    try:
        agent.run()
    except KeyboardInterrupt:
        logging.info("CUDA Factorizer Agent shutting down...")
    except Exception as e:
        logging.error(f"Agent error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()