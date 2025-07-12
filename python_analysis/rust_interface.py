"""
Rust Interface Module for Genomic Pleiotropy Analysis

This module provides Python interface to the Rust core functionality,
supporting both PyO3 bindings and subprocess-based communication.
"""

import subprocess
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum


class InterfaceMode(Enum):
    """Interface mode for Rust communication."""
    PYO3 = "pyo3"
    SUBPROCESS = "subprocess"


@dataclass
class TraitData:
    """Data class for trait information."""
    trait_id: str
    trait_name: str
    values: List[float]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GeneData:
    """Data class for gene information."""
    gene_id: str
    gene_symbol: str
    chromosome: str
    start_position: int
    end_position: int
    expression_values: Optional[List[float]] = None


@dataclass
class PleiotropyResult:
    """Data class for pleiotropy analysis results."""
    gene_id: str
    associated_traits: List[str]
    pleiotropy_score: float
    p_values: Dict[str, float]
    effect_sizes: Dict[str, float]


class RustInterface:
    """Interface for communicating with Rust pleiotropy core."""
    
    def __init__(self, 
                 mode: InterfaceMode = InterfaceMode.SUBPROCESS,
                 rust_binary_path: Optional[str] = None,
                 pyo3_module: Optional[Any] = None):
        """
        Initialize the Rust interface.
        
        Args:
            mode: Interface mode (PYO3 or SUBPROCESS)
            rust_binary_path: Path to Rust binary (for subprocess mode)
            pyo3_module: Imported PyO3 module (for PyO3 mode)
        """
        self.mode = mode
        
        if mode == InterfaceMode.SUBPROCESS:
            if rust_binary_path is None:
                # Try to find the binary in common locations
                possible_paths = [
                    "../target/release/pleiotropy_core",
                    "../target/debug/pleiotropy_core",
                    "./pleiotropy_core"
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        rust_binary_path = path
                        break
                
                if rust_binary_path is None:
                    raise ValueError("Rust binary not found. Please provide rust_binary_path.")
            
            self.rust_binary_path = rust_binary_path
            
        elif mode == InterfaceMode.PYO3:
            if pyo3_module is None:
                try:
                    import pleiotropy_core_py as pyo3_module
                except ImportError:
                    raise ImportError("PyO3 module not found. Please build the Rust extension first.")
            
            self.pyo3_module = pyo3_module
    
    def load_trait_data(self, 
                       trait_file: str,
                       format: str = 'csv') -> List[TraitData]:
        """
        Load trait data from file.
        
        Args:
            trait_file: Path to trait data file
            format: File format ('csv', 'parquet', 'json')
            
        Returns:
            List of TraitData objects
        """
        if self.mode == InterfaceMode.PYO3:
            # Use PyO3 binding
            return self._load_trait_data_pyo3(trait_file, format)
        else:
            # Use subprocess
            return self._load_trait_data_subprocess(trait_file, format)
    
    def _load_trait_data_pyo3(self, trait_file: str, format: str) -> List[TraitData]:
        """Load trait data using PyO3 bindings."""
        raw_data = self.pyo3_module.load_traits(trait_file, format)
        return [TraitData(**item) for item in raw_data]
    
    def _load_trait_data_subprocess(self, trait_file: str, format: str) -> List[TraitData]:
        """Load trait data using subprocess."""
        cmd = [
            self.rust_binary_path,
            "load-traits",
            "--file", trait_file,
            "--format", format
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Rust command failed: {result.stderr}")
        
        raw_data = json.loads(result.stdout)
        return [TraitData(**item) for item in raw_data]
    
    def analyze_pleiotropy(self,
                          gene_data: List[GeneData],
                          trait_data: List[TraitData],
                          threshold: float = 0.05,
                          method: str = 'standard') -> List[PleiotropyResult]:
        """
        Analyze pleiotropy for given genes and traits.
        
        Args:
            gene_data: List of gene data
            trait_data: List of trait data
            threshold: P-value threshold for associations
            method: Analysis method
            
        Returns:
            List of PleiotropyResult objects
        """
        if self.mode == InterfaceMode.PYO3:
            return self._analyze_pleiotropy_pyo3(gene_data, trait_data, threshold, method)
        else:
            return self._analyze_pleiotropy_subprocess(gene_data, trait_data, threshold, method)
    
    def _analyze_pleiotropy_pyo3(self,
                                gene_data: List[GeneData],
                                trait_data: List[TraitData],
                                threshold: float,
                                method: str) -> List[PleiotropyResult]:
        """Analyze pleiotropy using PyO3 bindings."""
        # Convert to dictionaries for PyO3
        genes_dict = [asdict(g) for g in gene_data]
        traits_dict = [asdict(t) for t in trait_data]
        
        results = self.pyo3_module.analyze_pleiotropy(
            genes_dict, traits_dict, threshold, method
        )
        
        return [PleiotropyResult(**r) for r in results]
    
    def _analyze_pleiotropy_subprocess(self,
                                     gene_data: List[GeneData],
                                     trait_data: List[TraitData],
                                     threshold: float,
                                     method: str) -> List[PleiotropyResult]:
        """Analyze pleiotropy using subprocess."""
        # Create temporary files for data exchange
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gene_file:
            json.dump([asdict(g) for g in gene_data], gene_file)
            gene_file_path = gene_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as trait_file:
            json.dump([asdict(t) for t in trait_data], trait_file)
            trait_file_path = trait_file.name
        
        try:
            cmd = [
                self.rust_binary_path,
                "analyze",
                "--genes", gene_file_path,
                "--traits", trait_file_path,
                "--threshold", str(threshold),
                "--method", method
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Rust command failed: {result.stderr}")
            
            raw_results = json.loads(result.stdout)
            return [PleiotropyResult(**r) for r in raw_results]
            
        finally:
            # Clean up temporary files
            os.unlink(gene_file_path)
            os.unlink(trait_file_path)
    
    def decrypt_trait_data(self,
                          encrypted_file: str,
                          key_file: str,
                          output_format: str = 'dataframe') -> Union[pd.DataFrame, List[TraitData]]:
        """
        Decrypt encrypted trait data.
        
        Args:
            encrypted_file: Path to encrypted data file
            key_file: Path to decryption key file
            output_format: Output format ('dataframe' or 'list')
            
        Returns:
            Decrypted data as DataFrame or list of TraitData
        """
        if self.mode == InterfaceMode.PYO3:
            decrypted_data = self.pyo3_module.decrypt_traits(encrypted_file, key_file)
        else:
            cmd = [
                self.rust_binary_path,
                "decrypt",
                "--input", encrypted_file,
                "--key", key_file,
                "--output-format", "json"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Decryption failed: {result.stderr}")
            
            decrypted_data = json.loads(result.stdout)
        
        if output_format == 'dataframe':
            # Convert to DataFrame
            trait_dict = {}
            for item in decrypted_data:
                trait_dict[item['trait_name']] = item['values']
            
            return pd.DataFrame(trait_dict)
        else:
            return [TraitData(**item) for item in decrypted_data]
    
    def compute_genetic_correlations(self,
                                   trait1_data: np.ndarray,
                                   trait2_data: np.ndarray,
                                   genotype_matrix: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute genetic correlations between traits.
        
        Args:
            trait1_data: First trait values
            trait2_data: Second trait values
            genotype_matrix: Optional genotype matrix
            
        Returns:
            Dictionary with correlation results
        """
        if self.mode == InterfaceMode.PYO3:
            return self.pyo3_module.compute_genetic_correlation(
                trait1_data.tolist(),
                trait2_data.tolist(),
                genotype_matrix.tolist() if genotype_matrix is not None else None
            )
        else:
            # Prepare data for subprocess
            data = {
                'trait1': trait1_data.tolist(),
                'trait2': trait2_data.tolist()
            }
            if genotype_matrix is not None:
                data['genotypes'] = genotype_matrix.tolist()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(data, temp_file)
                temp_path = temp_file.name
            
            try:
                cmd = [
                    self.rust_binary_path,
                    "genetic-correlation",
                    "--input", temp_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"Correlation computation failed: {result.stderr}")
                
                return json.loads(result.stdout)
                
            finally:
                os.unlink(temp_path)
    
    def run_gwas(self,
                 genotype_file: str,
                 phenotype_file: str,
                 covariates_file: Optional[str] = None,
                 output_dir: str = "./gwas_results") -> pd.DataFrame:
        """
        Run GWAS analysis using Rust core.
        
        Args:
            genotype_file: Path to genotype data
            phenotype_file: Path to phenotype data
            covariates_file: Optional path to covariates
            output_dir: Output directory for results
            
        Returns:
            DataFrame with GWAS results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.mode == InterfaceMode.PYO3:
            results = self.pyo3_module.run_gwas(
                genotype_file,
                phenotype_file,
                covariates_file,
                output_dir
            )
        else:
            cmd = [
                self.rust_binary_path,
                "gwas",
                "--genotypes", genotype_file,
                "--phenotypes", phenotype_file,
                "--output", output_dir
            ]
            
            if covariates_file:
                cmd.extend(["--covariates", covariates_file])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"GWAS failed: {result.stderr}")
            
            # Read results from output file
            results_file = os.path.join(output_dir, "gwas_results.json")
            with open(results_file, 'r') as f:
                results = json.load(f)
        
        return pd.DataFrame(results)
    
    def batch_process_traits(self,
                           trait_files: List[str],
                           processing_function: str,
                           **kwargs) -> Dict[str, Any]:
        """
        Batch process multiple trait files.
        
        Args:
            trait_files: List of trait file paths
            processing_function: Name of processing function
            **kwargs: Additional arguments for processing
            
        Returns:
            Dictionary with processing results
        """
        if self.mode == InterfaceMode.PYO3:
            return self.pyo3_module.batch_process(
                trait_files,
                processing_function,
                kwargs
            )
        else:
            # Create batch configuration
            batch_config = {
                'files': trait_files,
                'function': processing_function,
                'parameters': kwargs
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as config_file:
                json.dump(batch_config, config_file)
                config_path = config_file.name
            
            try:
                cmd = [
                    self.rust_binary_path,
                    "batch",
                    "--config", config_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"Batch processing failed: {result.stderr}")
                
                return json.loads(result.stdout)
                
            finally:
                os.unlink(config_path)
    
    def get_version(self) -> str:
        """Get version of the Rust core."""
        if self.mode == InterfaceMode.PYO3:
            return self.pyo3_module.get_version()
        else:
            cmd = [self.rust_binary_path, "--version"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.stdout.strip()


# Utility functions for data conversion
def dataframe_to_trait_data(df: pd.DataFrame, 
                           trait_column: str,
                           value_columns: List[str],
                           metadata_columns: Optional[List[str]] = None) -> List[TraitData]:
    """Convert DataFrame to list of TraitData objects."""
    trait_data_list = []
    
    for _, row in df.iterrows():
        metadata = {}
        if metadata_columns:
            metadata = {col: row[col] for col in metadata_columns if col in df.columns}
        
        trait_data = TraitData(
            trait_id=str(row.name),
            trait_name=row[trait_column],
            values=[row[col] for col in value_columns],
            metadata=metadata
        )
        trait_data_list.append(trait_data)
    
    return trait_data_list


def results_to_dataframe(results: List[PleiotropyResult]) -> pd.DataFrame:
    """Convert list of PleiotropyResult objects to DataFrame."""
    data = []
    for result in results:
        row = {
            'gene_id': result.gene_id,
            'n_associated_traits': len(result.associated_traits),
            'associated_traits': ';'.join(result.associated_traits),
            'pleiotropy_score': result.pleiotropy_score
        }
        
        # Add p-values and effect sizes
        for trait, p_val in result.p_values.items():
            row[f'p_value_{trait}'] = p_val
        
        for trait, effect in result.effect_sizes.items():
            row[f'effect_size_{trait}'] = effect
        
        data.append(row)
    
    return pd.DataFrame(data)