"""
Integration Example: Using Adaptive Confidence Protocol with Pleiotropy Analysis Pipeline

This script demonstrates how to integrate the adaptive confidence protocol
with the existing Rust-Python pleiotropy analysis pipeline.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from adaptive_confidence_protocol import (
    AdaptiveConfidenceProtocol,
    DetectionPhase,
    GeneContext,
    ValidationMetrics
)
from python_analysis.rust_interface import RustInterface
from python_analysis.statistical_analyzer import StatisticalAnalyzer


class PleiotropyDetectionPipeline:
    """
    Complete pipeline integrating adaptive confidence protocol
    with existing pleiotropy detection system.
    """
    
    def __init__(self, config_scenario: str = "balanced"):
        """Initialize pipeline with configuration scenario."""
        self.protocol = AdaptiveConfidenceProtocol()
        self.rust_interface = RustInterface()
        self.stat_analyzer = StatisticalAnalyzer()
        
        # Load configuration
        config_path = Path(__file__).parent / "configuration_templates.json"
        with open(config_path, 'r') as f:
            configs = json.load(f)
        
        # Set up based on scenario
        if config_scenario in configs["scenario_configurations"]:
            self.config = configs["scenario_configurations"][config_scenario]
            self._apply_configuration(self.config)
        else:
            # Use quick start profile
            self.config = configs["quick_start_profiles"].get(
                config_scenario, configs["quick_start_profiles"]["balanced"]
            )
    
    def _apply_configuration(self, config: Dict):
        """Apply configuration to protocol."""
        phase = DetectionPhase(config["phase"])
        self.protocol.update_phase(phase)
        
        # Update protocol parameters
        params = config["parameters"]
        phase_config = self.protocol.phase_configs[phase]
        phase_config.base_threshold = params["base_threshold"]
        phase_config.min_threshold = params["min_threshold"]
        phase_config.max_threshold = params["max_threshold"]
        
        self.protocol.learning_rate = params["learning_rate"]
        self.protocol.ensemble_weights = params["ensemble_weights"]
    
    def analyze_genome(self, 
                      fasta_path: str,
                      traits_path: str,
                      known_pleiotropic_genes: List[str] = None) -> Dict:
        """
        Run complete analysis pipeline with adaptive thresholds.
        
        Args:
            fasta_path: Path to genome FASTA file
            traits_path: Path to traits JSON file
            known_pleiotropic_genes: Optional list of known genes for validation
            
        Returns:
            Dictionary with analysis results
        """
        print(f"Starting pleiotropy analysis with {self.protocol.current_phase.value} phase")
        
        # Step 1: Run Rust analysis to get initial scores
        print("Running cryptanalysis engine...")
        rust_results = self.rust_interface.analyze_genome(fasta_path, traits_path)
        
        # Step 2: Extract gene contexts and scores
        gene_contexts = self._extract_gene_contexts(rust_results)
        method_scores = self._extract_method_scores(rust_results)
        
        # Step 3: Calculate trait correlations
        print("Calculating trait correlations...")
        trait_data = self._prepare_trait_data(rust_results)
        corr_matrix, _ = self.stat_analyzer.calculate_trait_correlations(trait_data)
        
        # Step 4: Apply adaptive thresholds
        print("Applying adaptive confidence thresholds...")
        filtered_genes = []
        threshold_details = []
        
        for gene_id, gene_data in rust_results["genes"].items():
            # Get gene context
            gene_context = gene_contexts[gene_id]
            
            # Get method scores for this gene
            gene_method_scores = method_scores[gene_id]
            
            # Get traits associated with this gene
            gene_traits = gene_data.get("traits", [])
            n_traits = len(gene_traits)
            
            # Get trait correlations for this gene's traits
            if n_traits > 1:
                trait_indices = [trait_data.columns.get_loc(t) for t in gene_traits 
                               if t in trait_data.columns]
                trait_corr = corr_matrix.iloc[trait_indices, trait_indices].values
            else:
                trait_corr = None
            
            # Calculate adaptive threshold
            threshold = self.protocol.get_adaptive_threshold(
                gene_context=gene_context,
                n_traits=n_traits,
                method_scores=gene_method_scores,
                trait_correlations=trait_corr
            )
            
            # Apply threshold
            overall_confidence = gene_data.get("confidence_score", 0)
            if overall_confidence >= threshold:
                filtered_genes.append({
                    "gene_id": gene_id,
                    "confidence_score": overall_confidence,
                    "adaptive_threshold": threshold,
                    "traits": gene_traits,
                    "passed": True
                })
            
            threshold_details.append({
                "gene_id": gene_id,
                "confidence_score": overall_confidence,
                "adaptive_threshold": threshold,
                "n_traits": n_traits,
                "complexity_score": gene_context.complexity_score,
                "passed": overall_confidence >= threshold
            })
        
        # Step 5: Validation if known genes provided
        validation_results = None
        if known_pleiotropic_genes:
            print("Validating against known pleiotropic genes...")
            validation_results = self._validate_predictions(
                threshold_details, known_pleiotropic_genes
            )
            
            # Update protocol based on validation
            self.protocol.update_from_validation(validation_results["metrics"])
        
        # Step 6: Estimate background noise
        all_scores = [d["confidence_score"] for d in threshold_details]
        self.protocol.estimate_background_noise(all_scores)
        
        # Step 7: Compile results
        results = {
            "phase": self.protocol.current_phase.value,
            "total_genes_analyzed": len(threshold_details),
            "pleiotropic_genes_detected": len(filtered_genes),
            "average_threshold": np.mean([d["adaptive_threshold"] for d in threshold_details]),
            "background_noise_level": self.protocol.background_noise_level,
            "filtered_genes": filtered_genes,
            "threshold_details": threshold_details,
            "validation_results": validation_results,
            "trait_correlations": corr_matrix.to_dict() if len(corr_matrix) < 20 else "Too large to display"
        }
        
        # Save protocol state
        state_path = Path(__file__).parent / "protocol_state.json"
        self.protocol.save_protocol_state(str(state_path))
        
        return results
    
    def _extract_gene_contexts(self, rust_results: Dict) -> Dict[str, GeneContext]:
        """Extract gene context information from Rust results."""
        contexts = {}
        
        for gene_id, gene_data in rust_results["genes"].items():
            # Extract or calculate gene properties
            sequence = gene_data.get("sequence", "")
            length = len(sequence)
            
            # Calculate GC content
            gc_count = sequence.count('G') + sequence.count('C')
            gc_content = gc_count / length if length > 0 else 0.5
            
            # Estimate complexity (simplified)
            unique_codons = len(set(sequence[i:i+3] for i in range(0, len(sequence)-2, 3)))
            codon_complexity = unique_codons / 64.0  # Normalize to 0-1
            
            # Count regulatory elements (simplified)
            regulatory_elements = len(gene_data.get("regulatory_regions", []))
            
            # Placeholder for conservation score
            evolutionary_conservation = np.random.uniform(0.5, 0.9)  # Would use real data
            
            contexts[gene_id] = GeneContext(
                gene_id=gene_id,
                length=length,
                gc_content=gc_content,
                codon_complexity=codon_complexity,
                regulatory_elements=regulatory_elements,
                evolutionary_conservation=evolutionary_conservation
            )
        
        return contexts
    
    def _extract_method_scores(self, rust_results: Dict) -> Dict[str, Dict[str, float]]:
        """Extract individual method scores from results."""
        method_scores = {}
        
        for gene_id, gene_data in rust_results["genes"].items():
            # Extract scores from different analysis methods
            scores = gene_data.get("method_scores", {})
            
            # Ensure all methods are represented
            method_scores[gene_id] = {
                "frequency_analysis": scores.get("frequency_analysis", 0.5),
                "pattern_detection": scores.get("pattern_detection", 0.5),
                "regulatory_context": scores.get("regulatory_context", 0.5),
                "statistical_significance": scores.get("statistical_significance", 0.5)
            }
        
        return method_scores
    
    def _prepare_trait_data(self, rust_results: Dict) -> pd.DataFrame:
        """Prepare trait data for correlation analysis."""
        # Extract trait associations
        trait_gene_matrix = {}
        all_traits = set()
        
        for gene_id, gene_data in rust_results["genes"].items():
            traits = gene_data.get("traits", [])
            all_traits.update(traits)
            trait_gene_matrix[gene_id] = traits
        
        # Create binary matrix
        trait_list = sorted(all_traits)
        gene_list = sorted(trait_gene_matrix.keys())
        
        data = []
        for gene in gene_list:
            row = [1 if trait in trait_gene_matrix[gene] else 0 for trait in trait_list]
            data.append(row)
        
        return pd.DataFrame(data, index=gene_list, columns=trait_list).T
    
    def _validate_predictions(self, 
                            threshold_details: List[Dict],
                            known_genes: List[str]) -> Dict:
        """Validate predictions against known pleiotropic genes."""
        # Create binary arrays
        all_genes = [d["gene_id"] for d in threshold_details]
        y_true = np.array([1 if gene in known_genes else 0 for gene in all_genes])
        y_scores = np.array([d["confidence_score"] for d in threshold_details])
        y_pred = np.array([1 if d["passed"] else 0 for d in threshold_details])
        
        # Calculate average threshold used
        avg_threshold = np.mean([d["adaptive_threshold"] for d in threshold_details])
        
        # Calculate validation metrics
        metrics = ValidationMetrics.calculate(y_true, y_scores, avg_threshold)
        
        # Find optimal thresholds
        optimal_f1, f1_value = self.protocol.optimize_threshold_by_metric(
            y_true, y_scores, metric="f1"
        )
        optimal_mcc, mcc_value = self.protocol.optimize_threshold_by_metric(
            y_true, y_scores, metric="mcc"
        )
        
        # ROC analysis
        roc_analysis = self.protocol.generate_roc_analysis(y_true, y_scores)
        
        return {
            "metrics": metrics,
            "optimal_thresholds": {
                "f1": {"threshold": optimal_f1, "value": f1_value},
                "mcc": {"threshold": optimal_mcc, "value": mcc_value}
            },
            "roc_analysis": {
                "auc": roc_analysis["auc"],
                "optimal_threshold": roc_analysis["optimal_threshold"]
            },
            "detected_known_genes": [
                gene for gene, pred in zip(all_genes, y_pred) 
                if pred == 1 and gene in known_genes
            ],
            "missed_known_genes": [
                gene for gene, pred in zip(all_genes, y_pred) 
                if pred == 0 and gene in known_genes
            ]
        }
    
    def run_iterative_analysis(self,
                             fasta_path: str,
                             traits_path: str,
                             known_genes: List[str],
                             max_iterations: int = 20) -> Dict:
        """
        Run iterative analysis with learning across phases.
        
        Args:
            fasta_path: Path to genome FASTA file
            traits_path: Path to traits JSON file
            known_genes: List of known pleiotropic genes
            max_iterations: Maximum iterations per phase
            
        Returns:
            Summary of iterative analysis
        """
        phases = [
            DetectionPhase.DISCOVERY,
            DetectionPhase.VALIDATION,
            DetectionPhase.CONFIRMATION,
            DetectionPhase.PRODUCTION
        ]
        
        all_results = []
        
        for phase in phases:
            print(f"\n{'='*60}")
            print(f"Starting {phase.value} phase")
            print(f"{'='*60}")
            
            self.protocol.update_phase(phase)
            
            best_f1 = 0
            iterations_without_improvement = 0
            
            for iteration in range(max_iterations):
                print(f"\nIteration {iteration + 1}/{max_iterations}")
                
                # Run analysis
                results = self.analyze_genome(fasta_path, traits_path, known_genes)
                
                if results["validation_results"]:
                    current_f1 = results["validation_results"]["metrics"].f1_score
                    print(f"F1 Score: {current_f1:.3f}")
                    
                    # Check for convergence
                    if current_f1 > best_f1:
                        best_f1 = current_f1
                        iterations_without_improvement = 0
                    else:
                        iterations_without_improvement += 1
                    
                    # Early stopping
                    if iterations_without_improvement >= 5:
                        print(f"Converged after {iteration + 1} iterations")
                        break
                
                all_results.append({
                    "phase": phase.value,
                    "iteration": iteration + 1,
                    "results": results
                })
        
        return {
            "total_iterations": len(all_results),
            "final_state": self.protocol.phase_configs,
            "performance_evolution": [
                {
                    "phase": r["phase"],
                    "iteration": r["iteration"],
                    "f1_score": r["results"]["validation_results"]["metrics"].f1_score
                    if r["results"]["validation_results"] else None
                }
                for r in all_results
            ],
            "final_detected_genes": all_results[-1]["results"]["filtered_genes"]
        }


def main():
    """Example usage of the integrated pipeline."""
    # Initialize pipeline with drug discovery configuration
    pipeline = PleiotropyDetectionPipeline(config_scenario="drug_target_discovery")
    
    # Example paths (these would be real files in practice)
    fasta_path = "/home/murr2k/projects/agentic/pleiotropy/test_ecoli_sample.fasta"
    traits_path = "/home/murr2k/projects/agentic/pleiotropy/test_traits.json"
    
    # Known pleiotropic genes for E. coli
    known_genes = ["crp", "fis", "ihfA", "ihfB", "fnr", "arcA", "narL", "ompR", "phoB", "cpxR"]
    
    # Run single analysis
    print("Running single-phase analysis...")
    results = pipeline.analyze_genome(fasta_path, traits_path, known_genes)
    
    print(f"\nAnalysis complete!")
    print(f"Total genes analyzed: {results['total_genes_analyzed']}")
    print(f"Pleiotropic genes detected: {results['pleiotropic_genes_detected']}")
    print(f"Average adaptive threshold: {results['average_threshold']:.3f}")
    
    if results["validation_results"]:
        metrics = results["validation_results"]["metrics"]
        print(f"\nValidation Results:")
        print(f"  Precision: {metrics.precision:.3f}")
        print(f"  Recall: {metrics.recall:.3f}")
        print(f"  F1 Score: {metrics.f1_score:.3f}")
        print(f"  MCC: {metrics.mcc:.3f}")
    
    # Save detailed results
    output_path = Path(__file__).parent / "analysis_results.json"
    with open(output_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in results.items()
        }
        json.dump(json_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()