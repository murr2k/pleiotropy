"""
Adaptive Confidence Protocol for Pleiotropic Gene Detection

This module implements a dynamic confidence threshold system that adapts
based on detection phase, gene characteristics, and validation feedback.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from scipy import stats
from sklearn.metrics import precision_recall_curve, roc_curve, auc, matthews_corrcoef
from sklearn.mixture import GaussianMixture
import warnings


class DetectionPhase(Enum):
    """Phases of pleiotropic gene detection with different stringency levels."""
    DISCOVERY = "discovery"
    VALIDATION = "validation"
    CONFIRMATION = "confirmation"
    PRODUCTION = "production"


@dataclass
class ThresholdConfig:
    """Configuration for adaptive thresholds."""
    phase: DetectionPhase
    base_threshold: float
    min_threshold: float
    max_threshold: float
    adjustment_factors: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "phase": self.phase.value,
            "base_threshold": self.base_threshold,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
            "adjustment_factors": self.adjustment_factors
        }


@dataclass
class GeneContext:
    """Context information for a gene affecting confidence thresholds."""
    gene_id: str
    length: int
    gc_content: float
    codon_complexity: float
    regulatory_elements: int
    evolutionary_conservation: float
    known_pleiotropic: bool = False
    
    @property
    def complexity_score(self) -> float:
        """Calculate overall gene complexity score."""
        # Normalize factors to 0-1 range
        length_factor = min(self.length / 5000, 1.0)  # Genes > 5kb are max complexity
        gc_factor = abs(self.gc_content - 0.5) * 2  # Deviation from 50% GC
        regulatory_factor = min(self.regulatory_elements / 10, 1.0)
        
        # Weighted combination
        complexity = (
            0.3 * length_factor +
            0.2 * gc_factor +
            0.3 * self.codon_complexity +
            0.2 * regulatory_factor
        )
        
        return complexity


@dataclass
class ValidationMetrics:
    """Metrics from validation against known pleiotropic genes."""
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    mcc: float
    auc_roc: float
    auc_pr: float
    
    @classmethod
    def calculate(cls, y_true: np.ndarray, y_scores: np.ndarray, threshold: float):
        """Calculate validation metrics."""
        y_pred = (y_scores >= threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate MCC
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # Calculate AUC scores
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_roc = auc(fpr, tpr)
        
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
        auc_pr = auc(recall_curve, precision_curve)
        
        return cls(
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            mcc=mcc,
            auc_roc=auc_roc,
            auc_pr=auc_pr
        )


class AdaptiveConfidenceProtocol:
    """
    Main class implementing adaptive confidence threshold system.
    
    This protocol dynamically adjusts detection thresholds based on:
    - Current detection phase
    - Gene complexity and characteristics
    - Number and correlation of traits
    - Background noise level
    - Validation feedback
    """
    
    def __init__(self):
        """Initialize the adaptive confidence protocol."""
        self.phase_configs = {
            DetectionPhase.DISCOVERY: ThresholdConfig(
                phase=DetectionPhase.DISCOVERY,
                base_threshold=0.35,
                min_threshold=0.25,
                max_threshold=0.45
            ),
            DetectionPhase.VALIDATION: ThresholdConfig(
                phase=DetectionPhase.VALIDATION,
                base_threshold=0.55,
                min_threshold=0.45,
                max_threshold=0.65
            ),
            DetectionPhase.CONFIRMATION: ThresholdConfig(
                phase=DetectionPhase.CONFIRMATION,
                base_threshold=0.75,
                min_threshold=0.65,
                max_threshold=0.85
            ),
            DetectionPhase.PRODUCTION: ThresholdConfig(
                phase=DetectionPhase.PRODUCTION,
                base_threshold=0.70,
                min_threshold=0.60,
                max_threshold=0.80
            )
        }
        
        self.current_phase = DetectionPhase.DISCOVERY
        self.validation_history: List[ValidationMetrics] = []
        self.background_noise_level = 0.1
        self.learning_rate = 0.1
        
        # Ensemble methods weights
        self.ensemble_weights = {
            "frequency_analysis": 0.3,
            "pattern_detection": 0.25,
            "regulatory_context": 0.25,
            "statistical_significance": 0.2
        }
        
        # Prior probability estimates
        self.prior_pleiotropic_probability = 0.05  # ~5% of genes are pleiotropic
        
    def get_adaptive_threshold(self,
                             gene_context: GeneContext,
                             n_traits: int,
                             method_scores: Dict[str, float],
                             trait_correlations: Optional[np.ndarray] = None) -> float:
        """
        Calculate adaptive threshold for a specific gene and context.
        
        Args:
            gene_context: Context information about the gene
            n_traits: Number of traits being tested
            method_scores: Confidence scores from different methods
            trait_correlations: Correlation matrix between traits
            
        Returns:
            Adaptive threshold value
        """
        config = self.phase_configs[self.current_phase]
        threshold = config.base_threshold
        
        # 1. Adjust for gene complexity
        complexity_adjustment = self._adjust_for_complexity(gene_context)
        threshold *= complexity_adjustment
        
        # 2. Adjust for number of traits
        trait_adjustment = self._adjust_for_trait_count(n_traits)
        threshold *= trait_adjustment
        
        # 3. Adjust for trait correlations
        if trait_correlations is not None:
            correlation_adjustment = self._adjust_for_correlations(trait_correlations)
            threshold *= correlation_adjustment
        
        # 4. Adjust for background noise
        noise_adjustment = 1.0 + (self.background_noise_level * 0.5)
        threshold *= noise_adjustment
        
        # 5. Bayesian adjustment based on prior knowledge
        if gene_context.known_pleiotropic:
            # Lower threshold for known pleiotropic genes
            threshold *= 0.85
        else:
            # Apply Bayesian adjustment
            bayesian_factor = self._calculate_bayesian_adjustment(
                method_scores, gene_context
            )
            threshold *= bayesian_factor
        
        # 6. Apply ensemble voting adjustment
        ensemble_confidence = self._calculate_ensemble_confidence(method_scores)
        if ensemble_confidence > 0.8:
            threshold *= 0.9  # More confident, lower threshold
        elif ensemble_confidence < 0.4:
            threshold *= 1.1  # Less confident, higher threshold
        
        # Ensure threshold stays within bounds
        threshold = np.clip(threshold, config.min_threshold, config.max_threshold)
        
        return threshold
    
    def _adjust_for_complexity(self, gene_context: GeneContext) -> float:
        """Adjust threshold based on gene complexity."""
        complexity = gene_context.complexity_score
        
        # More complex genes need slightly lower thresholds
        # as they're more likely to be pleiotropic
        if complexity > 0.7:
            return 0.95
        elif complexity > 0.5:
            return 1.0
        else:
            return 1.05
    
    def _adjust_for_trait_count(self, n_traits: int) -> float:
        """Adjust threshold based on number of traits."""
        if n_traits <= 2:
            return 1.1  # Fewer traits, higher threshold
        elif n_traits <= 5:
            return 1.0
        elif n_traits <= 10:
            return 0.95
        else:
            return 0.9  # Many traits, lower threshold
    
    def _adjust_for_correlations(self, trait_correlations: np.ndarray) -> float:
        """Adjust threshold based on trait correlations."""
        # Calculate average absolute correlation
        upper_triangle = np.triu(trait_correlations, k=1)
        avg_correlation = np.mean(np.abs(upper_triangle[upper_triangle != 0]))
        
        # High correlation means traits are redundant, need higher threshold
        if avg_correlation > 0.7:
            return 1.15
        elif avg_correlation > 0.5:
            return 1.05
        else:
            return 1.0
    
    def _calculate_bayesian_adjustment(self,
                                     method_scores: Dict[str, float],
                                     gene_context: GeneContext) -> float:
        """Calculate Bayesian adjustment factor."""
        # Calculate likelihood based on method scores
        avg_score = np.mean(list(method_scores.values()))
        
        # Simple Bayesian update
        # P(pleiotropic|evidence) = P(evidence|pleiotropic) * P(pleiotropic) / P(evidence)
        likelihood_pleiotropic = self._sigmoid(avg_score, 0.5, 10)
        likelihood_not_pleiotropic = 1 - likelihood_pleiotropic
        
        # Consider gene context in prior
        prior = self.prior_pleiotropic_probability
        if gene_context.evolutionary_conservation > 0.8:
            prior *= 1.5  # Conserved genes more likely pleiotropic
        
        # Bayesian update
        posterior = (likelihood_pleiotropic * prior) / (
            likelihood_pleiotropic * prior + 
            likelihood_not_pleiotropic * (1 - prior)
        )
        
        # Convert posterior to adjustment factor
        if posterior > 0.5:
            return 0.9 + (0.5 - posterior) * 0.2
        else:
            return 1.0 + (0.5 - posterior) * 0.2
    
    def _calculate_ensemble_confidence(self, method_scores: Dict[str, float]) -> float:
        """Calculate ensemble confidence from multiple methods."""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for method, score in method_scores.items():
            weight = self.ensemble_weights.get(method, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def update_phase(self, new_phase: DetectionPhase):
        """Update current detection phase."""
        self.current_phase = new_phase
    
    def update_from_validation(self, validation_metrics: ValidationMetrics):
        """Update thresholds based on validation results."""
        self.validation_history.append(validation_metrics)
        
        # Adjust base thresholds based on performance
        config = self.phase_configs[self.current_phase]
        
        # Target F1 score
        target_f1 = 0.75
        f1_error = validation_metrics.f1_score - target_f1
        
        # Update base threshold
        adjustment = -f1_error * self.learning_rate
        config.base_threshold += adjustment
        config.base_threshold = np.clip(
            config.base_threshold, 
            config.min_threshold, 
            config.max_threshold
        )
        
        # Update prior probability based on validation
        observed_positive_rate = (
            validation_metrics.true_positives + validation_metrics.false_negatives
        ) / (
            validation_metrics.true_positives + validation_metrics.false_negatives +
            validation_metrics.true_negatives + validation_metrics.false_positives
        )
        
        # Smooth update of prior
        self.prior_pleiotropic_probability = (
            0.9 * self.prior_pleiotropic_probability + 
            0.1 * observed_positive_rate
        )
    
    def estimate_background_noise(self, 
                                confidence_scores: List[float],
                                known_negatives: Optional[List[bool]] = None):
        """Estimate background noise level from confidence scores."""
        scores_array = np.array(confidence_scores)
        
        if known_negatives is not None:
            # Use known negatives to estimate noise
            negative_scores = scores_array[known_negatives]
            if len(negative_scores) > 0:
                self.background_noise_level = np.percentile(negative_scores, 95)
        else:
            # Use Gaussian mixture to separate signal from noise
            gmm = GaussianMixture(n_components=2, random_state=42)
            scores_reshaped = scores_array.reshape(-1, 1)
            gmm.fit(scores_reshaped)
            
            # Assume lower mean component is noise
            means = gmm.means_.flatten()
            noise_component = np.argmin(means)
            noise_mean = means[noise_component]
            noise_std = np.sqrt(gmm.covariances_[noise_component].flatten()[0])
            
            # Set noise level at 95th percentile of noise distribution
            self.background_noise_level = noise_mean + 1.645 * noise_std
    
    def optimize_threshold_by_metric(self,
                                   y_true: np.ndarray,
                                   y_scores: np.ndarray,
                                   metric: str = "f1") -> Tuple[float, float]:
        """
        Find optimal threshold for a given metric.
        
        Args:
            y_true: True labels
            y_scores: Confidence scores
            metric: Metric to optimize ("f1", "mcc", "precision", "recall")
            
        Returns:
            Tuple of (optimal_threshold, metric_value)
        """
        thresholds = np.linspace(0.1, 0.9, 100)
        metric_values = []
        
        for threshold in thresholds:
            metrics = ValidationMetrics.calculate(y_true, y_scores, threshold)
            
            if metric == "f1":
                metric_values.append(metrics.f1_score)
            elif metric == "mcc":
                metric_values.append(metrics.mcc)
            elif metric == "precision":
                metric_values.append(metrics.precision)
            elif metric == "recall":
                metric_values.append(metrics.recall)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        optimal_idx = np.argmax(metric_values)
        return thresholds[optimal_idx], metric_values[optimal_idx]
    
    def generate_roc_analysis(self,
                            y_true: np.ndarray,
                            y_scores: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
        """Generate ROC curve analysis."""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold by Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        return {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": roc_auc,
            "optimal_threshold": optimal_threshold,
            "optimal_tpr": tpr[optimal_idx],
            "optimal_fpr": fpr[optimal_idx]
        }
    
    def _sigmoid(self, x: float, center: float, steepness: float) -> float:
        """Sigmoid function for smooth transitions."""
        return 1 / (1 + np.exp(-steepness * (x - center)))
    
    def save_protocol_state(self, filepath: str):
        """Save current protocol state to JSON."""
        state = {
            "current_phase": self.current_phase.value,
            "phase_configs": {
                phase.value: config.to_dict() 
                for phase, config in self.phase_configs.items()
            },
            "background_noise_level": self.background_noise_level,
            "prior_pleiotropic_probability": self.prior_pleiotropic_probability,
            "ensemble_weights": self.ensemble_weights,
            "learning_rate": self.learning_rate,
            "validation_history": [
                {
                    "precision": vm.precision,
                    "recall": vm.recall,
                    "f1_score": vm.f1_score,
                    "mcc": vm.mcc,
                    "auc_roc": vm.auc_roc,
                    "auc_pr": vm.auc_pr
                }
                for vm in self.validation_history[-10:]  # Keep last 10
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_protocol_state(self, filepath: str):
        """Load protocol state from JSON."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.current_phase = DetectionPhase(state["current_phase"])
        self.background_noise_level = state["background_noise_level"]
        self.prior_pleiotropic_probability = state["prior_pleiotropic_probability"]
        self.ensemble_weights = state["ensemble_weights"]
        self.learning_rate = state["learning_rate"]
        
        # Restore phase configs
        for phase_name, config_dict in state["phase_configs"].items():
            phase = DetectionPhase(phase_name)
            self.phase_configs[phase] = ThresholdConfig(
                phase=phase,
                base_threshold=config_dict["base_threshold"],
                min_threshold=config_dict["min_threshold"],
                max_threshold=config_dict["max_threshold"],
                adjustment_factors=config_dict.get("adjustment_factors", {})
            )


def create_example_usage():
    """Create example usage of the adaptive confidence protocol."""
    
    # Initialize protocol
    protocol = AdaptiveConfidenceProtocol()
    
    # Example gene context
    gene_context = GeneContext(
        gene_id="lacZ",
        length=3075,
        gc_content=0.52,
        codon_complexity=0.75,
        regulatory_elements=5,
        evolutionary_conservation=0.85
    )
    
    # Example method scores
    method_scores = {
        "frequency_analysis": 0.72,
        "pattern_detection": 0.68,
        "regulatory_context": 0.81,
        "statistical_significance": 0.65
    }
    
    # Get adaptive threshold
    threshold = protocol.get_adaptive_threshold(
        gene_context=gene_context,
        n_traits=4,
        method_scores=method_scores
    )
    
    print(f"Adaptive threshold for {gene_context.gene_id}: {threshold:.3f}")
    
    # Simulate validation
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.1, 100)  # 10% positive
    y_scores = np.random.beta(2, 5, 100)
    y_scores[y_true == 1] += 0.3  # Make positives score higher
    
    # Calculate validation metrics
    metrics = ValidationMetrics.calculate(y_true, y_scores, threshold)
    print(f"\nValidation metrics:")
    print(f"  Precision: {metrics.precision:.3f}")
    print(f"  Recall: {metrics.recall:.3f}")
    print(f"  F1 Score: {metrics.f1_score:.3f}")
    print(f"  MCC: {metrics.mcc:.3f}")
    
    # Update protocol from validation
    protocol.update_from_validation(metrics)
    
    # Find optimal threshold
    optimal_threshold, optimal_f1 = protocol.optimize_threshold_by_metric(
        y_true, y_scores, metric="f1"
    )
    print(f"\nOptimal threshold for F1: {optimal_threshold:.3f} (F1={optimal_f1:.3f})")
    
    # Save state
    protocol.save_protocol_state(
        "/home/murr2k/projects/agentic/pleiotropy/memory/swarm-pipeline-debug-1752302724/confidence-protocol/protocol_state.json"
    )


if __name__ == "__main__":
    create_example_usage()