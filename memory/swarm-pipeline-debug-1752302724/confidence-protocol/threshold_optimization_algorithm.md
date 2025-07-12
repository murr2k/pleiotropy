# Adaptive Threshold Optimization Algorithm

## Overview

The adaptive threshold optimization algorithm dynamically adjusts confidence thresholds for pleiotropic gene detection based on multiple contextual factors and validation feedback. This document describes the mathematical framework and implementation details.

## Algorithm Components

### 1. Multi-Phase Detection Strategy

The algorithm operates in four distinct phases, each with different stringency levels:

```
Phase               Base Threshold   Range        Purpose
DISCOVERY           0.35            [0.25-0.45]  Initial broad search
VALIDATION          0.55            [0.45-0.65]  Filter candidates
CONFIRMATION        0.75            [0.65-0.85]  High-confidence detection
PRODUCTION          0.70            [0.60-0.80]  Balanced operation
```

### 2. Threshold Calculation Formula

The adaptive threshold `T_adaptive` is calculated as:

```
T_adaptive = T_base × C_complexity × C_traits × C_correlation × C_noise × C_bayesian × C_ensemble
```

Where:
- `T_base`: Base threshold for current phase
- `C_complexity`: Gene complexity adjustment factor
- `C_traits`: Number of traits adjustment factor
- `C_correlation`: Trait correlation adjustment factor
- `C_noise`: Background noise adjustment factor
- `C_bayesian`: Bayesian prior adjustment factor
- `C_ensemble`: Ensemble confidence adjustment factor

### 3. Adjustment Factors

#### 3.1 Gene Complexity Factor (C_complexity)

```python
complexity_score = 0.3 × length_factor + 
                  0.2 × gc_deviation + 
                  0.3 × codon_complexity + 
                  0.2 × regulatory_elements

C_complexity = {
    0.95  if complexity_score > 0.7  # Complex genes
    1.00  if 0.5 ≤ complexity_score ≤ 0.7
    1.05  if complexity_score < 0.5  # Simple genes
}
```

#### 3.2 Number of Traits Factor (C_traits)

```python
C_traits = {
    1.10  if n_traits ≤ 2    # Few traits
    1.00  if 3 ≤ n_traits ≤ 5
    0.95  if 6 ≤ n_traits ≤ 10
    0.90  if n_traits > 10    # Many traits
}
```

#### 3.3 Trait Correlation Factor (C_correlation)

```python
avg_correlation = mean(|correlation_matrix|)

C_correlation = {
    1.15  if avg_correlation > 0.7   # Highly correlated
    1.05  if 0.5 < avg_correlation ≤ 0.7
    1.00  if avg_correlation ≤ 0.5   # Independent traits
}
```

#### 3.4 Background Noise Factor (C_noise)

```python
C_noise = 1.0 + (noise_level × 0.5)
```

Where `noise_level` is estimated using Gaussian Mixture Models or known negative controls.

#### 3.5 Bayesian Adjustment Factor (C_bayesian)

Using Bayes' theorem to update beliefs:

```python
P(pleiotropic|evidence) = P(evidence|pleiotropic) × P(pleiotropic) / P(evidence)

likelihood = sigmoid(avg_method_score, center=0.5, steepness=10)
posterior = (likelihood × prior) / (likelihood × prior + (1-likelihood) × (1-prior))

C_bayesian = {
    0.9 + (0.5 - posterior) × 0.2  if posterior > 0.5
    1.0 + (0.5 - posterior) × 0.2  if posterior ≤ 0.5
}
```

#### 3.6 Ensemble Confidence Factor (C_ensemble)

```python
ensemble_confidence = Σ(method_score × method_weight) / Σ(method_weight)

C_ensemble = {
    0.9   if ensemble_confidence > 0.8   # High confidence
    1.0   if 0.4 ≤ ensemble_confidence ≤ 0.8
    1.1   if ensemble_confidence < 0.4   # Low confidence
}
```

### 4. Learning and Adaptation

#### 4.1 Validation Feedback Loop

After each validation round:

```python
f1_error = observed_f1 - target_f1
threshold_adjustment = -f1_error × learning_rate
T_base_new = clip(T_base + threshold_adjustment, T_min, T_max)
```

#### 4.2 Prior Probability Update

```python
observed_rate = (TP + FN) / (TP + FN + TN + FP)
prior_new = 0.9 × prior_old + 0.1 × observed_rate
```

### 5. Optimization Strategies

#### 5.1 ROC-based Optimization

Find optimal threshold using Youden's J statistic:

```python
J = sensitivity + specificity - 1 = TPR - FPR
optimal_threshold = threshold[argmax(J)]
```

#### 5.2 F1-based Optimization

Maximize F1 score across threshold range:

```python
F1 = 2 × (precision × recall) / (precision + recall)
optimal_threshold = threshold[argmax(F1)]
```

#### 5.3 Matthews Correlation Coefficient (MCC) Optimization

For balanced performance:

```python
MCC = (TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
optimal_threshold = threshold[argmax(MCC)]
```

### 6. Background Noise Estimation

Using Gaussian Mixture Model to separate signal from noise:

```python
GMM(n_components=2).fit(confidence_scores)
noise_component = component_with_lower_mean
noise_level = mean + 1.645 × std  # 95th percentile
```

### 7. Implementation Considerations

#### 7.1 Computational Efficiency

- Cache adjustment factors for repeated genes
- Use vectorized operations for batch processing
- Implement early stopping in optimization loops

#### 7.2 Robustness

- Clip thresholds to valid ranges
- Handle edge cases (no traits, missing data)
- Validate input parameters

#### 7.3 Interpretability

- Log all adjustment factors
- Provide reasoning for threshold decisions
- Generate confidence intervals

## Algorithm Pseudocode

```python
function calculate_adaptive_threshold(gene, traits, methods, phase):
    # Get base configuration
    config = phase_configs[phase]
    threshold = config.base_threshold
    
    # Calculate context
    gene_context = analyze_gene(gene)
    trait_correlations = calculate_correlations(traits)
    
    # Apply adjustments
    threshold *= adjust_for_complexity(gene_context)
    threshold *= adjust_for_trait_count(len(traits))
    threshold *= adjust_for_correlations(trait_correlations)
    threshold *= adjust_for_noise(background_noise_level)
    threshold *= calculate_bayesian_factor(methods, gene_context)
    threshold *= calculate_ensemble_factor(methods)
    
    # Apply bounds
    threshold = clip(threshold, config.min, config.max)
    
    return threshold
```

## Performance Metrics

### Expected Performance by Phase

| Phase | Precision | Recall | F1 Score | False Discovery Rate |
|-------|-----------|--------|----------|---------------------|
| Discovery | 0.40-0.60 | 0.80-0.95 | 0.50-0.70 | 0.40-0.60 |
| Validation | 0.60-0.80 | 0.60-0.80 | 0.60-0.80 | 0.20-0.40 |
| Confirmation | 0.80-0.95 | 0.40-0.60 | 0.50-0.70 | 0.05-0.20 |
| Production | 0.70-0.85 | 0.65-0.80 | 0.70-0.80 | 0.15-0.30 |

### Convergence Properties

The algorithm typically converges within:
- 5-10 validation rounds for stable datasets
- 15-20 rounds for noisy or evolving datasets
- Learning rate decay: `η_t = η_0 / (1 + decay_rate × t)`

## References

1. Youden, W.J. (1950). "Index for rating diagnostic tests"
2. Matthews, B.W. (1975). "Comparison of predicted and observed secondary structure"
3. Davis, J. & Goadrich, M. (2006). "The relationship between Precision-Recall and ROC curves"