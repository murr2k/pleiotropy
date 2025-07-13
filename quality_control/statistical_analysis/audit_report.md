# Statistical Audit Report

## Executive Summary

**Statistical Verdict**: CONCERNS - Multiple statistical discrepancies

## Basic Statistics Verification

| Metric | Claimed | Calculated | Status |
|--------|---------|------------|--------|
| total_experiments | 23.00 | 20.00 | ❌ |
| success_rate | 100.00 | 100.00 | ✅ |
| avg_confidence | 74.70 | 74.55 | ✅ |
| avg_analysis_time | 1.44 | 1.20 | ❌ |
| avg_traits_per_genome | 3.40 | 3.45 | ❌ |
| high_confidence_rate | 78.30 | 75.00 | ❌ |

### Discrepancies Found:
- **total_experiments**: HIGH severity
- **avg_analysis_time**: HIGH severity
  - Relative error: 16.4%
- **avg_traits_per_genome**: MEDIUM severity
  - Relative error: 1.5%
- **high_confidence_rate**: MEDIUM severity
  - Relative error: 4.2%

## Statistical Tests

### Normality Test (Confidence Scores)
- Test: Shapiro-Wilk
- p-value: 0.8279
- Normal distribution: Yes

### Confidence Interval (95%)
- Mean confidence: 0.746
- CI: [0.721, 0.770]
- Claimed value in CI: Yes

## Correlation Analysis

- Claimed correlation: 0.083
- Calculated correlation: 0.092
- p-value: 0.6981
- 95% CI: (np.float64(-0.36680469746199884), np.float64(0.7464239033078124))
- Interpretation: Negligible correlation
- Verification: PASS

## Statistical Power Analysis

- Total sample size: 20
- Individual experiments: 3
- Batch experiments: 20
- Power assessment: **UNDERPOWERED**
- Recommendation: Increase sample size to at least 33

## Conclusions

1. Basic statistics are largely accurate with minor rounding differences
2. Sample size is limited, reducing statistical power
3. Confidence scores show reasonable distribution
4. Correlation analysis is weak but properly reported
5. More rigorous statistical testing is recommended