# Data Manifest - Genomic Pleiotropy Cryptanalysis

**Last Updated**: January 13, 2025  
**Purpose**: Complete inventory of all experimental data files

## Overview

This manifest documents all data files used in the Genomic Pleiotropy Cryptanalysis experiments, their locations, and checksums for verification.

## Individual Experiments (3)

### 1. Escherichia coli K-12
- **Directory**: `trial_20250712_023446/`
- **Main Results**: `trial_20250712_023446/analysis_results.json`
- **Additional Files**:
  - `ecoli_results/analysis_results.json` - Detailed analysis
  - `synthetic_results/analysis_results.json` - Synthetic test data
  - `trait_definitions_fixed.json` - Trait patterns used
  - `analysis_stats.json` - Summary statistics
- **Visualizations**: 4 PNG files (confidence, complexity, cooccurrence, frequency)
- **Status**: ✅ Complete

### 2. Salmonella enterica Typhimurium
- **Directory**: `experiment_salmonella_20250712_174618/`
- **Main Results**: `experiment_salmonella_20250712_174618/analysis_results.json`
- **Source Directory**: `experiment_salmonella_20250712_153735/` (original run)
- **Additional Files**:
  - `salmonella_genome.fasta` - Input genome (5.0 MB)
  - `traits_salmonella.json` - Trait definitions
  - `salmonella_analysis_report.md` - Analysis summary
- **Status**: ✅ Restored from alternate location

### 3. Pseudomonas aeruginosa PAO1
- **Directory**: `experiment_pseudomonas_20250712_175007/`
- **Main Results**: `experiment_pseudomonas_20250712_175007/analysis_results.json`
- **Source Directory**: `experiment_pseudomonas_20250712_154530/` (original run)
- **Additional Files**:
  - `pseudomonas_genome.fasta` - Input genome (6.3 MB)
  - `traits_pseudomonas.json` - Trait definitions
  - `three_genome_comparison.md` - Comparative analysis
- **Status**: ✅ Restored from alternate location

## Batch Experiments (20 genomes)

### Directory: `batch_experiment_20_genomes_20250712_181857/`
- **Main Results**: `batch_simulation_results.json` ⚠️ SIMULATED DATA
- **Summary Report**: `comprehensive_statistical_report.md`
- **Analysis Scripts**:
  - `batch_analysis_simulator.py` - Simulation generator
  - `comprehensive_analysis_report.py` - Report generator
- **Outputs**:
  - `comprehensive_analysis_figure.png` - Multi-panel visualization
  - `confidence_distribution.png` - Confidence score distribution
  - `genome_size_correlation.png` - Size vs traits plot
  - `batch_analysis_report.md` - Initial summary
- **CSV Tables**: 5 statistical summary files
- **Status**: ⚠️ Contains simulated data, not real experiments

## File Verification

### Checksums (MD5)
```
dc31b07727cc1caa8c92e463bf562226  trial_20250712_023446/analysis_results.json
aca29a400931beeafc9cff5c753769fa  experiment_salmonella_20250712_174618/analysis_results.json
fadc001a5690287367106c5c27ddd286  experiment_pseudomonas_20250712_175007/analysis_results.json
1d84b435f748635292b99187bd697329  batch_experiment_20_genomes_20250712_181857/batch_simulation_results.json
```

Generated on: January 13, 2025

### File Sizes
- E. coli results: 13,979 bytes
- Salmonella results: 15,117 bytes
- Pseudomonas results: 19,352 bytes
- Batch simulation: ~100 KB

## Data Integrity Notes

1. **Individual Experiments**: All 3 files have been located and placed in expected directories
2. **Batch Data**: Clearly marked as SIMULATED - not real genomic analysis
3. **Original Locations**: Some files were in differently named directories from different run timestamps
4. **Completeness**: All expected JSON result files are now present

## Quality Control Findings

During QC validation, it was discovered that:
- Files existed but were in unexpected locations
- Directory naming inconsistencies caused confusion
- Batch data is simulated, not real experimental results

## Recommendations

1. Use consistent directory naming: `experiment_ORGANISM_YYYYMMDD_HHMMSS`
2. Implement automated file organization post-analysis
3. Clearly separate real vs simulated data
4. Add file checksums to all outputs
5. Create symlinks for backward compatibility

## Data Availability

All data files are currently stored locally. For scientific reproducibility, these should be:
1. Uploaded to a public repository (e.g., Zenodo, FigShare)
2. Assigned DOIs for permanent reference
3. Linked from the main README
4. Include raw FASTA input files

---

**Note**: This manifest resolves GitHub Issue #15 regarding missing data files. All expected files have been located and are now accessible in their documented locations.