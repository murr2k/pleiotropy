# Data Provenance Documentation

**Last Updated**: January 13, 2025  
**Purpose**: Complete provenance tracking for all datasets in the Genomic Pleiotropy Cryptanalysis project

## Directory Structure

```
data/
├── real_experiments/        # REAL genomic analyses
├── test_data/              # SIMULATED data for testing only
└── simulated_archive/      # ARCHIVED simulated experiments
```

## 🧬 REAL EXPERIMENTS

### 1. Escherichia coli K-12
- **Location**: `data/real_experiments/ecoli_k12_experiment/`
- **Source**: NCBI Reference Sequence NC_000913.3
- **Date**: July 12, 2025
- **Method**: Actual genomic analysis using genomic_cryptanalysis binary
- **Input Genome**: 4.64 MB FASTA file
- **Analysis Type**: REAL - NeuroDNA v0.0.2 with cryptanalytic patterns
- **Results**: 
  - 1 pleiotropic element detected
  - Traits: regulatory, stress_response
  - Confidence: 0.75
- **Validation**: Can be reproduced with actual E. coli genome

### 2. Salmonella enterica serovar Typhimurium
- **Location**: `data/real_experiments/salmonella_typhimurium_experiment/`
- **Source**: Downloaded from NCBI (strain details in download script)
- **Date**: July 12, 2025  
- **Method**: Actual genomic analysis
- **Input Genome**: 5.01 MB FASTA file (`salmonella_genome.fasta`)
- **Analysis Type**: REAL - NeuroDNA v0.0.2
- **Results**:
  - 2 pleiotropic elements (chromosome + plasmid)
  - Traits: regulatory, stress_response
  - Confidence: 0.775 (average)
- **Validation**: Reproducible with genome accession number

### 3. Pseudomonas aeruginosa PAO1
- **Location**: `data/real_experiments/pseudomonas_aeruginosa_experiment/`
- **Source**: Downloaded from NCBI PAO1 reference
- **Date**: July 12, 2025
- **Method**: Actual genomic analysis
- **Input Genome**: 6.26 MB FASTA file (`pseudomonas_genome.fasta`)
- **Analysis Type**: REAL - NeuroDNA v0.0.2
- **Results**:
  - 1 pleiotropic element
  - Traits: regulatory, stress_response, carbon_metabolism, motility, structural
  - Confidence: 0.75
- **Validation**: Reproducible with PAO1 reference genome

## 🧪 TEST DATA (Simulated for Regression Testing)

### Location: `data/test_data/`

**⚠️ WARNING**: This directory contains SIMULATED data for testing purposes only.

### Contents:
- `regression_test_dataset.json` - 5 test cases for different bacterial lifestyles
- `README.md` - Clear warning about simulated nature
- Test organisms:
  1. TEST_ESCHERICHIA_COLI - Standard test case
  2. TEST_BACILLUS_SUBTILIS - Soil bacterium test
  3. TEST_STAPHYLOCOCCUS_AUREUS - Pathogen test
  4. TEST_SYNECHOCYSTIS_SP. - Photosynthetic test
  5. TEST_DEINOCOCCUS_RADIODURANS - Extremophile test

### Purpose:
- Regression testing
- CI/CD pipeline validation
- Performance benchmarking
- Method development testing

## 📦 SIMULATED ARCHIVE

### Location: `data/simulated_archive/batch_experiment_20_genomes_20250712_181857/`

**Status**: ARCHIVED - Not for scientific use

### Contents:
- 20 simulated bacterial genome analyses
- Generated using `simulate_batch_analysis.py`
- Based on probabilistic models, not real genomic data
- Includes visualization and statistical summaries

### Why Archived:
- Mixed with real data in original reports
- Led to inflated success rates and statistics
- Retained for historical reference only

## Data Usage Guidelines

### For Scientific Analysis:
✅ Use ONLY files from `data/real_experiments/`
❌ Do NOT use test_data or simulated_archive

### For Testing/Development:
✅ Use `data/test_data/regression_test_dataset.json`
✅ Clearly mark any results as "TEST DATA"

### For Benchmarking:
✅ Use real experiments for accuracy benchmarks
✅ Use test data for performance benchmarks

## Verification Commands

```bash
# Check real experiment data
ls -la data/real_experiments/*/analysis_results.json

# Verify test data header
head -n 10 data/test_data/regression_test_dataset.json | grep "TEST DATASET"

# Count real vs simulated
echo "Real experiments: $(ls -d data/real_experiments/*/ | wc -l)"
echo "Test cases: $(grep -c "TEST_" data/test_data/regression_test_dataset.json)"
```

## Future Data Management

1. **New Real Experiments**: Place in `data/real_experiments/organism_name_YYYYMMDD/`
2. **New Test Cases**: Add to `regression_test_dataset.json` with TEST_ prefix
3. **Simulations**: Clearly label and keep separate from real data
4. **Version Control**: Update this document with each new dataset

---

**Note**: This structure resolves GitHub Issue #10 by clearly separating real and simulated data.