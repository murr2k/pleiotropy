# Test Dataset - SIMULATED DATA

## ⚠️ WARNING: This is NOT Real Experimental Data

This directory contains **simulated data** used exclusively for:
- Regression testing
- Method validation
- Performance benchmarking
- CI/CD pipeline testing

## Test Cases Included

1. **Escherichia coli** - Standard test organism
2. **Bacillus subtilis** - Soil bacterium lifestyle
3. **Staphylococcus aureus** - Pathogenic lifestyle
4. **Synechocystis sp.** - Photosynthetic lifestyle
5. **Deinococcus radiodurans** - Extremophile lifestyle

## Usage

```python
# Example usage in tests
import json

with open('regression_test_dataset.json', 'r') as f:
    test_data = json.load(f)
    
# Check that it's test data
assert "TEST DATASET" in test_data['metadata']['description']
assert test_data['metadata']['warning'] is not None

# Run tests
for test_case in test_data['test_cases']:
    assert test_case['test_id'].startswith('TEST_')
    # Perform regression tests...
```

## Data Structure

Each test case contains:
- `test_id`: Unique identifier starting with "TEST_"
- `organism`: Species name (simulated)
- `expected_results`: Known outputs for regression testing
- `test_notes`: Description of what this tests

**Created**: 2025-07-13
**Source**: Extracted from batch simulations
**Do NOT use for scientific analysis!**
