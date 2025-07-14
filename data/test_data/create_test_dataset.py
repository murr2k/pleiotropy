#!/usr/bin/env python3
"""
Create a clearly labeled test dataset from simulated data for regression testing
"""

import json
import os
from datetime import datetime

def create_test_dataset():
    """Extract a subset of simulated data for testing purposes"""
    
    # Load simulated batch data
    with open('../../batch_experiment_20_genomes_20250712_181857/batch_simulation_results.json', 'r') as f:
        simulated_data = json.load(f)
    
    # Select 5 diverse test cases
    test_organisms = [
        "Escherichia coli",      # Standard test organism
        "Bacillus subtilis",     # Soil bacterium (different lifestyle)
        "Staphylococcus aureus", # Pathogen
        "Synechocystis sp.",     # Photosynthetic
        "Deinococcus radiodurans" # Extremophile
    ]
    
    test_dataset = {
        "metadata": {
            "description": "TEST DATASET - SIMULATED DATA FOR REGRESSION TESTING ONLY",
            "warning": "This is NOT real experimental data. Do not use for scientific conclusions.",
            "created": datetime.now().isoformat(),
            "purpose": "Regression testing and method validation",
            "source": "Simulated using simulate_batch_analysis.py",
            "version": "1.0"
        },
        "test_cases": []
    }
    
    # Extract test cases
    for result in simulated_data:
        genome = result['genome']
        if genome['name'] in test_organisms:
            test_case = {
                "test_id": f"TEST_{genome['name'].replace(' ', '_').upper()}",
                "organism": f"{genome['name']} {genome['strain']}",
                "genome_size_mb": genome['genome_size_mb'],
                "lifestyle": genome['lifestyle'],
                "expected_results": {
                    "success": result['success'],
                    "n_pleiotropic_elements": result['summary']['n_pleiotropic_elements'],
                    "unique_traits": result['summary']['unique_traits'],
                    "avg_confidence": result['summary']['avg_confidence'],
                    "common_traits": result['summary'].get('common_traits', [])
                },
                "pleiotropic_genes": result['pleiotropic_genes'],
                "analysis_time": result['analysis_time'],
                "test_notes": f"Simulated {genome['lifestyle']} organism for testing"
            }
            test_dataset["test_cases"].append(test_case)
    
    # Save test dataset
    with open('regression_test_dataset.json', 'w') as f:
        json.dump(test_dataset, f, indent=2)
    
    # Create README for test data
    readme_content = """# Test Dataset - SIMULATED DATA

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

**Created**: {date}
**Source**: Extracted from batch simulations
**Do NOT use for scientific analysis!**
""".format(date=datetime.now().strftime('%Y-%m-%d'))
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("Test dataset created successfully!")
    print(f"- Created {len(test_dataset['test_cases'])} test cases")
    print("- Files: regression_test_dataset.json, README.md")

if __name__ == "__main__":
    create_test_dataset()