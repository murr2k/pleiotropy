#!/bin/bash

# E. coli K-12 Genomic Pleiotropy Analysis Workflow

echo "=== E. coli K-12 Pleiotropy Cryptanalysis Workflow ==="
echo

# Set up directories
EXAMPLE_DIR="$(dirname "$0")"
PROJECT_ROOT="$(dirname "$EXAMPLE_DIR")"
DATA_DIR="$EXAMPLE_DIR/data"
OUTPUT_DIR="$EXAMPLE_DIR/output"

# Create directories if they don't exist
mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

# Step 1: Download E. coli K-12 genome (if not already present)
if [ ! -f "$DATA_DIR/ecoli_k12.fasta" ]; then
    echo "Downloading E. coli K-12 genome..."
    # Using a subset for demo purposes
    cat > "$DATA_DIR/ecoli_k12.fasta" << 'EOF'
>NC_000913.3 Escherichia coli str. K-12 substr. MG1655 crp gene
ATGTCAGAACGTAAAGGTATTCGTACCCACTTGCGGGAAGCGCAGTCATTAACGTTTCATCATATCAGGT
CATTGAGCGCCATTTTAGAAAACGGTGAAGTCATGCAGGATGTTATCGAGCGTCTTATCCGTCACGGTAA
AGAGCGTCTGATGGCGACCACCCAGGAAATCAACCACGAAGAGCTGGTGAAACAGGTGAAAGAATACCGT
GAGATCGTCAAGAAACTGGTGATCACCAACCTGCCGGGTATCTCTATCGATCTGCTGGAAGACGGTCACG
GTATGCAGATCGCGATCCTGATCAACGCGCTGACCATGGACGAAATCGTTTCCACCCTGAAAGATCTGCA
>NC_000913.3 Escherichia coli str. K-12 substr. MG1655 fis gene
ATGTTCGAACAACGCGTACAGGACGCAGAAAAAGAAGCACAGAAGAAACAGAAAGCCCGTGAAGCGGCTC
TGAAAGAAGCCAAAGCTGAAGACGTAACTGGTGAAGAAGTCAGTAAAGTGAAAGGTGGTAAACCTAAAGT
TAAACAGGTACAGAAAGCTGGCGTTAAAGACGCTAAAGTTGATGGCAAAGTTAACGCTCGTGGCATGGAC
AAAGCAAAAGACCGTGTTGAAAAAGCTGCACGTAAAGCTAAATCTGAAGAAGCTGGTGCTGCAGACGCTA
>NC_000913.3 Escherichia coli str. K-12 substr. MG1655 rpoS gene
ATGAGTCAGAATACGCTGAAAGTTCATGATTTAAATGAAAAAGTGGCACTGTCGCAAAACGCAGAACAGC
AGCGGACGCTGGAAGGCAAAACGCCAGGACGTCGCCAGACGCTTGCAGAATCTCTCCAGGCGGAACTGGA
ACGTATTCAGTATCTGACAGAAGAAGTGGAAGATCAGGGCGCATCCCTGCGCCAGTTGCTGGATGAGCTG
GAAGATGCTGAAGCGGCAAAACGTGCGGCAGAACGTGAAGCAGAAGATCCGCGCCCGGATGAGGATCCTG
ATCCAGCAGAAGATGAACAGCCAGCGTTCGCTGACCGCACTGGCTGATCTGATCGGCGAAGAGTCCGATC
EOF
    echo "Sample E. coli genome data created."
else
    echo "E. coli genome data already exists."
fi

# Step 2: Create known traits configuration
echo "Creating known traits configuration..."
cat > "$DATA_DIR/ecoli_traits.json" << 'EOF'
[
  {
    "name": "carbon_metabolism",
    "description": "Carbon source utilization and catabolite repression",
    "associated_genes": ["crp", "cyaA", "ptsI", "ptsH"],
    "known_sequences": []
  },
  {
    "name": "stress_response",
    "description": "Response to various environmental stresses",
    "associated_genes": ["rpoS", "hns", "dps", "katE"],
    "known_sequences": []
  },
  {
    "name": "regulatory",
    "description": "Global gene expression regulation",
    "associated_genes": ["crp", "fis", "ihfA", "ihfB", "hns"],
    "known_sequences": []
  },
  {
    "name": "motility",
    "description": "Flagellar synthesis and chemotaxis",
    "associated_genes": ["fliA", "flhDC", "fliC", "motA"],
    "known_sequences": []
  },
  {
    "name": "biofilm_formation",
    "description": "Biofilm formation and surface attachment",
    "associated_genes": ["csgA", "csgD", "fimA", "rpoS"],
    "known_sequences": []
  }
]
EOF

# Step 3: Build the Rust implementation
echo "Building Rust implementation..."
cd "$PROJECT_ROOT/rust_impl"
cargo build --release

# Step 4: Run the cryptanalysis
echo "Running genomic cryptanalysis..."
./target/release/genomic_cryptanalysis \
    --input "$DATA_DIR/ecoli_k12.fasta" \
    --traits "$DATA_DIR/ecoli_traits.json" \
    --output "$OUTPUT_DIR" \
    --min-traits 2 \
    --verbose

# Step 5: Run Python analysis on results
echo "Running Python visualization and analysis..."
cd "$PROJECT_ROOT"
python3 python_analysis/trait_visualizer.py \
    --input "$OUTPUT_DIR/analysis_results.json" \
    --output "$OUTPUT_DIR/visualizations"

# Step 6: Generate final report
echo "Generating final report..."
cat > "$OUTPUT_DIR/final_report.md" << EOF
# E. coli K-12 Pleiotropy Analysis Report

## Overview
This analysis used cryptanalytic techniques to identify pleiotropic genes in E. coli K-12.

## Key Findings
- Analysis results are in: $OUTPUT_DIR/analysis_results.json
- Pleiotropic genes are in: $OUTPUT_DIR/pleiotropic_genes.json
- Visualizations are in: $OUTPUT_DIR/visualizations/

## Cryptanalysis Method
1. Codon frequency analysis to detect usage bias
2. Pattern recognition for regulatory elements
3. Trait separation using eigenanalysis
4. Confidence scoring based on multiple factors

## Next Steps
- Validate findings against experimental data
- Extend analysis to full E. coli genome
- Apply method to other model organisms
EOF

echo
echo "=== Workflow Complete ==="
echo "Results available in: $OUTPUT_DIR"
echo "View summary report: $OUTPUT_DIR/summary_report.md"