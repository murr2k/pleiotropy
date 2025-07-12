# Genomic Pleiotropy Cryptanalysis

A novel approach to understanding genomic pleiotropy by treating it as a cryptanalysis problem. This project implements algorithms to "decrypt" genomic sequences and identify how single genes encode multiple traits.

## 🧬 Overview

Pleiotropy - where one gene affects multiple traits - is a fundamental challenge in genomics. We approach this as a decryption problem where:

- **Genome = Ciphertext**: DNA sequences contain encrypted information about multiple traits
- **Genes = Polyalphabetic Units**: Each gene can encode multiple "messages" (traits)  
- **Codons = Cipher Symbols**: The 64 codons map to amino acids like cipher substitutions
- **Context = Decryption Key**: Environmental and regulatory context determines trait expression

## 🚀 Key Features

- **High-Performance Rust Core**: Parallel processing of genomic sequences
- **Cryptanalytic Algorithms**: Frequency analysis, pattern detection, context-aware decryption
- **Statistical Analysis**: Chi-squared tests, mutual information, PCA
- **Interactive Visualizations**: Heatmaps, networks, Sankey diagrams
- **E. coli Model System**: Validated against known pleiotropic genes

## 📁 Project Structure

```
pleiotropy/
├── genome_research/         # Research findings and data
│   ├── pleiotropy_overview.md
│   ├── ecoli_pleiotropic_genes.json
│   └── crypto_parallels.md
├── crypto_framework/        # Cryptanalysis algorithm design
│   └── algorithm_design.md
├── rust_impl/              # High-performance Rust implementation
│   ├── src/
│   │   ├── lib.rs         # Main API
│   │   ├── main.rs        # CLI interface
│   │   ├── types.rs       # Data structures
│   │   ├── sequence_parser.rs
│   │   ├── frequency_analyzer.rs
│   │   ├── crypto_engine.rs
│   │   └── trait_extractor.rs
│   └── Cargo.toml
├── python_analysis/        # Python visualization and analysis
│   ├── trait_visualizer.py
│   ├── statistical_analyzer.py
│   ├── rust_interface.py
│   ├── analysis_notebook.ipynb
│   └── requirements.txt
├── examples/               # Example workflows
│   └── ecoli_workflow.sh
└── README.md
```

## 🛠️ Installation

### Prerequisites
- Rust 1.70+ 
- Python 3.8+
- Git

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/murr2k/pleiotropy.git
cd pleiotropy

# Build Rust components
cd rust_impl
cargo build --release

# Install Python dependencies
cd ../python_analysis
pip install -r requirements.txt
```

## 📊 Usage

### Command Line Interface

```bash
# Analyze a genome file
./rust_impl/target/release/genomic_cryptanalysis \
    --input genome.fasta \
    --traits known_traits.json \
    --output results/ \
    --min-traits 2

# Run example E. coli workflow
./examples/ecoli_workflow.sh
```

### Python Analysis

```python
from trait_visualizer import TraitVisualizer
from statistical_analyzer import StatisticalAnalyzer

# Load results
viz = TraitVisualizer()
data = viz.load_trait_data("results/analysis_results.json")

# Create visualizations
viz.plot_trait_correlation_heatmap(data)
viz.create_trait_network(data)
```

### Jupyter Notebook

Open `python_analysis/analysis_notebook.ipynb` for an interactive analysis workflow.

## 🔬 Algorithm Details

### 1. Frequency Analysis
- Global codon usage patterns
- Trait-specific codon bias detection
- Synonymous codon preference analysis

### 2. Cryptographic Pattern Recognition
- Sliding window analysis (300bp windows)
- Eigenanalysis for trait pattern detection
- Regulatory motif identification

### 3. Context-Aware Decryption
- Promoter strength assessment
- Enhancer/silencer mapping
- Expression condition inference

### 4. Trait Separation
- Overlapping region deconvolution
- Confidence scoring based on multiple factors
- Pleiotropic pattern identification

## 📈 Example Results

Using E. coli K-12 as a model:
- Identified key pleiotropic genes (crp, fis, rpoS, hns)
- Detected trait-specific codon usage patterns
- Mapped regulatory contexts to trait expression
- Achieved >70% confidence in trait predictions

## 🧪 Validation

The system has been validated against:
- Known E. coli pleiotropic genes
- Published trait-gene associations
- Codon usage databases

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📚 Citation

If you use this software in your research, please cite:
```
Genomic Pleiotropy Cryptanalysis
Murray Kopit (2025)
https://github.com/murr2k/pleiotropy
```

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

## 🔮 Future Work

- Machine learning integration for pattern recognition
- Extension to other model organisms (yeast, C. elegans)
- Real-time streaming analysis
- GPU acceleration for large genomes

## 📧 Contact

For questions or collaborations:
- Open an issue on GitHub
- Email: murr2k@gmail.com

---

*Developed with ❤️ for the genomics community*