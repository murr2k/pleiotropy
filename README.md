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
├── trial_database/         # Trial tracking system (NEW)
│   ├── database/          # SQLite database and migrations
│   ├── api/               # FastAPI backend
│   ├── ui/                # React dashboard
│   └── swarm/             # Swarm agent coordination
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

## 🧪 Testing & Validation

### Automated Testing
- **Unit Tests**: >80% code coverage across all components
- **Integration Tests**: Full system workflow validation
- **Performance Tests**: Benchmarked for 1000+ concurrent trials
- **CI/CD Pipeline**: GitHub Actions for continuous testing

### Validation Results
- Known E. coli pleiotropic genes detected with >70% confidence
- Published trait-gene associations confirmed
- Codon usage patterns match established databases
- Swarm coordination tested with 5+ concurrent agents

### Quality Assurance
```bash
# Run all tests with coverage
pytest --cov=python_analysis --cov=trial_database

# Run performance benchmarks
pytest tests/performance --benchmark-only

# Check code quality
cd rust_impl && cargo clippy
python -m black python_analysis/ trial_database/
```

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

## 🗄️ Trial Database System

The project includes a comprehensive trial tracking system for managing cryptanalysis experiments:

### Components
1. **SQLite Database**: Stores trial proposals, test results, and metadata
2. **FastAPI Backend**: RESTful API for database operations
3. **React Dashboard**: Real-time UI showing swarm progress
4. **Swarm Coordination**: Agent communication and task distribution

### Features
- Track proposed trials with parameters and hypotheses
- Store test results with confidence scores and visualizations
- Real-time progress monitoring with WebSocket updates
- Tabular and graphical reporting capabilities
- Agent memory system for knowledge sharing

### Architecture

#### Database Layer (SQLite)
- **Trials Table**: Stores experiment configurations and hypotheses
- **Results Table**: Contains analysis outcomes with confidence scores
- **Agents Table**: Tracks AI agent status and workload
- **Progress Table**: Real-time updates on analysis progress

#### API Layer (FastAPI)
- RESTful endpoints for all CRUD operations
- WebSocket support for live progress updates
- JWT-based authentication for agents
- Batch operations for efficient data handling
- OpenAPI documentation at `/docs`

#### UI Layer (React + TypeScript)
- Real-time dashboard with WebSocket integration
- Interactive charts using Chart.js
- Tabular views with filtering and sorting
- Data export in CSV and JSON formats
- Responsive Material-UI design

#### Swarm Coordination
- Redis-based agent communication
- Automatic task distribution and failover
- Shared memory system for knowledge transfer
- Performance-based agent selection

### Quick Start

```bash
# Clone the repository
git clone https://github.com/murr2k/pleiotropy.git
cd pleiotropy

# Start with Docker (recommended)
./start_system.sh --docker -d

# Or start locally
./start_system.sh --local

# Access the services
# - Dashboard: http://localhost:5173
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Monitoring: http://localhost:3001 (Grafana)
```

### Development Setup

```bash
# Database setup
cd trial_database/database
pip install -r requirements.txt
python init_db.py

# API setup
cd ../api
pip install -r requirements.txt
uvicorn app.main:app --reload

# UI setup
cd ../ui
npm install
npm run dev

# Run tests
pytest tests/ --cov
cd rust_impl && cargo test
cd trial_database/ui && npm test
```

## 🔮 Future Work

- Machine learning integration for pattern recognition
- Extension to other model organisms (yeast, C. elegans)
- Real-time streaming analysis
- GPU acceleration for large genomes
- Expanded trial database with ML experiment tracking

## 📧 Contact

For questions or collaborations:
- Open an issue on GitHub
- Email: murr2k@gmail.com

---

*Developed with ❤️ for the genomics community*