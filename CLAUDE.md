# Claude AI Assistant Instructions

This document provides context and guidelines for AI assistants working on the Genomic Pleiotropy Cryptanalysis project.

## Project Overview

This project treats genomic pleiotropy (single genes affecting multiple traits) as a cryptanalysis problem. The core insight is that genomic sequences can be viewed as encrypted messages where:
- DNA sequences are ciphertext
- Genes are polyalphabetic cipher units
- Codons are cipher symbols
- Regulatory context acts as decryption keys

## Key Technical Components

### Rust Implementation (`rust_impl/`)
- **Performance Critical**: Use Rayon for parallelization
- **Memory Efficient**: Process large genomes in sliding windows
- **Type Safe**: Leverage Rust's type system for genomic data structures

### Python Analysis (`python_analysis/`)
- **Visualization Focus**: Interactive plots using Plotly, static with Matplotlib
- **Statistical Rigor**: Always include p-values and multiple testing correction
- **Rust Integration**: Use PyO3 bindings or subprocess communication

## Development Guidelines

### When Adding Features
1. **Maintain Separation**: Keep cryptanalysis algorithms in Rust, visualization in Python
2. **Document Algorithms**: Add mathematical details to `crypto_framework/`
3. **Test with E. coli**: Use K-12 strain as primary test organism
4. **Preserve Performance**: Profile any changes to core Rust components

### Code Style
- **Rust**: Follow standard Rust conventions, use `cargo fmt` and `cargo clippy`
- **Python**: Use Black formatter, type hints, docstrings for all public functions
- **Comments**: Focus on "why" not "what", explain cryptographic parallels

### Testing
```bash
# Run Rust tests
cd rust_impl && cargo test

# Run Python tests  
cd python_analysis && pytest

# Run integration test
./examples/ecoli_workflow.sh
```

## Common Tasks

### Adding a New Cryptanalysis Method
1. Design algorithm in `crypto_framework/algorithm_design.md`
2. Implement in `rust_impl/src/crypto_engine.rs`
3. Add trait extraction logic to `trait_extractor.rs`
4. Update Python visualization if needed

### Analyzing a New Organism
1. Add organism data to `genome_research/`
2. Create trait definitions JSON
3. Update example workflow
4. Validate against known pleiotropic genes

### Improving Performance
- Profile with `cargo flamegraph`
- Consider SIMD for codon counting
- Use `ndarray` for matrix operations
- Cache frequency tables

## Important Concepts

### Codon Usage Bias
- Different traits show distinct codon preferences
- Synonymous codons carry information
- Calculate chi-squared significance

### Regulatory Context
- Promoter strength affects trait expression
- Enhancers/silencers modify decryption
- Environmental conditions are part of the key

### Trait Separation
- Use eigenanalysis to separate overlapping signals
- Confidence scores based on multiple factors
- Validate against known gene-trait associations

## Debugging Tips

1. **Sequence Parsing Issues**: Check for non-standard characters in FASTA
2. **Low Confidence Scores**: Verify frequency table calculations
3. **Missing Traits**: Check regulatory context detection
4. **Performance Problems**: Profile window size and overlap settings

## Future Enhancements

- Machine learning for pattern recognition
- GPU acceleration for large-scale analysis  
- Real-time streaming genome analysis
- Extension to eukaryotic genomes

## External Resources

- [Codon Usage Database](http://www.kazusa.or.jp/codon/)
- [E. coli K-12 Reference](https://www.ncbi.nlm.nih.gov/nuccore/NC_000913.3)
- [Pleiotropy Reviews](https://pubmed.ncbi.nlm.nih.gov/?term=pleiotropy+review)

## Contact

For algorithmic questions or cryptanalysis insights, refer to:
- `crypto_framework/algorithm_design.md`
- Research papers in `genome_research/references/` (when added)

Remember: We're decrypting nature's multi-trait encoding system!