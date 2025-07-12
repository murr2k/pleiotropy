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

## Trial Database System

### Overview
A comprehensive system for tracking cryptanalysis trials and test results, enabling:
- Collaborative experiment management
- Real-time progress monitoring
- Knowledge sharing between swarm agents
- Historical analysis of successful approaches

### Architecture
1. **Database Layer** (SQLite)
   - `trials` table: experiment proposals with parameters
   - `results` table: test outcomes with metrics
   - `agents` table: swarm member tracking
   - `progress` table: real-time status updates

2. **API Layer** (FastAPI)
   - RESTful endpoints for CRUD operations
   - WebSocket support for live updates
   - Authentication for swarm agents
   - Batch operations for efficiency

3. **UI Layer** (React + TypeScript)
   - Dashboard with real-time progress
   - Tabular views with filtering/sorting
   - Interactive charts (Chart.js)
   - Agent activity monitoring

4. **Swarm Integration**
   - Shared memory system for coordination
   - Agent task assignment and tracking
   - Result aggregation and validation
   - Automatic report generation

### Development Guidelines
- Use TypeScript for type safety in UI
- Implement proper error handling and logging
- Write tests for critical paths
- Document API endpoints with OpenAPI
- Use Docker for consistent deployment

## Swarm Implementation

### Agent Types
1. **Database Architect**: Designs schema and manages migrations
2. **API Developer**: Creates FastAPI endpoints and WebSocket handlers
3. **UI Engineer**: Builds React components and real-time dashboards
4. **Integration Specialist**: Coordinates swarm and existing code
5. **QA Engineer**: Ensures quality through comprehensive testing

### Agent Communication
- **Memory Namespace**: `swarm-auto-centralized-[timestamp]`
- **Redis Pub/Sub**: Real-time task distribution
- **Heartbeat System**: 30-second intervals for health monitoring
- **Task Queue**: Priority-based assignment with failover

### Task Coordination
```python
# Example: Agent saves results to memory
Memory.store("swarm-auto-centralized-XXX/agent-name/task", {
    "results": analysis_output,
    "confidence": 0.85,
    "timestamp": datetime.now()
})

# Coordinator retrieves and aggregates
results = Memory.query("swarm-auto-centralized-XXX")
```

### Best Practices
1. **Batch Operations**: Always use MultiEdit and batch tools
2. **Memory Usage**: Save progress after each significant step
3. **Error Handling**: Implement retry logic with exponential backoff
4. **Performance**: Profile memory usage and optimize queries
5. **Monitoring**: Use Prometheus metrics for agent tracking

## Future Enhancements

- Machine learning for pattern recognition
- GPU acceleration for large-scale analysis  
- Real-time streaming genome analysis
- Extension to eukaryotic genomes
- Trial database ML integration for experiment optimization

## External Resources

- [Codon Usage Database](http://www.kazusa.or.jp/codon/)
- [E. coli K-12 Reference](https://www.ncbi.nlm.nih.gov/nuccore/NC_000913.3)
- [Pleiotropy Reviews](https://pubmed.ncbi.nlm.nih.gov/?term=pleiotropy+review)

## Contact

For algorithmic questions or cryptanalysis insights, refer to:
- `crypto_framework/algorithm_design.md`
- Research papers in `genome_research/references/` (when added)

### Swarm Debugging

1. **Agent Not Responding**: Check Redis connection and heartbeat logs
2. **Task Stuck**: Verify memory keys and task queue status
3. **Performance Issues**: Review agent workload distribution
4. **Integration Failures**: Check backward compatibility layer

### Monitoring Dashboard

Access Grafana at `http://localhost:3001` for:
- Agent status and workload
- Task completion rates
- System performance metrics
- Error rates and alerts

Remember: We're decrypting nature's multi-trait encoding system with the power of distributed AI agents!