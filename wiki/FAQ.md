# Frequently Asked Questions

## General Questions

### What is genomic pleiotropy?

Pleiotropy is a phenomenon where a single gene affects multiple, seemingly unrelated traits. For example, a gene might influence both metabolic rate and stress response. This project treats the detection of these multi-trait relationships as a cryptanalysis problem.

### Why use cryptanalysis for genomics?

Genomic sequences can be viewed as encrypted messages where:
- The genome contains multiple overlapping "messages" (traits)
- Codon usage patterns act as cipher patterns
- Regulatory elements serve as decryption keys

This approach has led to novel insights and improved detection of pleiotropic genes.

### What organisms are supported?

Currently supported:
- **E. coli** (all strains, optimized for K-12)
- **Synthetic sequences** for testing

In development:
- S. cerevisiae (yeast)
- C. elegans (worm)
- Human genome (2025)

### How accurate is the analysis?

- **Synthetic data**: 100% detection rate
- **E. coli**: ~87% accuracy for known pleiotropic genes
- **False positive rate**: <5%
- **Confidence threshold**: Adjustable (default 0.4)

## Technical Questions

### What are the system requirements?

**Minimum:**
- 4 CPU cores
- 8 GB RAM
- 20 GB storage
- Docker 20.10+

**Recommended:**
- 8+ CPU cores
- 16+ GB RAM
- 50+ GB SSD
- NVIDIA GPU (optional)

### How long does analysis take?

- **E. coli genome**: ~7 seconds
- **Synthetic test data**: <1 second
- **Human chromosome**: <5 seconds (planned)

Performance depends on:
- Genome size
- Number of traits
- System resources
- GPU availability

### Can I use custom trait definitions?

Yes! Create a JSON file with your trait definitions:

```json
{
  "traits": [
    {
      "name": "my_trait",
      "description": "Custom trait description",
      "associated_genes": ["gene1", "gene2"],
      "codon_patterns": ["CTG", "GAA"]
    }
  ]
}
```

### How do I interpret confidence scores?

Confidence scores (0-1) indicate the likelihood that a gene is truly pleiotropic:
- **0.8-1.0**: Very high confidence
- **0.6-0.8**: High confidence
- **0.4-0.6**: Moderate confidence
- **<0.4**: Low confidence (filtered by default)

Scores are based on:
- Number of traits detected
- Codon usage patterns
- Statistical significance
- Regulatory context

## Usage Questions

### How do I run analysis via command line?

```bash
# Using Docker
docker exec pleiotropy-rust-analyzer genomic_cryptanalysis \
  --input genome.fasta \
  --traits traits.json \
  --output results/

# Using local installation
./rust_impl/target/release/genomic_cryptanalysis \
  --input genome.fasta \
  --traits traits.json \
  --min-traits 2 \
  --min-confidence 0.4
```

### Can I analyze multiple genomes?

Yes, use batch processing:

```bash
# Via API
curl -X POST http://localhost:8080/api/batch \
  -H "Authorization: Bearer $TOKEN" \
  -d @batch_request.json

# Via script
for genome in genomes/*.fasta; do
  ./analyze.sh "$genome"
done
```

### How do I export results?

Results can be exported in multiple formats:

```bash
# JSON (default)
curl http://localhost:8080/api/results/$TRIAL_ID?format=json

# CSV
curl http://localhost:8080/api/results/$TRIAL_ID?format=csv

# PDF report
curl http://localhost:8080/api/results/$TRIAL_ID?format=pdf
```

### Can I integrate with my pipeline?

Yes! We provide:
- REST API for programmatic access
- Python client library
- Docker images for integration
- WebSocket for real-time updates

Example:
```python
from pleiotropy import PleiotropyClient

client = PleiotropyClient("http://localhost:8080")
result = client.analyze("genome.fasta", min_confidence=0.5)
```

## Troubleshooting

### Analysis returns 0 genes

Common causes:
1. **Confidence threshold too high**: Try lowering to 0.3
2. **Trait definitions missing**: Ensure traits.json is valid
3. **Sequence quality**: Check for non-standard characters
4. **Old version**: Update to latest (includes NeuroDNA fix)

### Docker containers won't start

1. Check port conflicts:
   ```bash
   sudo lsof -i :3000
   sudo lsof -i :8080
   ```

2. Clean Docker resources:
   ```bash
   docker system prune -f
   docker-compose down -v
   docker-compose up -d
   ```

3. Check logs:
   ```bash
   docker-compose logs coordinator
   ```

### Memory errors during analysis

1. Increase Docker memory limit
2. Process smaller sequences
3. Reduce concurrent analyses
4. Enable swap memory

### API returns 401 Unauthorized

1. Check token expiration
2. Verify token format: `Bearer <token>`
3. Regenerate token if needed
4. Check CORS settings

## Data & Privacy

### Is my genomic data secure?

- All data is processed locally
- No data is sent to external servers
- Files are deleted after analysis (configurable)
- Optional encryption at rest
- Audit logging available

### Can I use this for clinical data?

**No**, this tool is for research only. It is not:
- FDA approved
- HIPAA compliant (yet)
- Validated for clinical use
- A replacement for medical advice

### How do I cite this project?

```bibtex
@software{pleiotropy2025,
  title = {Genomic Pleiotropy Cryptanalysis},
  author = {Kopit, Murray},
  year = {2025},
  url = {https://github.com/murr2k/pleiotropy},
  version = {1.0.0}
}
```

## Development

### How can I contribute?

See our [Contributing Guide](Contributing) for details. We welcome:
- Bug reports and fixes
- Feature implementations
- Documentation improvements
- Test additions
- Performance optimizations

### Is there a development roadmap?

Yes! See our [Roadmap](Roadmap) for planned features and timeline.

### How do I report bugs?

1. Check [existing issues](https://github.com/murr2k/pleiotropy/issues)
2. Create a new issue with:
   - Clear description
   - Steps to reproduce
   - System information
   - Error messages

### Can I use this commercially?

Yes, under the MIT license. You can:
- Use commercially
- Modify freely
- Distribute
- Use privately

Requirements:
- Include license notice
- No warranty provided

## Advanced Topics

### How does NeuroDNA integration work?

NeuroDNA v0.0.2 provides neural network-inspired pattern recognition:
1. Analyzes codon frequency distributions
2. Matches against learned trait patterns
3. Calculates multi-factor confidence scores
4. Falls back to traditional methods if needed

### Can I add custom algorithms?

Yes! Implement the trait detection interface:

```rust
trait TraitDetector {
    fn detect(&self, sequence: &Sequence) -> Vec<Trait>;
}
```

See [Architecture](Architecture) for details.

### How do I optimize performance?

1. **Enable GPU acceleration** (when available)
2. **Increase parallelization**:
   ```toml
   [analysis]
   num_threads = 16
   ```
3. **Use batch processing**
4. **Enable caching**:
   ```yaml
   cache:
     enabled: true
     size: 10GB
   ```

### What databases are integrated?

Currently integrated:
- Local file storage
- Custom trait definitions

Planned integrations:
- NCBI GenBank
- Ensembl
- UniProt
- KEGG

---

*Don't see your question? Ask in [GitHub Discussions](https://github.com/murr2k/pleiotropy/discussions) or check our [documentation](Home).*