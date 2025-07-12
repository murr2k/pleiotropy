# CUDA Quick Start Guide

Get up and running with GPU acceleration in 5 minutes!

## Prerequisites Checklist

- [ ] NVIDIA GPU (GTX 1060 or newer)
- [ ] Linux (Ubuntu 20.04+) or Windows 11 with WSL2
- [ ] 8GB+ system RAM
- [ ] 20GB free disk space

## Step 1: Check Your GPU

```bash
# Check if you have an NVIDIA GPU
lspci | grep -i nvidia

# If you see output, check the driver
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.154.05   Driver Version: 535.154.05   CUDA Version: 12.2    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| 30%   45C    P8    20W / 215W |    500MiB /  8192MiB |      2%      Default |
+-------------------------------+----------------------+----------------------+
```

## Step 2: Install CUDA (Ubuntu/WSL2)

```bash
# Quick install script
wget https://raw.githubusercontent.com/genomic-pleiotropy/scripts/main/install_cuda.sh
chmod +x install_cuda.sh
./install_cuda.sh

# Or manual install
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-8

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Step 3: Build the Project

```bash
# Clone and build
git clone https://github.com/genomic-pleiotropy/cryptanalysis.git
cd cryptanalysis/rust_impl

# Test CUDA detection
cargo run --features cuda --bin cuda-info

# Build with CUDA support
cargo build --release --features cuda
```

## Step 4: Run Your First GPU-Accelerated Analysis

```bash
# Download example data
wget https://github.com/genomic-pleiotropy/data/raw/main/examples/ecoli_sample.fasta
wget https://github.com/genomic-pleiotropy/data/raw/main/examples/ecoli_traits.json

# Run analysis with automatic GPU acceleration
../target/release/genomic-cryptanalysis analyze \
    ecoli_sample.fasta \
    --traits ecoli_traits.json \
    --output results/

# Check if GPU was used
cat results/performance_stats.json
```

## Step 5: Verify GPU Acceleration

Look for these indicators that GPU acceleration is working:

```json
{
  "performance_stats": {
    "cuda_available": true,
    "cuda_device": "NVIDIA GeForce RTX 2070",
    "cuda_calls": 156,
    "cpu_calls": 12,
    "avg_gpu_time_ms": 4.3,
    "avg_cpu_time_ms": 78.2,
    "speedup_factor": 18.2
  }
}
```

## Simple Python Example

```python
# example.py
import genomic_cryptanalysis as gc

# Check CUDA status
print(f"CUDA available: {gc.cuda_available()}")
if gc.cuda_available():
    print(f"GPU: {gc.cuda_info()}")

# Analyze a genome (GPU acceleration is automatic)
analyzer = gc.GenomicAnalyzer()
results = analyzer.analyze_file(
    "ecoli_sample.fasta",
    traits_file="ecoli_traits.json"
)

# Print results
print(f"\nFound {len(results.pleiotropic_genes)} pleiotropic genes")
print(f"GPU speedup: {results.performance.speedup}x")

# First few results
for gene in results.pleiotropic_genes[:5]:
    print(f"{gene.id}: {', '.join(gene.traits)} (confidence: {gene.confidence:.2f})")
```

## Troubleshooting Quick Fixes

### CUDA Not Found
```bash
# Set CUDA path explicitly
export CUDA_PATH=/usr/local/cuda-11.8
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

### Out of Memory
```bash
# Reduce batch size
genomic-cryptanalysis analyze genome.fasta --batch-size 1000
```

### Slow Performance
```bash
# Check GPU utilization
watch -n 1 nvidia-smi

# Enable profiling
CUDA_PROFILE=1 genomic-cryptanalysis analyze genome.fasta
```

### Force CPU Mode (for testing)
```bash
# Disable CUDA temporarily
export GENOMIC_CRYPTO_DISABLE_CUDA=1
genomic-cryptanalysis analyze genome.fasta

# Or use command line flag
genomic-cryptanalysis analyze genome.fasta --force-cpu
```

## Next Steps

1. **Read the full guide**: [CUDA Acceleration Guide](CUDA_ACCELERATION_GUIDE.md)
2. **Try larger genomes**: Download from [NCBI](https://www.ncbi.nlm.nih.gov/genome/)
3. **Tune performance**: See [Performance Tuning](CUDA_ACCELERATION_GUIDE.md#performance-tuning)
4. **Join the community**: [Discord](https://discord.gg/genomic-crypto)

## Quick Performance Test

Compare CPU vs GPU performance:

```bash
# Run benchmark
genomic-cryptanalysis benchmark --compare

# Example output:
# Dataset: E. coli K-12 (4.6 Mbp)
# CPU (16 cores): 12.4 seconds
# GPU (RTX 2070): 0.73 seconds
# Speedup: 17.0x
```

## Common Use Cases

### Batch Processing Multiple Genomes
```bash
# Process all FASTA files in a directory
for file in genomes/*.fasta; do
    genomic-cryptanalysis analyze "$file" \
        --traits universal_traits.json \
        --output "results/$(basename $file .fasta)/"
done
```

### Large Genome Analysis
```bash
# For genomes > 100 Mbp, use streaming mode
genomic-cryptanalysis analyze large_genome.fasta \
    --streaming \
    --window-size 10000000 \
    --traits traits.json
```

### Real-time Monitoring
```bash
# Monitor GPU usage during analysis
genomic-cryptanalysis analyze genome.fasta --gpu-monitor &

# In another terminal
tail -f cuda_metrics.log
```

## Getting Help

- **Quick check**: `genomic-cryptanalysis --help`
- **CUDA info**: `genomic-cryptanalysis cuda-info --detailed`
- **GitHub Issues**: [Report problems](https://github.com/genomic-pleiotropy/cryptanalysis/issues)
- **Community Chat**: [Discord server](https://discord.gg/genomic-crypto)

---

**Remember**: GPU acceleration is automatic! Just build with `--features cuda` and the system will use your GPU whenever possible, falling back to CPU when needed.