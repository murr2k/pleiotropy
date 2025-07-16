# Docker GPU Setup for CUDA Factorization

This document provides instructions for setting up GPU acceleration in Docker for the CUDA composite number factorizer and semiprime seeker.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with compute capability 3.5 or higher
- Recommended: GTX 1060 or better (RTX series preferred)
- Minimum 4GB GPU memory
- Host system with 8GB+ RAM

### Software Requirements
- Docker 19.03+ with GPU support
- NVIDIA GPU drivers (450.80.02+)
- NVIDIA Container Toolkit (nvidia-docker2)

## Installation

### 1. Install NVIDIA GPU Drivers

**Ubuntu/Debian:**
```bash
# Check current driver
nvidia-smi

# Install latest drivers if needed
sudo apt update
sudo apt install nvidia-driver-470
sudo reboot
```

**CentOS/RHEL:**
```bash
# Install EPEL repository
sudo yum install epel-release

# Install NVIDIA drivers
sudo yum install nvidia-driver nvidia-settings
sudo reboot
```

### 2. Install NVIDIA Container Toolkit

**Ubuntu/Debian:**
```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt update
sudo apt install nvidia-docker2

# Restart Docker
sudo systemctl restart docker
```

**CentOS/RHEL:**
```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo

# Install nvidia-docker2
sudo yum install nvidia-docker2

# Restart Docker
sudo systemctl restart docker
```

### 3. Verify GPU Support

```bash
# Test NVIDIA Docker integration
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Expected output: GPU information and driver version
```

## Usage

### Starting with GPU Support

```bash
# Clone the repository
git clone https://github.com/murr2k/pleiotropy.git
cd pleiotropy

# Start with GPU acceleration
./start_system.sh --gpu -d

# Verify GPU services are running
./start_system.sh --status
```

### Available Services with GPU

When started with `--gpu`, the following services get GPU access:

1. **rust_analyzer**: Enhanced with CUDA support for genomic analysis
2. **cuda_factorizer**: Dedicated GPU factorization service

### GPU Service Configuration

The GPU configuration includes:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=compute,utility
runtime: nvidia
```

## Performance Validation

### Test GPU Acceleration

```bash
# Start the GPU-enabled system
./start_system.sh --gpu -d

# Test CUDA factorization via API
curl -X POST http://localhost:8080/api/factorize \
  -H "Content-Type: application/json" \
  -d '{"number": 2133019384970323}'

# Expected response includes GPU usage confirmation
```

### Monitor GPU Usage

```bash
# Monitor GPU utilization during factorization
watch nvidia-smi

# Check container GPU access
docker exec pleiotropy-rust-analyzer nvidia-smi
docker exec pleiotropy-cuda-factorizer nvidia-smi
```

## Benchmarking

### Expected Performance Improvements

On GTX 2070 (reference hardware):

| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| 16-digit semiprime | 2.4s | 0.135s | 18x |
| 31-digit semiprime | 6.8 min | 18.9s | 22x |
| Codon counting | 1.2s | 0.03s | 40x |
| Pattern matching | 0.8s | 0.016s | 50x |

### Running Benchmarks

```bash
# Start GPU system
./start_system.sh --gpu -d

# Run factorization benchmark
docker exec pleiotropy-cuda-factorizer python benchmark_factorization.py

# Run genomic analysis benchmark  
docker exec pleiotropy-rust-analyzer cargo test --features cuda benchmark_tests -- --ignored
```

## Troubleshooting

### Common Issues

**1. "nvidia-docker not found"**
```bash
# Check installation
dpkg -l | grep nvidia-docker
sudo apt reinstall nvidia-docker2
sudo systemctl restart docker
```

**2. "CUDA driver version is insufficient"**
```bash
# Check driver version
nvidia-smi
# Update drivers if version < 450.80.02
sudo apt update && sudo apt upgrade nvidia-driver-*
```

**3. "No CUDA-capable device found"**
```bash
# Check GPU visibility
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# Ensure GPU is not being used by other processes
sudo fuser -v /dev/nvidia*
```

**4. "Out of memory" errors**
```bash
# Check GPU memory
nvidia-smi

# Reduce batch size or restart services
./start_system.sh --stop
./start_system.sh --gpu -d
```

### Debug Commands

```bash
# Check NVIDIA Docker configuration
docker info | grep nvidia

# Test basic GPU access
docker run --rm --gpus all ubuntu nvidia-smi

# Check container GPU allocation
docker inspect pleiotropy-cuda-factorizer | grep -i gpu

# View detailed logs
docker logs pleiotropy-cuda-factorizer
docker logs pleiotropy-rust-analyzer
```

### Performance Tuning

**For GTX 1060/1070:**
```bash
# Reduce thread count for older GPUs
export CUDA_THREAD_COUNT=512
./start_system.sh --gpu -d
```

**For RTX 3080/4090:**
```bash
# Increase batch size for newer GPUs
export CUDA_BATCH_SIZE=2048
./start_system.sh --gpu -d
```

## Alternative: CPU Fallback

If GPU setup fails, the system automatically falls back to CPU:

```bash
# Start without GPU
./start_system.sh --docker -d

# All functionality available, just slower
# CPU-only factorization still works
```

## Docker Compose Details

### GPU Override File

The GPU configuration is in `docker-compose.gpu.yml`:

```yaml
services:
  rust_analyzer:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  cuda_factorizer:
    # Dedicated GPU factorization service
    runtime: nvidia
    # ... additional GPU configuration
```

### Manual Docker Compose Usage

```bash
# Start with GPU support manually
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Stop GPU services
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml down
```

## Security Considerations

### GPU Resource Isolation

- Each container gets access to specific GPU devices
- Memory isolation between containers
- No privileged access required

### Network Security

```bash
# Restrict GPU service access
# Edit docker-compose.gpu.yml to limit port exposure
ports:
  - "127.0.0.1:8080:8080"  # Localhost only
```

## Monitoring and Maintenance

### GPU Health Monitoring

```bash
# Add to Grafana dashboard
# GPU utilization metrics
nvidia_gpu_utilization_percent
nvidia_gpu_memory_usage_bytes
nvidia_gpu_temperature_celsius

# Container GPU metrics
container_gpu_utilization
container_gpu_memory_usage
```

### Automatic GPU Recovery

The system includes automatic GPU recovery:

1. Detects GPU failures
2. Restarts affected containers
3. Falls back to CPU if needed
4. Alerts operators of issues

## Production Deployment

### Scaling Considerations

```bash
# Scale GPU services based on load
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --scale cuda_factorizer=2

# Multi-GPU setup (if available)
export CUDA_VISIBLE_DEVICES=0,1
./start_system.sh --gpu -d
```

### Resource Management

```yaml
# Production resource limits
services:
  cuda_factorizer:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Support

For GPU-specific issues:

1. Check NVIDIA documentation: https://docs.nvidia.com/datacenter/cloud-native/
2. Docker GPU guide: https://docs.docker.com/config/containers/resource_constraints/#gpu
3. File issue with system logs and `nvidia-smi` output

## References

- [NVIDIA Container Toolkit Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker GPU Support Documentation](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [CUDA Docker Images](https://hub.docker.com/r/nvidia/cuda)
- [cudarc Rust Library](https://docs.rs/cudarc/latest/cudarc/)