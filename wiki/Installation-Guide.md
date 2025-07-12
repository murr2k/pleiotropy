# Installation Guide

## 📋 Prerequisites

### System Requirements

#### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS 12+, or Windows 10+ with WSL2
- **CPU**: 4 cores (x86_64 or ARM64)
- **RAM**: 8 GB
- **Storage**: 20 GB available space
- **Network**: Stable internet connection

#### Recommended Requirements
- **OS**: Ubuntu 22.04 LTS or macOS 13+
- **CPU**: 8+ cores
- **RAM**: 16+ GB
- **Storage**: 50+ GB SSD
- **GPU**: NVIDIA GPU with CUDA 11+ (optional)

### Software Dependencies

#### Required
- **Docker**: 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose**: 1.29+ (included with Docker Desktop)
- **Git**: 2.25+ ([Install Git](https://git-scm.com/downloads))

#### Optional (for development)
- **Rust**: 1.70+ ([Install Rust](https://rustup.rs/))
- **Python**: 3.8+ ([Install Python](https://python.org/))
- **Node.js**: 16+ ([Install Node.js](https://nodejs.org/))

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/murr2k/pleiotropy.git
cd pleiotropy
```

### 2. Start the System
```bash
# Start all services with Docker
./start_system.sh --docker -d

# Verify deployment
./start_system.sh --status
```

### 3. Access the Services
- **Web UI**: http://localhost:3000
- **API**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs
- **Monitoring**: http://localhost:3001 (admin/admin)

### 4. Run Your First Analysis
```bash
# Using the CLI
docker exec pleiotropy-rust-analyzer genomic_cryptanalysis \
  --input /data/test_synthetic.fasta \
  --traits /data/test_traits.json \
  --output /data/results/

# Or via the Web UI at http://localhost:3000
```

## 🐳 Docker Installation (Recommended)

### Step 1: Install Docker

#### Ubuntu/Debian
```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's GPG key
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

#### macOS
```bash
# Install Docker Desktop
brew install --cask docker

# Or download from https://www.docker.com/products/docker-desktop/
```

#### Windows (WSL2)
1. Install WSL2: https://docs.microsoft.com/en-us/windows/wsl/install
2. Install Docker Desktop: https://www.docker.com/products/docker-desktop/
3. Enable WSL2 integration in Docker Desktop settings

### Step 2: Deploy with Docker Compose

```bash
# Clone repository
git clone https://github.com/murr2k/pleiotropy.git
cd pleiotropy

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### Step 3: Verify Installation

```bash
# Check all services are running
curl http://localhost:8080/health

# Expected output:
# {"status": "healthy", "services": {...}}
```

## 💻 Local Development Setup

### Rust Components

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Build Rust components
cd rust_impl
cargo build --release

# Run tests
cargo test

# Run with sample data
cargo run --release -- \
  --input ../test_synthetic.fasta \
  --traits ../test_traits.json \
  --output ../output/
```

### Python Components

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
cd python_analysis
pip install -r requirements.txt

# Run statistical analysis
python statistical_analyzer.py ../output/analysis_results.json

# Launch Jupyter notebook
jupyter notebook analysis_notebook.ipynb
```

### Frontend Development

```bash
# Install Node.js dependencies
cd trial_database/ui
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run tests
npm test
```

### Database Setup

```bash
# Initialize database
cd trial_database/database
python init_db.py

# Run migrations
alembic upgrade head

# Seed with sample data
python seed_data.py
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
CORS_ORIGINS=http://localhost:3000

# Database
DATABASE_URL=sqlite:///./trial_database.db

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=admin

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Analysis Settings
MIN_CONFIDENCE_THRESHOLD=0.4
MAX_CONCURRENT_ANALYSES=10
ANALYSIS_TIMEOUT=300
```

### Docker Compose Configuration

Customize `docker-compose.yml` for your environment:

```yaml
services:
  coordinator:
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8080
    ports:
      - "8080:8080"
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
```

## 🌐 Cloud Deployment

### AWS Deployment

```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS credentials
aws configure

# Deploy to ECS
./deploy/aws/deploy-ecs.sh
```

### Google Cloud Platform

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash

# Initialize gcloud
gcloud init

# Deploy to GKE
./deploy/gcp/deploy-gke.sh
```

### Azure Deployment

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Deploy to AKS
./deploy/azure/deploy-aks.sh
```

## 🧪 Testing the Installation

### Run Integration Tests

```bash
# Run all tests
./run_tests.sh

# Run specific test suite
pytest tests/integration/test_full_pipeline.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### Verify Core Functionality

```bash
# Test Rust analyzer
docker exec pleiotropy-rust-analyzer genomic_cryptanalysis --help

# Test API
curl -X POST http://localhost:8080/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"genome_file": "test.fasta", "traits_file": "traits.json"}'

# Test WebSocket connection
wscat -c ws://localhost:8080/ws/progress
```

## 🔍 Troubleshooting

### Common Issues

#### Docker Issues

**Problem**: Permission denied errors
```bash
# Solution: Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

**Problem**: Port already in use
```bash
# Find process using port
sudo lsof -i :3000

# Change port in docker-compose.yml
ports:
  - "3001:3000"  # Change to available port
```

#### Build Issues

**Problem**: Rust compilation errors
```bash
# Update Rust
rustup update

# Clean and rebuild
cargo clean
cargo build --release
```

**Problem**: Python dependency conflicts
```bash
# Use virtual environment
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Runtime Issues

**Problem**: Out of memory errors
```bash
# Increase Docker memory limit
docker-compose down
# Edit docker-compose.yml to increase memory limits
docker-compose up -d
```

**Problem**: Slow performance
```bash
# Check resource usage
docker stats

# Scale services
docker-compose up -d --scale rust_analyzer=2
```

### Getting Help

1. Check the [FAQ](FAQ)
2. Search [GitHub Issues](https://github.com/murr2k/pleiotropy/issues)
3. Join [Discussions](https://github.com/murr2k/pleiotropy/discussions)
4. Contact support: support@pleiotropy.dev

## 📚 Next Steps

After successful installation:

1. Read the [Architecture](Architecture) documentation
2. Try the [Tutorial](Tutorial) for a hands-on walkthrough
3. Explore the [API Reference](API-Reference)
4. Join the [Community](https://github.com/murr2k/pleiotropy/discussions)

---

*Having issues? Check our [Troubleshooting Guide](Troubleshooting) or [open an issue](https://github.com/murr2k/pleiotropy/issues/new).*