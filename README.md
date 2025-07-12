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

## 🛠️ Installation & Deployment

### Prerequisites
- Docker 20.10+ and Docker Compose 1.29+
- Git
- Optional (for local development): Rust 1.70+, Python 3.8+, Node.js 16+

### 🚀 Quick Deployment (Recommended)

**Using Docker Swarm (Production Ready)**
```bash
# Clone the repository
git clone https://github.com/murr2k/pleiotropy.git
cd pleiotropy

# Start the complete system
./start_system.sh --docker -d

# Verify deployment
./start_system.sh --status
```

**Access Points After Deployment:**
- **Swarm Coordinator API**: http://localhost:8080
- **Dashboard UI**: http://localhost:3000
- **Monitoring (Grafana)**: http://localhost:3001 (admin/admin)
- **Metrics (Prometheus)**: http://localhost:9090
- **Redis**: localhost:6379

### 🔧 Local Development Setup

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

# Install trial database dependencies
cd ../trial_database/api
pip install -r requirements.txt

# Install UI dependencies
cd ../ui
npm install

# Start local services
cd ../..
./start_system.sh --local
```

## 🐳 Docker Deployment

### System Architecture

The system deploys as a microservices architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web UI        │    │  Coordinator    │    │  Redis Cache    │
│  (React/TS)     │    │  (Python/API)   │    │  (Shared Mem)   │
│  Port: 3000     │◄──►│  Port: 8080     │◄──►│  Port: 6379     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │ Rust Analyzer   │    │ Python Visualizer│    │   Monitoring    │
    │   Agent         │    │     Agent       │    │ (Grafana+Prom)  │
    │  (Background)   │    │  (Background)   │    │ Port: 3001/9090  │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Service Management

```bash
# Start all services
./start_system.sh --docker -d

# Check service status
./start_system.sh --status

# View logs
./start_system.sh --logs

# Stop all services
./start_system.sh --stop

# Restart a specific service
docker-compose restart coordinator

# Scale agents (if needed)
docker-compose up -d --scale rust_analyzer=2
```

### Health Monitoring

**Automated Health Checks:**
- Redis: Ping every 5 seconds
- Coordinator: HTTP health endpoint every 10 seconds
- Agents: Heartbeat via Redis every 30 seconds

**Manual Health Verification:**
```bash
# Check all containers
docker ps

# Test Redis
docker exec pleiotropy-redis redis-cli ping

# Test Coordinator API
curl http://localhost:8080/health

# Check agent status
curl http://localhost:8080/api/agents/status
```

### Data Persistence

**Volumes:**
- `redis_data`: Redis persistence
- `prometheus_data`: Metrics storage
- `grafana_data`: Dashboard configurations
- `./reports`: Analysis outputs (host-mounted)

**Backup Strategy:**
```bash
# Backup all data
docker run --rm -v pleiotropy_redis_data:/data -v $(pwd):/backup alpine tar czf /backup/redis-backup.tar.gz /data
docker run --rm -v pleiotropy_prometheus_data:/data -v $(pwd):/backup alpine tar czf /backup/prometheus-backup.tar.gz /data

# Restore data
docker run --rm -v pleiotropy_redis_data:/data -v $(pwd):/backup alpine tar xzf /backup/redis-backup.tar.gz -C /
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

### Production Deployment Checklist

**Pre-Deployment:**
- [ ] Docker and Docker Compose installed
- [ ] Firewall configured (ports 3000, 8080, 3001, 9090)
- [ ] SSL certificates ready (for production)
- [ ] Backup strategy in place

**Deployment Steps:**
```bash
# 1. Clone and deploy
git clone https://github.com/murr2k/pleiotropy.git
cd pleiotropy
./start_system.sh --docker -d

# 2. Verify services
./start_system.sh --status

# 3. Run system tests
python trial_database/tests/test_integration.py

# 4. Check monitoring
curl http://localhost:3001  # Grafana
curl http://localhost:9090  # Prometheus
```

**Post-Deployment:**
- [ ] All services healthy
- [ ] Monitoring dashboards accessible
- [ ] Test analysis workflow
- [ ] Configure log rotation
- [ ] Set up alerting (optional)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/murr2k/pleiotropy.git
cd pleiotropy

# Start with Docker (recommended)
./start_system.sh --docker -d

# Access the services
# - Dashboard: http://localhost:3000
# - Coordinator API: http://localhost:8080
# - API Documentation: http://localhost:8080/docs
# - Monitoring: http://localhost:3001 (admin/admin)
# - Metrics: http://localhost:9090
```

### Development Setup

```bash
# Complete development environment
# Database setup
cd trial_database/database
pip install -r requirements.txt
python init_db.py

# API setup
cd ../api
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# UI setup
cd ../ui
npm install
npm run dev

# Swarm setup
cd ../swarm
pip install -r requirements.txt
python coordinator.py

# Run comprehensive tests
pytest tests/ --cov --cov-report=html
cd ../../rust_impl && cargo test
cd ../trial_database/ui && npm test
cd .. && python -m pytest swarm/tests/
```

### Environment Configuration

**Production Environment Variables:**
```bash
# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=  # Set in production

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
CORS_ORIGINS=*  # Restrict in production

# Database Configuration
DATABASE_URL=sqlite:///./trial_database.db

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=admin  # Change in production

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

**Development Environment:**
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
vim .env

# Load environment
source .env
```

## 🔧 Troubleshooting

### Common Issues

**Services Won't Start:**
```bash
# Check Docker status
docker ps -a
docker-compose logs

# Clean restart
./start_system.sh --stop
docker system prune -f
./start_system.sh --docker -d
```

**Redis Connection Errors:**
```bash
# Check Redis connectivity
docker exec pleiotropy-redis redis-cli ping

# Restart Redis
docker-compose restart redis
```

**Agent Communication Issues:**
```bash
# Check agent status
curl http://localhost:8080/api/agents/status

# Restart agents
docker-compose restart rust_analyzer python_visualizer
```

**Performance Issues:**
```bash
# Monitor resource usage
docker stats

# Check logs for bottlenecks
docker-compose logs --tail=100 coordinator

# Scale agents if needed
docker-compose up -d --scale rust_analyzer=2
```

### Log Analysis

```bash
# View all logs
./start_system.sh --logs

# Filter specific service
docker-compose logs coordinator

# Follow logs in real-time
docker-compose logs -f rust_analyzer

# Export logs for analysis
docker-compose logs > system_logs.txt
```

### Performance Tuning

**Memory Optimization:**
```bash
# Adjust container memory limits
# Edit docker-compose.yml:
services:
  coordinator:
    mem_limit: 512m
    memswap_limit: 1g
```

**CPU Optimization:**
```bash
# Set CPU limits
services:
  rust_analyzer:
    cpus: '2.0'
    cpu_shares: 1024
```

## 🔮 Future Work

- Machine learning integration for pattern recognition
- Extension to other model organisms (yeast, C. elegans)
- Real-time streaming analysis with Kafka
- GPU acceleration for large genomes (CUDA support)
- Kubernetes deployment for cloud scalability
- Advanced monitoring with custom metrics
- Expanded trial database with ML experiment tracking
- Multi-tenant support for shared environments

## 📧 Contact & Support

**For Questions or Collaborations:**
- Open an issue on GitHub
- Email: murr2k@gmail.com

**System Administration:**
- Monitor system health at http://localhost:3001
- Check API status at http://localhost:8080/health
- Review logs with `./start_system.sh --logs`

**Emergency Procedures:**
```bash
# System emergency restart
./start_system.sh --stop
docker system prune -f
./start_system.sh --docker -d

# Data recovery
# See backup/restore procedures in Docker Deployment section
```

## 📈 Monitoring & Maintenance

### Grafana Dashboards

**System Overview Dashboard:**
- Agent health and workload distribution
- Task completion rates and success metrics
- System resource utilization
- Error rates and alert thresholds

**Analysis Dashboard:**
- Trial success rates by organism
- Confidence score distributions
- Processing time metrics
- Data quality indicators

**Access Grafana:**
1. Navigate to http://localhost:3001
2. Login: admin/admin
3. Navigate to "Swarm Dashboard"

### Prometheus Metrics

**Key Metrics Collected:**
```
# Agent metrics
agent_heartbeat_last_seen
agent_task_completion_rate
agent_error_count

# System metrics
redis_connection_count
api_request_duration
trial_processing_time

# Analysis metrics
confidence_score_distribution
trait_detection_accuracy
```

### Maintenance Tasks

**Daily:**
- Check service health status
- Review error logs
- Monitor disk space usage

**Weekly:**
- Backup database and configurations
- Update system metrics baseline
- Review performance trends

**Monthly:**
- Update Docker images
- Archive old trial data
- Performance optimization review

### Alerting Setup (Optional)

```bash
# Install alertmanager
docker run -d --name alertmanager \
  -p 9093:9093 \
  prom/alertmanager

# Configure alerts
vim monitoring/alerts.yml
```

**Sample Alert Rules:**
```yaml
groups:
- name: pleiotropy
  rules:
  - alert: AgentDown
    expr: agent_heartbeat_last_seen > 300
    labels:
      severity: critical
    annotations:
      summary: "Agent {{ $labels.agent }} is down"
      
  - alert: HighErrorRate
    expr: rate(agent_error_count[5m]) > 0.1
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
```

---

*Developed with ❤️ for the genomics community*

**System Status**: ✅ Production Ready | 🐳 Docker Deployed | 📊 Monitored | 🤖 Swarm Enabled