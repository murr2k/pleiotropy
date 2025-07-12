# User Guide: Accessing Pleiotropy Analysis Services

This guide provides step-by-step instructions for accessing and using all components of the Genomic Pleiotropy Cryptanalysis system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Service Overview](#service-overview)
3. [Web Dashboard](#web-dashboard)
4. [API Interface](#api-interface)
5. [Monitoring Interface](#monitoring-interface)
6. [Command Line Tools](#command-line-tools)
7. [Analysis Workflows](#analysis-workflows)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Starting the System

```bash
# Clone the repository
git clone https://github.com/murr2k/pleiotropy.git
cd pleiotropy

# Start all services
./start_system.sh --docker -d

# Verify system is ready
./start_system.sh --status
```

### Accessing Services

Once the system is running, access points are:

| Service | URL | Purpose |
|---------|-----|---------|
| **Web Dashboard** | http://localhost:3000 | Main user interface |
| **API Interface** | http://localhost:8080 | REST API and documentation |
| **Monitoring** | http://localhost:3001 | System monitoring (admin/admin) |
| **Metrics** | http://localhost:9090 | Raw metrics data |

## Service Overview

### Architecture Summary

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Dashboard │────│  Coordinator    │────│   Redis Cache   │
│  (React/TS)     │    │  (Python/API)   │    │  (Shared Mem)   │
│  Port: 3000     │    │  Port: 8080     │    │  Port: 6379     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │ Rust Analyzer   │    │ Python Visualizer│    │   Monitoring    │
    │   Agent         │    │     Agent       │    │ (Grafana+Prom)  │
    │  (Background)   │    │  (Background)   │    │ Port: 3001/9090  │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Service Functions

- **Web Dashboard**: Primary user interface for submitting analyses and viewing results
- **Coordinator API**: Central service managing tasks and data flow
- **Rust Analyzer**: High-performance genomic sequence analysis engine
- **Python Visualizer**: Statistical analysis and visualization generation
- **Redis Cache**: Shared memory for agent communication and caching
- **Monitoring Stack**: System health and performance monitoring

## Web Dashboard

### Accessing the Dashboard

1. **Open your web browser**
2. **Navigate to**: http://localhost:3000
3. **Wait for the interface to load** (should take < 2 seconds)

### Dashboard Features

#### Main Navigation
- **Home**: System overview and quick stats
- **Trials**: Submit new analyses and view results
- **Agents**: Monitor analysis agent status
- **Results**: Browse and download completed analyses
- **Settings**: System configuration options

#### Submitting an Analysis

1. **Click "New Trial"** on the home page
2. **Upload genome file**: 
   - Supported formats: FASTA (.fasta, .fa, .fas)
   - Maximum size: 100MB
   - Example: E. coli K-12 genome
3. **Select organism type**:
   - E. coli (recommended for testing)
   - Custom (requires trait definition file)
4. **Configure parameters**:
   - Minimum traits: 2-5 (default: 2)
   - Window size: 300-1000bp (default: 300)
   - Confidence threshold: 0.5-0.9 (default: 0.7)
5. **Submit analysis**
6. **Monitor progress** in real-time

#### Viewing Results

1. **Navigate to "Results"** tab
2. **Filter by**:
   - Date range
   - Organism type
   - Confidence score
   - Analysis status
3. **Click on a result** to view details:
   - Pleiotropic genes identified
   - Confidence scores
   - Trait associations
   - Visualization plots
4. **Download options**:
   - JSON data export
   - CSV summary
   - High-resolution plots (PNG/SVG)

#### Real-time Updates

The dashboard automatically updates with:
- Analysis progress notifications
- Agent status changes
- System health alerts
- New results available

### Dashboard Troubleshooting

| Issue | Solution |
|-------|----------|
| Page won't load | Check if service is running: `curl http://localhost:3000` |
| Upload fails | Verify file format and size limits |
| No real-time updates | Check WebSocket connection in browser console |
| Analysis stuck | Check agent status in monitoring interface |

## API Interface

### Accessing the API

1. **Base URL**: http://localhost:8080
2. **API Documentation**: http://localhost:8080/docs
3. **OpenAPI Spec**: http://localhost:8080/openapi.json

### Authentication

Current implementation uses simple API key authentication:

```bash
# Set API key (if configured)
export API_KEY="your-api-key-here"

# Use in requests
curl -H "Authorization: Bearer $API_KEY" http://localhost:8080/api/trials
```

### Core API Endpoints

#### Health Check
```bash
# Check API health
curl http://localhost:8080/health

# Expected response:
# {"status": "healthy", "timestamp": "2023-12-07T10:30:00Z"}
```

#### Agent Management
```bash
# Get agent status
curl http://localhost:8080/api/agents/status | jq

# Expected response:
# [
#   {
#     "name": "rust_analyzer",
#     "status": "active",
#     "last_seen": "2023-12-07T10:29:45Z",
#     "tasks_completed": 15,
#     "current_load": 0.3
#   },
#   {
#     "name": "python_visualizer", 
#     "status": "idle",
#     "last_seen": "2023-12-07T10:29:50Z",
#     "tasks_completed": 12,
#     "current_load": 0.0
#   }
# ]
```

#### Trial Management
```bash
# Submit new analysis
curl -X POST http://localhost:8080/api/trials/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "genome_file": "path/to/genome.fasta",
    "organism": "ecoli",
    "parameters": {
      "min_traits": 2,
      "window_size": 300,
      "confidence_threshold": 0.7
    }
  }'

# Get trial status
curl http://localhost:8080/api/trials/{trial_id}/status

# Get trial results
curl http://localhost:8080/api/trials/{trial_id}/results
```

#### Results Retrieval
```bash
# List all results
curl http://localhost:8080/api/results?limit=10&offset=0

# Get specific result
curl http://localhost:8080/api/results/{result_id}

# Download result data
curl http://localhost:8080/api/results/{result_id}/download > result.json
```

### API Usage Examples

#### Python Client Example
```python
import requests
import json

# API base URL
BASE_URL = "http://localhost:8080"

class PleiotropyClient:
    def __init__(self, base_url=BASE_URL, api_key=None):
        self.base_url = base_url
        self.headers = {'Content-Type': 'application/json'}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
    
    def submit_analysis(self, genome_file, organism="ecoli", **params):
        \"\"\"Submit a new genome analysis\"\"\"
        data = {
            "genome_file": genome_file,
            "organism": organism,
            "parameters": params
        }
        response = requests.post(f"{self.base_url}/api/trials/analyze", 
                               json=data, headers=self.headers)
        return response.json()
    
    def get_trial_status(self, trial_id):
        \"\"\"Get status of a running trial\"\"\"
        response = requests.get(f"{self.base_url}/api/trials/{trial_id}/status",
                              headers=self.headers)
        return response.json()
    
    def get_results(self, trial_id):
        \"\"\"Get completed analysis results\"\"\"
        response = requests.get(f"{self.base_url}/api/trials/{trial_id}/results",
                              headers=self.headers)
        return response.json()

# Usage example
client = PleiotropyClient()

# Submit analysis
result = client.submit_analysis(
    genome_file="ecoli_k12.fasta",
    organism="ecoli",
    min_traits=2,
    confidence_threshold=0.7
)
trial_id = result['trial_id']

# Check status
status = client.get_trial_status(trial_id)
print(f"Analysis status: {status['status']}")

# Get results when complete
if status['status'] == 'completed':
    results = client.get_results(trial_id)
    print(f"Found {len(results['pleiotropic_genes'])} pleiotropic genes")
```

#### JavaScript/Node.js Example
```javascript
const axios = require('axios');

class PleiotropyClient {
    constructor(baseUrl = 'http://localhost:8080', apiKey = null) {
        this.baseUrl = baseUrl;
        this.headers = {'Content-Type': 'application/json'};
        if (apiKey) {
            this.headers['Authorization'] = `Bearer ${apiKey}`;
        }
    }

    async submitAnalysis(genomeFile, organism = 'ecoli', params = {}) {
        const data = {
            genome_file: genomeFile,
            organism: organism,
            parameters: params
        };
        
        const response = await axios.post(`${this.baseUrl}/api/trials/analyze`, 
                                        data, {headers: this.headers});
        return response.data;
    }

    async getTrialStatus(trialId) {
        const response = await axios.get(`${this.baseUrl}/api/trials/${trialId}/status`,
                                       {headers: this.headers});
        return response.data;
    }

    async getResults(trialId) {
        const response = await axios.get(`${this.baseUrl}/api/trials/${trialId}/results`,
                                       {headers: this.headers});
        return response.data;
    }
}

// Usage example
async function analyzeGenome() {
    const client = new PleiotropyClient();
    
    try {
        // Submit analysis
        const submission = await client.submitAnalysis(
            'ecoli_k12.fasta', 
            'ecoli', 
            {min_traits: 2, confidence_threshold: 0.7}
        );
        
        console.log(`Analysis submitted: ${submission.trial_id}`);
        
        // Poll for completion
        let status;
        do {
            await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5 seconds
            status = await client.getTrialStatus(submission.trial_id);
            console.log(`Status: ${status.status}`);
        } while (status.status === 'running');
        
        // Get results
        if (status.status === 'completed') {
            const results = await client.getResults(submission.trial_id);
            console.log(`Analysis complete! Found ${results.pleiotropic_genes.length} pleiotropic genes`);
        }
        
    } catch (error) {
        console.error('Analysis failed:', error.message);
    }
}

analyzeGenome();
```

## Monitoring Interface

### Accessing Grafana

1. **Navigate to**: http://localhost:3001
2. **Login credentials**:
   - Username: `admin`
   - Password: `admin` (change in production)
3. **Navigate to**: Dashboards → Swarm Dashboard

### Key Monitoring Panels

#### System Overview
- **Service Status**: All services health indicators
- **Uptime**: System availability metrics
- **Active Connections**: Current user/API connections
- **Resource Usage**: CPU, Memory, Disk utilization

#### Agent Activity
- **Agent Heartbeats**: Real-time agent communication status
- **Task Queue**: Pending and active analysis tasks
- **Processing Rates**: Analysis throughput metrics
- **Error Rates**: Failed analysis attempts

#### Performance Metrics
- **API Response Times**: Average and 95th percentile response times
- **Database Operations**: Query performance and connection pools
- **Cache Hit Rates**: Redis cache efficiency
- **Network I/O**: Data transfer rates

### Setting Up Alerts

1. **Navigate to**: Alerting → Alert Rules
2. **Create new rule**:
   - Query: `up{job="pleiotropy"} == 0`
   - Condition: `IS BELOW 1`
   - Evaluation: Every `30s` for `1m`
3. **Configure notification**:
   - Contact point: Email/Slack/Teams
   - Message template: Custom alert message
4. **Save and test**

### Custom Dashboards

Create custom dashboards for specific monitoring needs:

```json
{
  "dashboard": {
    "id": null,
    "title": "Custom Pleiotropy Dashboard",
    "tags": ["pleiotropy"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Analysis Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(analysis_completed_total[5m]) / rate(analysis_started_total[5m]) * 100",
            "legendFormat": "Success Rate %"
          }
        ]
      }
    ]
  }
}
```

## Command Line Tools

### System Management

```bash
# Start system
./start_system.sh --docker -d

# Check status
./start_system.sh --status

# View logs
./start_system.sh --logs

# Stop system
./start_system.sh --stop
```

### Direct Analysis Tools

```bash
# Run Rust analyzer directly
cd rust_impl
cargo run -- --input genome.fasta --output results/ --min-traits 2

# Run Python visualization
cd python_analysis
python trait_visualizer.py --input results/analysis.json --output plots/
```

### Database Operations

```bash
# Redis operations
docker exec pleiotropy-redis redis-cli ping
docker exec pleiotropy-redis redis-cli INFO memory
docker exec pleiotropy-redis redis-cli FLUSHDB  # Clear cache (careful!)

# Backup operations
docker exec pleiotropy-redis redis-cli BGSAVE
docker run --rm -v pleiotropy_redis_data:/data -v $(pwd):/backup alpine tar czf /backup/redis-backup.tar.gz /data
```

### Health Checks

```bash
# Comprehensive health check
curl http://localhost:8080/health && \
curl http://localhost:3000 && \
docker exec pleiotropy-redis redis-cli ping && \
echo "All services healthy"

# Agent status check
curl -s http://localhost:8080/api/agents/status | jq '.[] | {name: .name, status: .status, last_seen: .last_seen}'
```

## Analysis Workflows

### Basic E. coli Analysis

1. **Prepare genome file**:
   ```bash
   # Download E. coli K-12 genome
   wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/825/GCF_000005825.2_ASM582v2/GCF_000005825.2_ASM582v2_genomic.fna.gz
   gunzip GCF_000005825.2_ASM582v2_genomic.fna.gz
   mv GCF_000005825.2_ASM582v2_genomic.fna ecoli_k12.fasta
   ```

2. **Submit via Web Interface**:
   - Upload `ecoli_k12.fasta`
   - Select organism: E. coli
   - Use default parameters
   - Submit analysis

3. **Monitor progress**:
   - Watch real-time updates in dashboard
   - Check agent status in monitoring interface
   - Estimated completion: 5-15 minutes

4. **Review results**:
   - Pleiotropic genes identified
   - Confidence scores and trait associations
   - Download visualization plots

### Custom Organism Analysis

1. **Prepare trait definition file**:
   ```json
   {
     "organism": "custom",
     "traits": {
       "metabolism": {
         "keywords": ["metabolic", "enzyme", "pathway"],
         "go_terms": ["GO:0008152"]
       },
       "stress_response": {
         "keywords": ["stress", "response", "survival"],
         "go_terms": ["GO:0006950"]
       }
     }
   }
   ```

2. **Submit analysis**:
   - Upload genome and trait definition files
   - Select organism: Custom
   - Adjust confidence threshold if needed

3. **Interpret results**:
   - Focus on high-confidence predictions (>0.7)
   - Cross-reference with known gene functions
   - Validate findings with literature

### Batch Analysis

```bash
# Batch analysis script
#!/bin/bash

GENOMES_DIR="./genomes"
RESULTS_DIR="./batch_results"

mkdir -p "$RESULTS_DIR"

for genome_file in "$GENOMES_DIR"/*.fasta; do
    organism=$(basename "$genome_file" .fasta)
    echo "Analyzing $organism..."
    
    # Submit analysis via API
    trial_id=$(curl -s -X POST http://localhost:8080/api/trials/analyze \
        -H "Content-Type: application/json" \
        -d "{\"genome_file\": \"$genome_file\", \"organism\": \"ecoli\"}" | \
        jq -r '.trial_id')
    
    echo "Trial ID: $trial_id"
    
    # Wait for completion
    while true; do
        status=$(curl -s http://localhost:8080/api/trials/$trial_id/status | jq -r '.status')
        echo "Status: $status"
        
        if [ "$status" = "completed" ]; then
            break
        elif [ "$status" = "failed" ]; then
            echo "Analysis failed for $organism"
            break
        fi
        
        sleep 30
    done
    
    # Download results
    if [ "$status" = "completed" ]; then
        curl -s http://localhost:8080/api/trials/$trial_id/results > "$RESULTS_DIR/${organism}_results.json"
        echo "Results saved for $organism"
    fi
done

echo "Batch analysis complete"
```

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check Docker status
docker ps -a

# Check logs for errors
docker-compose logs coordinator
docker-compose logs rust_analyzer

# Restart specific service
docker-compose restart coordinator
```

#### Analysis Stuck
```bash
# Check agent status
curl http://localhost:8080/api/agents/status

# Check task queue
docker exec pleiotropy-redis redis-cli LLEN task_queue

# Restart agents
docker-compose restart rust_analyzer python_visualizer
```

#### Slow Performance
```bash
# Check system resources
docker stats

# Check Redis memory
docker exec pleiotropy-redis redis-cli INFO memory

# Scale services if needed
docker-compose up -d --scale rust_analyzer=2
```

#### Connection Issues
```bash
# Check network connectivity
curl -I http://localhost:8080
curl -I http://localhost:3000
curl -I http://localhost:3001

# Check firewall settings
sudo ufw status

# Restart network stack
docker-compose down && docker-compose up -d
```

### Getting Help

1. **Check logs first**: `./start_system.sh --logs`
2. **Review monitoring dashboard**: http://localhost:3001
3. **Check API health**: http://localhost:8080/health
4. **Consult troubleshooting guides**: See MONITORING_GUIDE.md
5. **Create GitHub issue**: Include logs and system information

### Support Information

- **Documentation**: README.md, CLAUDE.md
- **API Reference**: http://localhost:8080/docs
- **Monitoring**: http://localhost:3001
- **GitHub Issues**: https://github.com/murr2k/pleiotropy/issues

---

*This user guide is updated regularly. Check the repository for the latest version and additional examples.*