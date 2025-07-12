# API Reference

## 🌐 Base URL

```
http://localhost:8080/api
```

## 🔐 Authentication

All API requests require authentication using JWT tokens.

### Get Authentication Token

```http
POST /auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "password123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Using the Token

Include the token in the Authorization header:
```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## 📍 Endpoints

### Health Check

Check system health and service status.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-12T10:30:00Z",
  "services": {
    "database": "up",
    "redis": "up",
    "rust_analyzer": "up",
    "python_visualizer": "up"
  },
  "version": "1.0.0"
}
```

### Analysis Endpoints

#### Submit Analysis

Submit a new genomic analysis task.

```http
POST /analyze
Content-Type: application/json
Authorization: Bearer {token}

{
  "genome_file": "path/to/genome.fasta",
  "traits_file": "path/to/traits.json",
  "parameters": {
    "min_confidence": 0.4,
    "min_traits": 2,
    "window_size": 300,
    "use_neurodna": true
  },
  "metadata": {
    "organism": "E. coli K-12",
    "description": "Test analysis"
  }
}
```

**Response:**
```json
{
  "trial_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "estimated_time": 30,
  "queue_position": 3,
  "created_at": "2025-01-12T10:30:00Z"
}
```

#### Get Analysis Status

```http
GET /analyze/{trial_id}/status
Authorization: Bearer {token}
```

**Response:**
```json
{
  "trial_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 0.45,
  "current_step": "frequency_analysis",
  "elapsed_time": 15.3,
  "estimated_remaining": 18.2,
  "messages": [
    {
      "timestamp": "2025-01-12T10:30:15Z",
      "level": "info",
      "message": "Completed sequence parsing"
    }
  ]
}
```

#### Get Analysis Results

```http
GET /analyze/{trial_id}/results
Authorization: Bearer {token}
```

**Response:**
```json
{
  "trial_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "completion_time": "2025-01-12T10:30:45Z",
  "analysis_time": 28.7,
  "results": {
    "sequences_analyzed": 1,
    "pleiotropic_genes": [
      {
        "gene_id": "NC_000913.3",
        "traits": ["stress_response", "regulatory", "carbon_metabolism"],
        "confidence": 0.75,
        "start_position": 1234,
        "end_position": 5678,
        "codon_patterns": {
          "CTG": 0.08,
          "GAA": 0.12
        }
      }
    ],
    "trait_summary": {
      "stress_response": 15,
      "regulatory": 12,
      "carbon_metabolism": 8
    },
    "frequency_table": {
      "total_codons": 1543289,
      "codon_frequencies": {
        "AAA": 0.0234,
        "AAC": 0.0187
        // ... more codons
      }
    }
  },
  "metadata": {
    "organism": "E. coli K-12",
    "parameters_used": {
      "min_confidence": 0.4,
      "min_traits": 2
    }
  }
}
```

#### Cancel Analysis

```http
POST /analyze/{trial_id}/cancel
Authorization: Bearer {token}
```

**Response:**
```json
{
  "trial_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "cancelled",
  "message": "Analysis cancelled by user"
}
```

### Trial Management

#### List Trials

```http
GET /trials?page=1&limit=20&status=completed&sort_by=created_at&order=desc
Authorization: Bearer {token}
```

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `limit` (int): Items per page (default: 20, max: 100)
- `status` (string): Filter by status (queued, processing, completed, failed, cancelled)
- `sort_by` (string): Sort field (created_at, completed_at, analysis_time)
- `order` (string): Sort order (asc, desc)
- `organism` (string): Filter by organism
- `date_from` (string): Filter by date (ISO 8601)
- `date_to` (string): Filter by date (ISO 8601)

**Response:**
```json
{
  "trials": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "E. coli analysis #42",
      "status": "completed",
      "created_at": "2025-01-12T10:30:00Z",
      "completed_at": "2025-01-12T10:30:45Z",
      "organism": "E. coli K-12",
      "pleiotropic_genes_found": 23,
      "confidence_avg": 0.68
    }
  ],
  "pagination": {
    "total": 156,
    "page": 1,
    "limit": 20,
    "pages": 8
  }
}
```

#### Get Trial Details

```http
GET /trials/{trial_id}
Authorization: Bearer {token}
```

#### Delete Trial

```http
DELETE /trials/{trial_id}
Authorization: Bearer {token}
```

### Agent Management

#### List Agents

```http
GET /agents/status
Authorization: Bearer {token}
```

**Response:**
```json
{
  "agents": [
    {
      "id": "rust-analyzer-01",
      "name": "Rust Analyzer 01",
      "type": "rust_analyzer",
      "status": "active",
      "last_heartbeat": "2025-01-12T10:30:00Z",
      "current_task": "550e8400-e29b-41d4-a716-446655440000",
      "tasks_completed": 143,
      "error_rate": 0.02,
      "average_task_time": 25.3
    }
  ],
  "summary": {
    "total_agents": 5,
    "active_agents": 4,
    "idle_agents": 1,
    "failed_agents": 0
  }
}
```

#### Get Agent Logs

```http
GET /agents/{agent_id}/logs?limit=100
Authorization: Bearer {token}
```

### Data Management

#### Upload Genome File

```http
POST /data/upload
Content-Type: multipart/form-data
Authorization: Bearer {token}

file: genome.fasta
metadata: {"organism": "E. coli", "strain": "K-12"}
```

**Response:**
```json
{
  "file_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "filename": "genome.fasta",
  "size": 4641652,
  "hash": "sha256:2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae",
  "upload_time": "2025-01-12T10:30:00Z"
}
```

#### List Uploaded Files

```http
GET /data/files
Authorization: Bearer {token}
```

#### Download Results

```http
GET /data/download/{trial_id}
Authorization: Bearer {token}
Accept: application/json, text/csv, application/zip
```

### Visualization

#### Generate Visualization

```http
POST /visualize
Content-Type: application/json
Authorization: Bearer {token}

{
  "trial_id": "550e8400-e29b-41d4-a716-446655440000",
  "type": "trait_network",
  "parameters": {
    "min_edge_weight": 0.5,
    "layout": "force-directed",
    "color_scheme": "viridis"
  }
}
```

**Response:**
```json
{
  "visualization_id": "viz-123456",
  "type": "trait_network",
  "format": "html",
  "url": "/visualizations/viz-123456.html",
  "preview_url": "/visualizations/viz-123456.png"
}
```

#### Available Visualization Types

- `trait_network`: Gene-trait interaction network
- `codon_heatmap`: Codon usage heatmap
- `confidence_distribution`: Confidence score distribution
- `pca_plot`: PCA of trait patterns
- `temporal_analysis`: Analysis over time

### Statistics

#### Get System Statistics

```http
GET /stats/system
Authorization: Bearer {token}
```

**Response:**
```json
{
  "period": "last_30_days",
  "total_analyses": 1247,
  "successful_analyses": 1231,
  "failed_analyses": 16,
  "average_analysis_time": 28.3,
  "total_genes_detected": 15892,
  "unique_users": 47,
  "popular_organisms": [
    {"organism": "E. coli K-12", "count": 823},
    {"organism": "Synthetic", "count": 312}
  ],
  "resource_usage": {
    "cpu_avg": 0.65,
    "memory_avg": 0.58,
    "storage_used": "12.3 GB"
  }
}
```

## 🔄 WebSocket API

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/progress');

ws.onopen = () => {
  // Subscribe to trial updates
  ws.send(JSON.stringify({
    action: 'subscribe',
    trial_id: '550e8400-e29b-41d4-a716-446655440000'
  }));
};
```

### Message Types

#### Progress Update
```json
{
  "type": "progress",
  "trial_id": "550e8400-e29b-41d4-a716-446655440000",
  "progress": 0.75,
  "step": "trait_extraction",
  "message": "Extracting trait patterns"
}
```

#### Status Change
```json
{
  "type": "status_change",
  "trial_id": "550e8400-e29b-41d4-a716-446655440000",
  "old_status": "processing",
  "new_status": "completed"
}
```

#### Error
```json
{
  "type": "error",
  "trial_id": "550e8400-e29b-41d4-a716-446655440000",
  "error": "Out of memory",
  "details": "Required 8GB, available 4GB"
}
```

## 🔍 Error Responses

### Standard Error Format

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "Trial not found",
    "details": {
      "trial_id": "550e8400-e29b-41d4-a716-446655440000"
    },
    "timestamp": "2025-01-12T10:30:00Z",
    "request_id": "req-123456"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Invalid or missing authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `RESOURCE_NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

## ⚡ Rate Limits

| Endpoint | Rate Limit | Window |
|----------|------------|--------|
| `/analyze` | 100 | 1 hour |
| `/trials` | 1000 | 1 hour |
| `/agents/*` | 500 | 1 hour |
| `/data/upload` | 50 | 1 hour |
| Other endpoints | 2000 | 1 hour |

**Rate Limit Headers:**
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1673521234
```

## 🔨 Client Libraries

### Python
```python
from pleiotropy import PleiotropyClient

client = PleiotropyClient(
    base_url="http://localhost:8080",
    api_key="your-api-key"
)

# Submit analysis
result = client.analyze(
    genome_file="genome.fasta",
    traits_file="traits.json",
    min_confidence=0.4
)

# Get results
analysis = client.get_results(result.trial_id)
print(f"Found {len(analysis.pleiotropic_genes)} genes")
```

### JavaScript/TypeScript
```typescript
import { PleiotropyClient } from '@pleiotropy/client';

const client = new PleiotropyClient({
  baseURL: 'http://localhost:8080',
  apiKey: 'your-api-key'
});

// Submit analysis
const result = await client.analyze({
  genomeFile: 'genome.fasta',
  traitsFile: 'traits.json',
  parameters: {
    minConfidence: 0.4
  }
});

// Subscribe to progress
client.onProgress(result.trialId, (progress) => {
  console.log(`Progress: ${progress.percent}%`);
});
```

### cURL Examples
```bash
# Submit analysis
curl -X POST http://localhost:8080/api/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "genome_file": "test.fasta",
    "traits_file": "traits.json"
  }'

# Get results
curl http://localhost:8080/api/analyze/$TRIAL_ID/results \
  -H "Authorization: Bearer $TOKEN"
```

## 📚 OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
```
http://localhost:8080/openapi.json
```

Interactive API documentation (Swagger UI):
```
http://localhost:8080/docs
```

ReDoc documentation:
```
http://localhost:8080/redoc
```

---

*For more examples and use cases, see the [Tutorial](Tutorial) and [Examples](Examples) sections.*