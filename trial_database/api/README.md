# Genomic Pleiotropy Trial Tracking API

FastAPI backend for tracking genomic pleiotropy analysis trials, results, and agent activities.

## Features

- **RESTful API** with full CRUD operations
- **WebSocket support** for real-time updates
- **JWT Authentication** for agent security
- **Batch operations** for efficient data handling
- **Async/await** throughout for high performance
- **OpenAPI documentation** at `/docs` and `/redoc`

## Quick Start

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export DATABASE_URL="postgresql+asyncpg://postgres:password@localhost/pleiotropy_trials"
export SECRET_KEY="your-secret-key-here"
```

3. Run the application:
```bash
uvicorn app.main:app --reload
```

4. Access the API:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Docker Deployment

```bash
# From the trial_database directory
docker-compose up -d
```

## API Usage Examples

### Authentication

#### Register a new agent:
```bash
curl -X POST "http://localhost:8000/api/v1/agents/register" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "analyzer-01",
    "password": "securepassword123",
    "role": "analyzer",
    "capabilities": ["genome_analysis", "pattern_recognition"]
  }'
```

#### Login to get access token:
```bash
curl -X POST "http://localhost:8000/api/v1/agents/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=analyzer-01&password=securepassword123"
```

### Trials

#### Create a new trial:
```bash
curl -X POST "http://localhost:8000/api/v1/trials/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "E. coli K-12 Pleiotropy Analysis",
    "description": "Analyzing pleiotropic genes in E. coli K-12 strain",
    "organism": "Escherichia coli K-12",
    "genome_file": "/data/genomes/ecoli_k12.fasta",
    "parameters": {
      "window_size": 1000,
      "min_confidence": 0.75,
      "trait_count": 5
    }
  }'
```

#### List trials with pagination:
```bash
curl "http://localhost:8000/api/v1/trials/?page=1&page_size=20&status=running"
```

#### Update trial status:
```bash
curl -X PATCH "http://localhost:8000/api/v1/trials/1" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "status": "running"
  }'
```

### Results

#### Add analysis results:
```bash
curl -X POST "http://localhost:8000/api/v1/results/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "trial_id": 1,
    "gene_id": "lacZ",
    "traits": ["lactose_metabolism", "stress_response"],
    "confidence_scores": {
      "lactose_metabolism": 0.92,
      "stress_response": 0.78
    },
    "codon_usage_bias": {
      "AUG": 0.95,
      "UAA": 0.03
    }
  }'
```

#### Batch create results:
```bash
curl -X POST "http://localhost:8000/api/v1/results/batch" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "create",
    "items": [
      {
        "trial_id": 1,
        "gene_id": "araB",
        "traits": ["arabinose_metabolism", "carbon_utilization"],
        "confidence_scores": {"arabinose_metabolism": 0.88, "carbon_utilization": 0.76},
        "codon_usage_bias": {"AUG": 0.93, "UAG": 0.02}
      },
      {
        "trial_id": 1,
        "gene_id": "trpA",
        "traits": ["tryptophan_biosynthesis", "amino_acid_metabolism"],
        "confidence_scores": {"tryptophan_biosynthesis": 0.94, "amino_acid_metabolism": 0.85},
        "codon_usage_bias": {"AUG": 0.91, "UGA": 0.04}
      }
    ]
  }'
```

### Progress Updates

#### Report progress:
```bash
curl -X POST "http://localhost:8000/api/v1/progress/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "trial_id": 1,
    "stage": "gene_analysis",
    "progress_percentage": 45.5,
    "current_task": "Analyzing codon usage patterns",
    "genes_processed": 2150,
    "total_genes": 4300
  }'
```

#### Get latest progress:
```bash
curl "http://localhost:8000/api/v1/progress/trial/1/latest" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### WebSocket Connection

```javascript
// JavaScript WebSocket client example
const ws = new WebSocket('ws://localhost:8000/ws/connect?client_id=ui-001&agent_name=dashboard');

ws.onopen = () => {
  console.log('Connected to WebSocket');
  
  // Subscribe to trial updates
  ws.send(JSON.stringify({
    type: 'subscribe',
    trial_id: 1
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Received:', message);
  
  // Handle different message types
  switch(message.type) {
    case 'progress_update':
      updateProgressBar(message.data);
      break;
    case 'result_added':
      addResultToTable(message.data);
      break;
    case 'trial_updated':
      refreshTrialStatus(message.data);
      break;
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};
```

## API Endpoints Summary

### Agents
- `POST /api/v1/agents/register` - Register new agent
- `POST /api/v1/agents/login` - Login and get token
- `GET /api/v1/agents/me` - Get current agent info
- `GET /api/v1/agents/` - List all agents (coordinator only)
- `PATCH /api/v1/agents/{id}` - Update agent
- `DELETE /api/v1/agents/{id}` - Deactivate agent

### Trials
- `POST /api/v1/trials/` - Create trial
- `GET /api/v1/trials/` - List trials
- `GET /api/v1/trials/{id}` - Get trial details
- `PATCH /api/v1/trials/{id}` - Update trial
- `DELETE /api/v1/trials/{id}` - Delete trial
- `POST /api/v1/trials/batch` - Batch create trials

### Results
- `POST /api/v1/results/` - Create result
- `GET /api/v1/results/` - List results
- `GET /api/v1/results/{id}` - Get result details
- `PATCH /api/v1/results/{id}` - Update/validate result
- `DELETE /api/v1/results/{id}` - Delete result
- `POST /api/v1/results/batch` - Batch create results
- `GET /api/v1/results/trial/{id}/summary` - Get trial results summary

### Progress
- `POST /api/v1/progress/` - Create progress update
- `GET /api/v1/progress/trial/{id}` - Get trial progress history
- `GET /api/v1/progress/trial/{id}/latest` - Get latest progress
- `PATCH /api/v1/progress/{id}` - Update progress
- `GET /api/v1/progress/active` - Get all active trials progress
- `POST /api/v1/progress/batch` - Batch create progress updates

### WebSocket
- `WS /ws/connect` - WebSocket connection endpoint
- `GET /ws/stats` - WebSocket connection statistics

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| DATABASE_URL | PostgreSQL connection string | postgresql+asyncpg://postgres:password@localhost/pleiotropy_trials |
| SECRET_KEY | JWT secret key | your-secret-key-here-change-in-production |
| ACCESS_TOKEN_EXPIRE_MINUTES | Token expiration time | 30 |
| CORS_ORIGINS | Allowed CORS origins | ["http://localhost:3000", "http://localhost:5173"] |
| LOG_LEVEL | Logging level | INFO |

## Development

### Database Migrations

```bash
# Create a new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_trials.py
```

### Performance Considerations

1. **Connection Pooling**: The API uses SQLAlchemy's async connection pooling
2. **Batch Operations**: Use batch endpoints for bulk data operations
3. **WebSocket Efficiency**: Messages are queued and processed asynchronously
4. **Pagination**: All list endpoints support pagination to limit data transfer
5. **Indexing**: Database indexes on frequently queried fields (trial_id, gene_id, status)

## Security

1. **JWT Authentication**: All endpoints except registration require valid tokens
2. **Password Hashing**: Bcrypt with salt for password storage
3. **Role-Based Access**: Certain operations restricted by agent role
4. **CORS Configuration**: Explicitly configured allowed origins
5. **Input Validation**: Pydantic models validate all input data

## Monitoring

The API provides several monitoring endpoints:

- `/health` - Basic health check
- `/ws/stats` - WebSocket connection statistics
- `/api/v1/agents/stats/active` - Active agent statistics

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check DATABASE_URL is correct
   - Ensure PostgreSQL is running
   - Verify network connectivity

2. **WebSocket Disconnections**
   - Check client_id uniqueness
   - Verify network stability
   - Monitor message queue size

3. **Authentication Errors**
   - Ensure token is included in Authorization header
   - Check token expiration
   - Verify agent is active

### Logs

Application logs include:
- Request/response details
- WebSocket connections/disconnections
- Database query performance
- Error stack traces

## License

This project is part of the Genomic Pleiotropy Cryptanalysis system.