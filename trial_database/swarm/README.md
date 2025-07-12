# Swarm Coordination System

This directory contains the swarm coordination system for the Genomic Pleiotropy Cryptanalysis project. The system orchestrates multiple specialized agents to perform distributed analysis tasks.

## Architecture

### Core Components

1. **Coordinator** (`coordinator.py`)
   - Central hub for task distribution and agent management
   - Handles task queuing, assignment, and result aggregation
   - Monitors agent health via heartbeats
   - Provides memory synchronization across agents

2. **Base Agent** (`base_agent.py`)
   - Abstract base class for all swarm agents
   - Handles registration, heartbeats, and task execution
   - Provides memory save/load functionality
   - Implements event publishing/subscription

3. **Specialized Agents**
   - **Rust Analyzer Agent** (`rust_analyzer_agent.py`): Handles cryptographic analysis using Rust components
   - **Python Visualizer Agent** (`python_visualizer_agent.py`): Creates visualizations and statistical analyses

4. **Integration Module** (`integration.py`)
   - Bridges swarm system with existing codebase
   - Provides backward compatibility functions
   - Implements workflow orchestration

## Communication Protocol

The system uses Redis for all inter-component communication:

### Channels
- `{namespace}:agents` - Agent registration and status
- `{namespace}:tasks` - Task queue and status
- `{namespace}:results` - Task results
- `{namespace}:heartbeats` - Agent heartbeat messages
- `{namespace}:events` - Event broadcasting

### Memory Keys
- `{namespace}:{agent_type}:{key}` - Agent-specific data
- `{namespace}:coordinator:state` - Coordinator state
- `{namespace}:analysis:{task_id}` - Analysis results

## Task Flow

1. Task submitted to coordinator
2. Coordinator finds suitable agent based on capabilities
3. Task assigned to agent with best performance score
4. Agent processes task and stores result
5. Coordinator aggregates result and updates statistics
6. Result available via API or memory

## Agent Capabilities

### Rust Analyzer
- `crypto_analysis` - Full cryptographic analysis pipeline
- `sequence_parsing` - Parse genomic sequences (FASTA/GenBank)
- `frequency_analysis` - Analyze codon frequencies
- `trait_extraction` - Extract traits from analysis results

### Python Visualizer
- `visualization` - Create various plot types
- `statistics` - Run statistical analyses
- `heatmap` - Frequency heatmaps
- `scatter` - Trait scatter plots
- `distribution` - Value distributions
- `report_generation` - Generate HTML reports

## API Endpoints

The coordinator exposes these endpoints (when running with web server):

- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `POST /tasks` - Submit new task
- `GET /tasks/{task_id}` - Get task status
- `GET /agents` - List active agents
- `GET /agents/{agent_id}` - Get agent status

## Running the System

### Docker (Recommended)
```bash
# Start all services
../../start_system.sh --docker

# Start in background
../../start_system.sh --docker -d

# View logs
../../start_system.sh --logs

# Stop services
../../start_system.sh --stop
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis
redis-server

# Start coordinator
python coordinator.py

# Start agents (in separate terminals)
python rust_analyzer_agent.py
python python_visualizer_agent.py
```

## Testing

```bash
# Run integration tests
python test_integration.py

# Run specific test
pytest test_integration.py::TestSwarmIntegration::test_coordinator_task_submission -v

# Run with coverage
pytest test_integration.py --cov=. --cov-report=html
```

## Performance Tuning

### Coordinator Settings
- `task_queue`: Async queue size (default: unlimited)
- `executor.max_workers`: Thread pool size (default: 10)
- `heartbeat_timeout`: Agent offline threshold (default: 30s)

### Agent Settings
- `heartbeat_interval`: How often to send heartbeats (default: 10s)
- `memory_report_interval`: Memory sync frequency (default: 30s)

### Redis Settings
- `maxmemory`: Set appropriate limit for your system
- `maxmemory-policy`: Use `allkeys-lru` for automatic cleanup

## Monitoring

The system includes Prometheus metrics and Grafana dashboards:

### Key Metrics
- `swarm_active_agents` - Number of active agents
- `swarm_task_queue_size` - Current queue size
- `swarm_tasks_completed_total` - Total completed tasks
- `swarm_task_duration_seconds` - Task execution time
- `swarm_agent_performance_score` - Agent performance scores

### Grafana Dashboard
Access at http://localhost:3001 (default password: admin/admin)

## Troubleshooting

### Agent Not Registering
- Check Redis connectivity
- Verify agent has unique ID
- Check coordinator logs for errors

### Tasks Not Being Assigned
- Verify agent capabilities match task type
- Check agent status (must be AVAILABLE)
- Ensure heartbeats are being sent

### Memory Issues
- Monitor Redis memory usage
- Adjust key expiration times
- Use memory profiling tools

### Performance Problems
- Check task distribution balance
- Monitor agent performance scores
- Profile slow operations

## Extension Points

### Adding New Agent Types

1. Create new agent class inheriting from `BaseSwarmAgent`
2. Implement `execute_task()` and `collect_metrics()`
3. Define agent capabilities
4. Add Dockerfile if needed
5. Update docker-compose.yml

### Adding New Task Types

1. Define task type in agent capabilities
2. Implement task handler in agent
3. Add visualization support if needed
4. Update integration module

### Custom Workflows

1. Define workflow in integration module
2. Chain tasks with dependencies
3. Add error handling and retries
4. Create workflow-specific visualizations