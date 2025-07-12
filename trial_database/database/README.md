# Trial Database Documentation

## Overview

The trial database is a SQLite-based system designed to track experiments, results, and agent activities in the Genomic Pleiotropy Cryptanalysis project. It provides a robust foundation for managing scientific trials with full audit trails and progress tracking.

## Database Schema

### Core Tables

#### 1. **agents**
Tracks AI agents performing experiments.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| name | TEXT | Unique agent identifier |
| type | TEXT | Agent type (orchestrator, database_architect, etc.) |
| status | TEXT | Current status (active, idle, offline) |
| last_heartbeat | TIMESTAMP | Last activity timestamp |
| tasks_completed | INTEGER | Counter for completed tasks |
| memory_keys | JSON | List of memory namespace keys |
| created_at | TIMESTAMP | Creation timestamp |
| updated_at | TIMESTAMP | Last update timestamp |

#### 2. **trials**
Main experimental runs.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| name | TEXT | Trial name |
| description | TEXT | Detailed description |
| parameters | JSON | Experimental parameters |
| hypothesis | TEXT | Scientific hypothesis |
| status | TEXT | Trial status (pending, running, completed, failed, cancelled) |
| created_at | TIMESTAMP | Creation timestamp |
| updated_at | TIMESTAMP | Last update timestamp |
| created_by_agent | INTEGER | Foreign key to agents |

#### 3. **results**
Stores outcomes of trials.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| trial_id | INTEGER | Foreign key to trials |
| metrics | JSON | Various metrics (codon bias, trait confidence, etc.) |
| confidence_score | REAL | Overall confidence (0.0-1.0) |
| visualizations | JSON | Paths to generated visualizations |
| timestamp | TIMESTAMP | Result timestamp |
| agent_id | INTEGER | Foreign key to agents |

#### 4. **progress**
Tracks task progress for long-running operations.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| agent_id | INTEGER | Foreign key to agents |
| task_id | TEXT | Unique task identifier |
| status | TEXT | Progress status (started, in_progress, completed, failed) |
| message | TEXT | Status message |
| percentage | INTEGER | Progress percentage (0-100) |
| timestamp | TIMESTAMP | Update timestamp |

### Indices

- `idx_trials_status`: Optimizes status-based queries
- `idx_trials_created_by`: Optimizes agent-based queries
- `idx_results_trial`: Optimizes trial result lookups
- `idx_results_confidence`: Optimizes confidence-based queries
- `idx_progress_agent`: Optimizes agent progress lookups
- `idx_progress_task`: Optimizes task-based queries
- `idx_agents_type`: Optimizes agent type queries
- `idx_agents_status`: Optimizes agent status queries

## Usage

### Initialization

```bash
# Create database with test data
python trial_database/database/init_db.py --seed

# Force recreate database
python trial_database/database/init_db.py --force --seed
```

### Migrations

```bash
# Check migration status
python trial_database/database/migrations.py status

# Apply all pending migrations
python trial_database/database/migrations.py migrate

# Rollback to specific version
python trial_database/database/migrations.py rollback 003
```

### Common Operations

```python
from trial_database.database.utils import DatabaseUtils

db = DatabaseUtils()

# Create a new trial
trial = db.create_trial(
    name="E.coli Pattern Analysis",
    description="Analyze codon patterns in E.coli genome",
    parameters={"window_size": 1000, "overlap": 500},
    hypothesis="Pleiotropic genes show distinct patterns",
    agent_id=1
)

# Add a result
result = db.add_result(
    trial_id=trial.id,
    metrics={"genes_analyzed": 4289, "patterns_found": 156},
    confidence_score=0.87,
    agent_id=1,
    visualizations={"heatmap": "path/to/heatmap.png"}
)

# Update progress
progress = db.create_progress(agent_id=1, task_id="analysis_123")
db.update_progress("analysis_123", 50, "Processing gene sequences")
db.update_progress("analysis_123", 100, "Analysis complete")

# Query operations
active_agents = db.get_active_agents()
high_conf_results = db.get_high_confidence_results(min_confidence=0.85)
stats = db.get_trial_statistics()
```

## Design Decisions

### 1. **SQLite Choice**
- **Portability**: Single file database, easy to backup and share
- **Simplicity**: No server setup required
- **Sufficient Performance**: Adequate for trial tracking workload
- **Built-in JSON Support**: Native JSON columns for flexible data

### 2. **JSON Fields**
- **parameters**: Flexible experimental parameters without schema changes
- **metrics**: Extensible result metrics
- **memory_keys**: Dynamic list of memory namespaces
- **visualizations**: Variable visualization outputs

### 3. **Status Enums**
- Enforced via CHECK constraints
- Clear state transitions
- Prevents invalid states

### 4. **Timestamp Tracking**
- Automatic created_at/updated_at
- Audit trail via timestamps
- Progress timeline tracking

### 5. **Foreign Key Constraints**
- Data integrity enforcement
- Cascade deletes for results
- Agent-trial relationships

## Migration System

The migration system supports:
- **Version Tracking**: Sequential version numbers
- **Up/Down Migrations**: Reversible schema changes
- **Atomic Operations**: All-or-nothing migrations
- **Migration History**: Full audit trail

Available migrations:
1. **001**: Add tags field to trials
2. **002**: Add metadata and computation time to results
3. **003**: Add capabilities and configuration to agents
4. **004**: Add trial dependency tracking
5. **005**: Add audit logging

## Best Practices

1. **Always use transactions** for multi-table operations
2. **Update heartbeats** regularly for active agents
3. **Set appropriate indices** for frequent queries
4. **Use JSON fields** for variable data structures
5. **Clean up old progress entries** periodically
6. **Backup database** before migrations

## Performance Considerations

1. **Index Usage**: Ensure queries use appropriate indices
2. **JSON Queries**: Minimize complex JSON field queries
3. **Batch Operations**: Use bulk inserts/updates when possible
4. **Connection Pooling**: Reuse database connections
5. **Progress Cleanup**: Remove old completed progress entries

## Security Notes

1. **Input Validation**: All inputs validated via SQLAlchemy
2. **SQL Injection**: Prevented by parameterized queries
3. **Access Control**: Implement at application layer
4. **Backup Strategy**: Regular automated backups recommended

## Future Enhancements

1. **Full-Text Search**: Add FTS5 for trial/result search
2. **Partitioning**: Archive old trials to separate tables
3. **Replication**: Add read replicas for scaling
4. **Compression**: Compress large JSON fields
5. **Event Streaming**: Add triggers for real-time updates