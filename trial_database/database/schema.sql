-- Genomic Pleiotropy Cryptanalysis Trial Database Schema
-- SQLite database for tracking experiments, results, and agent activities

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Agents table: tracks AI agents performing experiments
CREATE TABLE IF NOT EXISTS agents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL CHECK(type IN ('orchestrator', 'database_architect', 'genome_analyst', 'crypto_specialist', 'visualization_engineer', 'performance_optimizer')),
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'idle', 'offline')),
    last_heartbeat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tasks_completed INTEGER DEFAULT 0,
    memory_keys JSON DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trials table: main experimental runs
CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    parameters JSON NOT NULL,  -- Stores experimental parameters as JSON
    hypothesis TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by_agent INTEGER NOT NULL,
    FOREIGN KEY (created_by_agent) REFERENCES agents(id)
);

-- Results table: stores outcomes of trials
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id INTEGER NOT NULL,
    metrics JSON NOT NULL,  -- Stores various metrics as JSON (e.g., codon bias, trait confidence)
    confidence_score REAL CHECK(confidence_score >= 0.0 AND confidence_score <= 1.0),
    visualizations JSON DEFAULT '{}',  -- Stores paths to generated visualizations
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    agent_id INTEGER NOT NULL,
    FOREIGN KEY (trial_id) REFERENCES trials(id) ON DELETE CASCADE,
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

-- Progress table: tracks task progress for long-running operations
CREATE TABLE IF NOT EXISTS progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id INTEGER NOT NULL,
    task_id TEXT NOT NULL,  -- Unique identifier for the task
    status TEXT NOT NULL CHECK(status IN ('started', 'in_progress', 'completed', 'failed')),
    message TEXT,
    percentage INTEGER DEFAULT 0 CHECK(percentage >= 0 AND percentage <= 100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

-- Create indices for performance
CREATE INDEX IF NOT EXISTS idx_trials_status ON trials(status);
CREATE INDEX IF NOT EXISTS idx_trials_created_by ON trials(created_by_agent);
CREATE INDEX IF NOT EXISTS idx_results_trial ON results(trial_id);
CREATE INDEX IF NOT EXISTS idx_results_confidence ON results(confidence_score);
CREATE INDEX IF NOT EXISTS idx_progress_agent ON progress(agent_id);
CREATE INDEX IF NOT EXISTS idx_progress_task ON progress(task_id);
CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(type);
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);

-- Triggers to update timestamps
CREATE TRIGGER IF NOT EXISTS update_trials_timestamp 
AFTER UPDATE ON trials
BEGIN
    UPDATE trials SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_agents_timestamp 
AFTER UPDATE ON agents
BEGIN
    UPDATE agents SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;