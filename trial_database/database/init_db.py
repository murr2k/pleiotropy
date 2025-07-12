#!/usr/bin/env python3
"""
Initialize the Genomic Pleiotropy Cryptanalysis trial database.
Creates the database, applies schema, and optionally seeds with test data.
"""
import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from trial_database.database.models import (
    Base, Agent, Trial, Result, Progress,
    AgentType, AgentStatus, TrialStatus, ProgressStatus
)


class DatabaseInitializer:
    """Handles database initialization and seeding."""
    
    def __init__(self, db_path: str = "trial_database/database/trials.db"):
        """Initialize the database manager."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_url = f"sqlite:///{self.db_path}"
        self.engine = None
        self.SessionLocal = None
    
    def create_database(self):
        """Create the database and apply schema."""
        print(f"Creating database at: {self.db_path}")
        
        # Create engine with foreign key support
        self.engine = create_engine(
            self.db_url, 
            echo=False,
            connect_args={"check_same_thread": False}
        )
        
        # Enable foreign keys for SQLite
        with self.engine.connect() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
        
        # Create all tables
        Base.metadata.create_all(bind=self.engine)
        
        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        print("Database schema created successfully!")
    
    def apply_raw_schema(self):
        """Apply the raw SQL schema for any SQLite-specific features."""
        schema_path = self.db_path.parent / "schema.sql"
        if schema_path.exists():
            print(f"Applying raw SQL schema from: {schema_path}")
            conn = sqlite3.connect(str(self.db_path))
            with open(schema_path, 'r') as f:
                conn.executescript(f.read())
            conn.close()
    
    def seed_agents(self, session):
        """Seed the database with agent data."""
        print("Seeding agents...")
        
        agents_data = [
            {
                "name": "orchestrator-001",
                "type": AgentType.ORCHESTRATOR,
                "status": AgentStatus.ACTIVE,
                "memory_keys": ["swarm-auto-centralized-1752300927219/orchestrator"]
            },
            {
                "name": "db-architect-001",
                "type": AgentType.DATABASE_ARCHITECT,
                "status": AgentStatus.ACTIVE,
                "memory_keys": ["swarm-auto-centralized-1752300927219/database-architect"]
            },
            {
                "name": "genome-analyst-001",
                "type": AgentType.GENOME_ANALYST,
                "status": AgentStatus.ACTIVE,
                "memory_keys": ["swarm-auto-centralized-1752300927219/genome-analyst"]
            },
            {
                "name": "crypto-specialist-001",
                "type": AgentType.CRYPTO_SPECIALIST,
                "status": AgentStatus.ACTIVE,
                "memory_keys": ["swarm-auto-centralized-1752300927219/crypto-specialist"]
            },
            {
                "name": "viz-engineer-001",
                "type": AgentType.VISUALIZATION_ENGINEER,
                "status": AgentStatus.IDLE,
                "memory_keys": ["swarm-auto-centralized-1752300927219/visualization-engineer"]
            },
            {
                "name": "perf-optimizer-001",
                "type": AgentType.PERFORMANCE_OPTIMIZER,
                "status": AgentStatus.OFFLINE,
                "memory_keys": ["swarm-auto-centralized-1752300927219/performance-optimizer"]
            }
        ]
        
        agents = []
        for data in agents_data:
            agent = Agent(**data)
            session.add(agent)
            agents.append(agent)
        
        session.commit()
        return agents
    
    def seed_trials(self, session, agents):
        """Seed the database with trial data."""
        print("Seeding trials...")
        
        trials_data = [
            {
                "name": "E.coli K-12 Codon Bias Analysis",
                "description": "Initial analysis of codon usage patterns in E.coli K-12 genome",
                "parameters": {
                    "organism": "E.coli K-12",
                    "window_size": 1000,
                    "overlap": 500,
                    "min_confidence": 0.7,
                    "traits": ["motility", "metabolism", "stress_response"]
                },
                "hypothesis": "Pleiotropic genes will show distinct codon usage patterns for different traits",
                "status": TrialStatus.COMPLETED,
                "created_by_agent": agents[2].id  # genome-analyst
            },
            {
                "name": "Cryptographic Pattern Detection",
                "description": "Apply frequency analysis to identify polyalphabetic patterns",
                "parameters": {
                    "algorithm": "chi_squared_frequency",
                    "key_length_range": [3, 9],
                    "confidence_threshold": 0.8,
                    "parallel_threads": 8
                },
                "hypothesis": "Regulatory sequences act as cipher keys for trait expression",
                "status": TrialStatus.RUNNING,
                "created_by_agent": agents[3].id  # crypto-specialist
            },
            {
                "name": "Multi-trait Visualization Test",
                "description": "Generate heatmaps for trait-codon associations",
                "parameters": {
                    "visualization_type": "heatmap",
                    "color_scheme": "viridis",
                    "clustering": True,
                    "interactive": True
                },
                "hypothesis": "Visual patterns will reveal hidden trait relationships",
                "status": TrialStatus.PENDING,
                "created_by_agent": agents[4].id  # viz-engineer
            }
        ]
        
        trials = []
        for data in trials_data:
            trial = Trial(**data)
            session.add(trial)
            trials.append(trial)
        
        session.commit()
        return trials
    
    def seed_results(self, session, trials, agents):
        """Seed the database with result data."""
        print("Seeding results...")
        
        # Result for completed trial
        result1 = Result(
            trial_id=trials[0].id,
            metrics={
                "total_genes_analyzed": 4289,
                "pleiotropic_genes_found": 156,
                "avg_codon_bias": 0.342,
                "trait_separation_score": 0.827,
                "chi_squared_p_value": 0.0001
            },
            confidence_score=0.89,
            visualizations={
                "codon_heatmap": "visualizations/trial_1/codon_heatmap.png",
                "trait_network": "visualizations/trial_1/trait_network.html"
            },
            agent_id=agents[2].id
        )
        
        # Partial result for running trial
        result2 = Result(
            trial_id=trials[1].id,
            metrics={
                "patterns_detected": 42,
                "key_candidates": 7,
                "decryption_success_rate": 0.73
            },
            confidence_score=0.65,
            visualizations={},
            agent_id=agents[3].id
        )
        
        session.add_all([result1, result2])
        session.commit()
        
        return [result1, result2]
    
    def seed_progress(self, session, agents, trials):
        """Seed the database with progress data."""
        print("Seeding progress entries...")
        
        progress_entries = [
            Progress(
                agent_id=agents[2].id,
                task_id=f"trial_{trials[0].id}_analysis",
                status=ProgressStatus.COMPLETED,
                message="Successfully analyzed all genes",
                percentage=100
            ),
            Progress(
                agent_id=agents[3].id,
                task_id=f"trial_{trials[1].id}_crypto",
                status=ProgressStatus.IN_PROGRESS,
                message="Running frequency analysis on codon patterns",
                percentage=73
            ),
            Progress(
                agent_id=agents[4].id,
                task_id=f"trial_{trials[2].id}_viz_prep",
                status=ProgressStatus.STARTED,
                message="Preparing visualization pipeline",
                percentage=15
            )
        ]
        
        session.add_all(progress_entries)
        session.commit()
        
        return progress_entries
    
    def seed_database(self):
        """Seed the database with test data."""
        session = self.SessionLocal()
        
        try:
            # Seed in order due to foreign key constraints
            agents = self.seed_agents(session)
            trials = self.seed_trials(session, agents)
            results = self.seed_results(session, trials, agents)
            progress = self.seed_progress(session, agents, trials)
            
            print(f"Seeding complete!")
            print(f"  - {len(agents)} agents")
            print(f"  - {len(trials)} trials")
            print(f"  - {len(results)} results")
            print(f"  - {len(progress)} progress entries")
            
        except Exception as e:
            session.rollback()
            print(f"Error during seeding: {e}")
            raise
        finally:
            session.close()
    
    def verify_database(self):
        """Verify the database was created correctly."""
        session = self.SessionLocal()
        
        try:
            agent_count = session.query(Agent).count()
            trial_count = session.query(Trial).count()
            result_count = session.query(Result).count()
            progress_count = session.query(Progress).count()
            
            print("\nDatabase verification:")
            print(f"  - Agents: {agent_count}")
            print(f"  - Trials: {trial_count}")
            print(f"  - Results: {result_count}")
            print(f"  - Progress entries: {progress_count}")
            
            # Test a query
            active_trials = session.query(Trial).filter(
                Trial.status == TrialStatus.RUNNING
            ).count()
            print(f"  - Active trials: {active_trials}")
            
        finally:
            session.close()


def main():
    """Main initialization function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize the trial database")
    parser.add_argument(
        "--db-path",
        default="trial_database/database/trials.db",
        help="Path to the database file"
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed the database with test data"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreate the database (will delete existing)"
    )
    
    args = parser.parse_args()
    
    # Initialize database
    initializer = DatabaseInitializer(args.db_path)
    
    # Remove existing database if force flag is set
    if args.force and initializer.db_path.exists():
        print(f"Removing existing database: {initializer.db_path}")
        initializer.db_path.unlink()
    
    # Create database
    initializer.create_database()
    
    # Apply raw schema (for triggers, etc.)
    # initializer.apply_raw_schema()  # Optional, as SQLAlchemy handles most needs
    
    # Seed if requested
    if args.seed:
        initializer.seed_database()
    
    # Verify
    initializer.verify_database()
    
    print(f"\nDatabase initialized successfully at: {initializer.db_path}")


if __name__ == "__main__":
    main()