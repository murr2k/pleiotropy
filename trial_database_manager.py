#!/usr/bin/env python3
"""
Trial Database Manager for Genomic Pleiotropy Cryptanalysis
Manages the trial database with real-time tracking and agent coordination.

Memory Namespace: swarm-pleiotropy-analysis-1752302124
"""

import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from trial_database.database.models import (
    Agent, Trial, Result, Progress,
    AgentType, AgentStatus, TrialStatus, ProgressStatus
)
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


class PleiotropyTrialManager:
    """Manages trials for pleiotropic analysis with real-time tracking."""
    
    def __init__(self, db_path: str = "/home/murr2k/projects/agentic/pleiotropy/trial_database/database/trials.db"):
        self.db_path = Path(db_path)
        self.db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(
            self.db_url,
            echo=False,
            connect_args={"check_same_thread": False}
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.memory_namespace = "swarm-pleiotropy-analysis-1752302124"
        
        # Initialize database if needed
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Ensure the database and tables exist."""
        try:
            # Test if tables exist by trying a simple query
            session = self.SessionLocal()
            session.query(Agent).count()
            session.close()
        except Exception:
            # Database doesn't exist or tables missing, create them
            print("🔄 Initializing database...")
            from trial_database.database.models import Base
            
            # Enable foreign keys for SQLite
            with self.engine.connect() as conn:
                conn.execute(text("PRAGMA foreign_keys = ON"))
            
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            print("✅ Database initialized successfully!")
        
    def create_ecoli_trial(self) -> int:
        """Create the main E. coli K-12 pleiotropic analysis trial."""
        session = self.SessionLocal()
        
        try:
            # Get or create the database manager agent
            db_agent = session.query(Agent).filter(
                Agent.type == AgentType.DATABASE_ARCHITECT
            ).first()
            
            if not db_agent:
                # Create a database architect agent
                db_agent = Agent(
                    name="database-manager-main",
                    type=AgentType.DATABASE_ARCHITECT,
                    status=AgentStatus.ACTIVE,
                    memory_keys=[f"{self.memory_namespace}/database-manager"]
                )
                session.add(db_agent)
                session.commit()
                print("✓ Created database architect agent")
            
            # Trial parameters
            trial_params = {
                "memory_namespace": self.memory_namespace,
                "input_files": {
                    "genome_sequence": "test_ecoli_sample.fasta",
                    "trait_definitions": "test_traits.json"
                },
                "analysis_method": "cryptanalytic_pleiotropy_detection",
                "target_genes": ["crp", "fis", "rpoS", "hns", "ihfA"],
                "trait_categories": [
                    "carbon_metabolism",
                    "stress_response", 
                    "regulatory",
                    "dna_processing",
                    "motility",
                    "biofilm_formation"
                ],
                "algorithm_settings": {
                    "window_size": 1000,
                    "overlap": 500,
                    "confidence_threshold": 0.7,
                    "frequency_analysis": "chi_squared",
                    "parallel_processing": True,
                    "threads": 5
                },
                "expected_results": {
                    "pleiotropic_genes": 5,
                    "trait_associations": ">=15",
                    "confidence_scores": ">=0.7"
                }
            }
            
            trial = Trial(
                name="E. coli K-12 Pleiotropic Analysis",
                description=(
                    "Comprehensive cryptanalytic analysis of pleiotropy in E. coli K-12 "
                    "focusing on CRP, FIS, RpoS, H-NS, and IHF-α regulatory genes. "
                    "Uses cryptographic frequency analysis to identify multi-trait "
                    "codon usage patterns."
                ),
                parameters=trial_params,
                hypothesis=(
                    "Pleiotropic regulatory genes (crp, fis, rpoS, hns, ihfA) will exhibit "
                    "distinct cryptographic signatures in their codon usage patterns that "
                    "correspond to different trait categories. Each trait will show "
                    "characteristic frequency distributions when genes are analyzed as "
                    "polyalphabetic ciphers with regulatory context as decryption keys."
                ),
                status=TrialStatus.RUNNING,
                created_by_agent=db_agent.id
            )
            
            session.add(trial)
            session.commit()
            
            trial_id = trial.id
            print(f"✓ Created trial: {trial.name} (ID: {trial_id})")
            
            return trial_id
            
        except Exception as e:
            session.rollback()
            print(f"✗ Failed to create trial: {e}")
            raise
        finally:
            session.close()
    
    def setup_swarm_agents(self) -> List[Dict]:
        """Set up agents for the swarm analysis with updated memory keys."""
        session = self.SessionLocal()
        
        try:
            # Define the 5 agents for this analysis
            swarm_agents = [
                {
                    "name": "genome-analyzer-swarm-001",
                    "type": AgentType.GENOME_ANALYST,
                    "status": AgentStatus.ACTIVE,
                    "memory_key": f"{self.memory_namespace}/genome-analyzer",
                    "tasks": [
                        "Parse FASTA sequences",
                        "Extract codon frequencies",
                        "Identify ORFs and regulatory regions"
                    ]
                },
                {
                    "name": "crypto-specialist-swarm-001", 
                    "type": AgentType.CRYPTO_SPECIALIST,
                    "status": AgentStatus.ACTIVE,
                    "memory_key": f"{self.memory_namespace}/crypto-specialist",
                    "tasks": [
                        "Apply frequency analysis algorithms",
                        "Detect polyalphabetic patterns",
                        "Calculate chi-squared statistics"
                    ]
                },
                {
                    "name": "trait-extractor-swarm-001",
                    "type": AgentType.GENOME_ANALYST,
                    "status": AgentStatus.ACTIVE, 
                    "memory_key": f"{self.memory_namespace}/trait-extractor",
                    "tasks": [
                        "Map genes to trait categories",
                        "Extract trait-specific signatures",
                        "Validate against known associations"
                    ]
                },
                {
                    "name": "visualizer-swarm-001",
                    "type": AgentType.VISUALIZATION_ENGINEER,
                    "status": AgentStatus.IDLE,
                    "memory_key": f"{self.memory_namespace}/visualizer",
                    "tasks": [
                        "Generate codon heatmaps",
                        "Create trait association networks",
                        "Produce confidence score plots"
                    ]
                },
                {
                    "name": "database-manager-swarm-001",
                    "type": AgentType.DATABASE_ARCHITECT,
                    "status": AgentStatus.ACTIVE,
                    "memory_key": f"{self.memory_namespace}/database-manager",
                    "tasks": [
                        "Record trial progress",
                        "Store intermediate results", 
                        "Maintain real-time statistics"
                    ]
                }
            ]
            
            agents_created = []
            
            for agent_data in swarm_agents:
                # Check if agent already exists
                existing = session.query(Agent).filter(
                    Agent.name == agent_data["name"]
                ).first()
                
                if existing:
                    # Update existing agent
                    existing.status = agent_data["status"]
                    existing.memory_keys = [agent_data["memory_key"]]
                    existing.updated_at = datetime.utcnow()
                    agent = existing
                else:
                    # Create new agent
                    agent = Agent(
                        name=agent_data["name"],
                        type=agent_data["type"],
                        status=agent_data["status"],
                        memory_keys=[agent_data["memory_key"]],
                        tasks_completed=0
                    )
                    session.add(agent)
                
                agents_created.append({
                    "id": agent.id if existing else None,
                    "name": agent_data["name"],
                    "type": agent_data["type"],
                    "memory_key": agent_data["memory_key"],
                    "tasks": agent_data["tasks"]
                })
            
            session.commit()
            
            print(f"✓ Configured {len(agents_created)} swarm agents")
            for agent in agents_created:
                print(f"  - {agent['name']}: {agent['type']}")
            
            return agents_created
            
        except Exception as e:
            session.rollback()
            print(f"✗ Failed to setup agents: {e}")
            raise
        finally:
            session.close()
    
    def initialize_progress_tracking(self, trial_id: int, agents: List[Dict]) -> None:
        """Initialize progress tracking for all agents."""
        session = self.SessionLocal()
        
        try:
            progress_entries = []
            
            for agent_data in agents:
                # Get actual agent from database
                agent = session.query(Agent).filter(
                    Agent.name == agent_data["name"]
                ).first()
                
                if not agent:
                    continue
                
                for i, task in enumerate(agent_data["tasks"]):
                    progress = Progress(
                        agent_id=agent.id,
                        task_id=f"trial_{trial_id}_{agent.name}_task_{i+1}",
                        status=ProgressStatus.STARTED,
                        message=f"Initialized: {task}",
                        percentage=0
                    )
                    progress_entries.append(progress)
                    session.add(progress)
            
            session.commit()
            
            print(f"✓ Initialized {len(progress_entries)} progress tracking entries")
            
        except Exception as e:
            session.rollback()
            print(f"✗ Failed to initialize progress tracking: {e}")
            raise
        finally:
            session.close()
    
    def record_memory_operation(self, operation: str, data: Dict) -> None:
        """Record a memory operation for the trial."""
        timestamp = datetime.utcnow().isoformat()
        memory_key = f"{self.memory_namespace}/database-manager/{operation}"
        
        operation_record = {
            "timestamp": timestamp,
            "operation": operation,
            "data": data,
            "memory_key": memory_key
        }
        
        # Save to file system as well for redundancy
        memory_dir = Path(f"trial_database/memory/{self.memory_namespace}")
        memory_dir.mkdir(parents=True, exist_ok=True)
        
        operation_file = memory_dir / f"{operation}_{int(time.time())}.json"
        with open(operation_file, 'w') as f:
            json.dump(operation_record, f, indent=2)
        
        print(f"💾 Recorded memory operation: {memory_key}")
    
    def get_trial_statistics(self, trial_id: int) -> Dict[str, Any]:
        """Get real-time trial statistics."""
        session = self.SessionLocal()
        
        try:
            trial = session.query(Trial).get(trial_id)
            if not trial:
                return {}
            
            # Agent status counts
            agent_stats = session.execute(text("""
                SELECT a.status, COUNT(*) as count
                FROM agents a
                WHERE a.name LIKE '%swarm%'
                GROUP BY a.status
            """)).fetchall()
            
            # Progress summary
            progress_stats = session.execute(text(f"""
                SELECT p.status, AVG(p.percentage) as avg_percentage, COUNT(*) as count
                FROM progress p
                JOIN agents a ON p.agent_id = a.id
                WHERE a.name LIKE '%swarm%'
                GROUP BY p.status
            """)).fetchall()
            
            # Results count
            results_count = session.query(Result).filter(
                Result.trial_id == trial_id
            ).count()
            
            stats = {
                "trial_id": trial_id,
                "trial_name": trial.name,
                "trial_status": trial.status,
                "agents": {row[0]: row[1] for row in agent_stats},
                "progress": {
                    row[0]: {"avg_percentage": row[1], "count": row[2]} 
                    for row in progress_stats
                },
                "results_recorded": results_count,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            return stats
            
        finally:
            session.close()
    
    def update_agent_progress(self, agent_name: str, task_id: str, 
                            percentage: int, message: str) -> None:
        """Update progress for a specific agent task."""
        session = self.SessionLocal()
        
        try:
            # Find the agent
            agent = session.query(Agent).filter(
                Agent.name == agent_name
            ).first()
            
            if not agent:
                print(f"⚠️  Agent not found: {agent_name}")
                return
            
            # Find or create progress entry
            progress = session.query(Progress).filter(
                Progress.agent_id == agent.id,
                Progress.task_id == task_id
            ).first()
            
            if not progress:
                progress = Progress(
                    agent_id=agent.id,
                    task_id=task_id,
                    status=ProgressStatus.IN_PROGRESS,
                    message=message,
                    percentage=percentage
                )
                session.add(progress)
            else:
                progress.percentage = percentage
                progress.message = message
                progress.status = (
                    ProgressStatus.COMPLETED if percentage >= 100 
                    else ProgressStatus.IN_PROGRESS
                )
                progress.timestamp = datetime.utcnow()
            
            # Update agent heartbeat
            agent.update_heartbeat()
            
            session.commit()
            
            print(f"📊 Updated {agent_name}: {percentage}% - {message}")
            
        except Exception as e:
            session.rollback()
            print(f"✗ Failed to update progress: {e}")
        finally:
            session.close()
    
    def store_analysis_result(self, trial_id: int, agent_name: str, 
                            metrics: Dict, confidence: float,
                            visualizations: Dict = None) -> None:
        """Store analysis results from an agent."""
        session = self.SessionLocal()
        
        try:
            agent = session.query(Agent).filter(
                Agent.name == agent_name
            ).first()
            
            if not agent:
                print(f"⚠️  Agent not found: {agent_name}")
                return
            
            result = Result(
                trial_id=trial_id,
                metrics=metrics,
                confidence_score=confidence,
                visualizations=visualizations or {},
                agent_id=agent.id
            )
            
            session.add(result)
            agent.increment_tasks()
            session.commit()
            
            print(f"💾 Stored result from {agent_name} (confidence: {confidence:.3f})")
            
        except Exception as e:
            session.rollback()
            print(f"✗ Failed to store result: {e}")
        finally:
            session.close()
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for the real-time dashboard."""
        session = self.SessionLocal()
        
        try:
            # Get current trial info
            current_trial = session.query(Trial).filter(
                Trial.status == TrialStatus.RUNNING
            ).order_by(Trial.created_at.desc()).first()
            
            if not current_trial:
                return {"error": "No active trial found"}
            
            # Get agent statuses
            agents = session.query(Agent).filter(
                Agent.name.like('%swarm%')
            ).all()
            
            # Get recent progress
            recent_progress = session.query(Progress).join(Agent).filter(
                Agent.name.like('%swarm%')
            ).order_by(Progress.timestamp.desc()).limit(10).all()
            
            # Get results summary
            results = session.query(Result).filter(
                Result.trial_id == current_trial.id
            ).all()
            
            dashboard = {
                "trial": {
                    "id": current_trial.id,
                    "name": current_trial.name,
                    "status": current_trial.status,
                    "started": current_trial.created_at.isoformat(),
                    "parameters": current_trial.parameters
                },
                "agents": [
                    {
                        "name": agent.name,
                        "type": agent.type,
                        "status": agent.status,
                        "tasks_completed": agent.tasks_completed,
                        "last_heartbeat": agent.last_heartbeat.isoformat()
                    }
                    for agent in agents
                ],
                "progress": [
                    {
                        "agent": prog.agent.name,
                        "task": prog.task_id,
                        "status": prog.status,
                        "percentage": prog.percentage,
                        "message": prog.message,
                        "timestamp": prog.timestamp.isoformat()
                    }
                    for prog in recent_progress
                ],
                "results": [
                    {
                        "agent": result.agent.name,
                        "confidence": result.confidence_score,
                        "metrics": result.metrics,
                        "timestamp": result.timestamp.isoformat()
                    }
                    for result in results
                ],
                "statistics": self.get_trial_statistics(current_trial.id),
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return dashboard
            
        finally:
            session.close()


def main():
    """Initialize the trial database for E. coli pleiotropic analysis."""
    print("🧬 TRIAL DATABASE MANAGER - Genomic Pleiotropy Cryptanalysis")
    print("=" * 70)
    print(f"Memory Namespace: swarm-pleiotropy-analysis-1752302124")
    print()
    
    manager = PleiotropyTrialManager()
    
    # 1. Create the main trial
    print("1. Creating E. coli K-12 Pleiotropic Analysis Trial...")
    trial_id = manager.create_ecoli_trial()
    
    # Record trial creation
    manager.record_memory_operation("trial_created", {
        "trial_id": trial_id,
        "trial_name": "E. coli K-12 Pleiotropic Analysis",
        "genes": ["crp", "fis", "rpoS", "hns", "ihfA"],
        "traits": 6,
        "confidence_threshold": 0.7
    })
    
    # 2. Setup swarm agents
    print("\n2. Setting up Swarm Agents...")
    agents = manager.setup_swarm_agents()
    
    # Record agent setup
    manager.record_memory_operation("agents_configured", {
        "agent_count": len(agents),
        "agents": [{"name": a["name"], "type": a["type"]} for a in agents]
    })
    
    # 3. Initialize progress tracking
    print("\n3. Initializing Progress Tracking...")
    manager.initialize_progress_tracking(trial_id, agents)
    
    # Record progress initialization
    manager.record_memory_operation("progress_initialized", {
        "trial_id": trial_id,
        "tracking_entries": len(agents) * 3  # 3 tasks per agent
    })
    
    # 4. Generate initial dashboard data
    print("\n4. Generating Dashboard Data...")
    dashboard = manager.generate_dashboard_data()
    
    # Save dashboard snapshot
    dashboard_file = Path("trial_database/dashboard_snapshot.json")
    with open(dashboard_file, 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    print(f"📊 Dashboard data saved to: {dashboard_file}")
    
    # 5. Display trial summary
    print("\n" + "=" * 70)
    print("🎯 TRIAL INITIALIZED SUCCESSFULLY")
    print("=" * 70)
    print(f"Trial ID: {trial_id}")
    print(f"Trial Name: E. coli K-12 Pleiotropic Analysis")
    print(f"Agents: {len(agents)} configured")
    print(f"Target Genes: crp, fis, rpoS, hns, ihfA")
    print(f"Trait Categories: 6")
    print(f"Confidence Threshold: 0.7")
    print(f"Memory Namespace: swarm-pleiotropy-analysis-1752302124")
    print()
    print("✅ Database is ready for real-time analysis tracking!")
    print("✅ Agents are configured and awaiting task assignment!")
    print("✅ Progress tracking is active!")
    print("✅ Results storage is ready!")
    
    # Record final initialization
    manager.record_memory_operation("initialization_complete", {
        "trial_id": trial_id,
        "status": "ready",
        "agents_ready": len(agents),
        "next_steps": [
            "Agents begin analysis tasks",
            "Real-time progress updates",
            "Results collection and storage",
            "Dashboard monitoring"
        ]
    })
    
    return trial_id, agents, manager


if __name__ == "__main__":
    main()