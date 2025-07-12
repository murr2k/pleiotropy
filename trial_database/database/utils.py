"""
Database utilities and common queries for the Genomic Pleiotropy Cryptanalysis trial database.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json

from sqlalchemy import create_engine, func, and_, or_, desc
from sqlalchemy.orm import sessionmaker, Session

from trial_database.database.models import (
    Agent, Trial, Result, Progress,
    AgentType, AgentStatus, TrialStatus, ProgressStatus
)


class DatabaseUtils:
    """Utility class for common database operations."""
    
    def __init__(self, db_path: str = "trial_database/database/trials.db"):
        """Initialize database utilities."""
        self.db_url = f"sqlite:///{db_path}"
        self.engine = create_engine(
            self.db_url,
            connect_args={"check_same_thread": False}
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    # Agent queries
    def get_agent_by_name(self, name: str) -> Optional[Agent]:
        """Get an agent by name."""
        with self.get_session() as session:
            return session.query(Agent).filter(Agent.name == name).first()
    
    def get_active_agents(self) -> List[Agent]:
        """Get all active agents."""
        with self.get_session() as session:
            return session.query(Agent).filter(Agent.status == AgentStatus.ACTIVE).all()
    
    def update_agent_heartbeat(self, agent_id: int) -> bool:
        """Update an agent's heartbeat."""
        with self.get_session() as session:
            agent = session.query(Agent).filter(Agent.id == agent_id).first()
            if agent:
                agent.update_heartbeat()
                session.commit()
                return True
            return False
    
    def get_agent_workload(self) -> List[Dict[str, Any]]:
        """Get workload statistics for all agents."""
        with self.get_session() as session:
            results = session.query(
                Agent.name,
                Agent.type,
                Agent.status,
                func.count(Trial.id).label('trials_created'),
                func.count(Result.id).label('results_generated')
            ).outerjoin(
                Trial, Agent.id == Trial.created_by_agent
            ).outerjoin(
                Result, Agent.id == Result.agent_id
            ).group_by(Agent.id).all()
            
            return [
                {
                    "name": r[0],
                    "type": r[1],
                    "status": r[2],
                    "trials_created": r[3],
                    "results_generated": r[4]
                }
                for r in results
            ]
    
    # Trial queries
    def create_trial(self, name: str, description: str, parameters: Dict[str, Any],
                    hypothesis: str, agent_id: int) -> Trial:
        """Create a new trial."""
        with self.get_session() as session:
            trial = Trial(
                name=name,
                description=description,
                parameters=parameters,
                hypothesis=hypothesis,
                created_by_agent=agent_id
            )
            session.add(trial)
            session.commit()
            session.refresh(trial)
            return trial
    
    def get_trial_by_id(self, trial_id: int) -> Optional[Trial]:
        """Get a trial by ID."""
        with self.get_session() as session:
            return session.query(Trial).filter(Trial.id == trial_id).first()
    
    def get_trials_by_status(self, status: TrialStatus) -> List[Trial]:
        """Get all trials with a specific status."""
        with self.get_session() as session:
            return session.query(Trial).filter(Trial.status == status).all()
    
    def update_trial_status(self, trial_id: int, status: TrialStatus) -> bool:
        """Update a trial's status."""
        with self.get_session() as session:
            trial = session.query(Trial).filter(Trial.id == trial_id).first()
            if trial:
                trial.set_status(status)
                session.commit()
                return True
            return False
    
    def get_recent_trials(self, limit: int = 10) -> List[Trial]:
        """Get the most recent trials."""
        with self.get_session() as session:
            return session.query(Trial).order_by(
                desc(Trial.created_at)
            ).limit(limit).all()
    
    # Result queries
    def add_result(self, trial_id: int, metrics: Dict[str, Any], 
                  confidence_score: float, agent_id: int,
                  visualizations: Dict[str, str] = None) -> Result:
        """Add a result to a trial."""
        with self.get_session() as session:
            result = Result(
                trial_id=trial_id,
                metrics=metrics,
                confidence_score=confidence_score,
                agent_id=agent_id,
                visualizations=visualizations or {}
            )
            session.add(result)
            session.commit()
            session.refresh(result)
            return result
    
    def get_high_confidence_results(self, min_confidence: float = 0.8) -> List[Result]:
        """Get results with high confidence scores."""
        with self.get_session() as session:
            return session.query(Result).filter(
                Result.confidence_score >= min_confidence
            ).order_by(desc(Result.confidence_score)).all()
    
    def get_trial_results(self, trial_id: int) -> List[Result]:
        """Get all results for a specific trial."""
        with self.get_session() as session:
            return session.query(Result).filter(
                Result.trial_id == trial_id
            ).order_by(Result.timestamp).all()
    
    # Progress queries
    def create_progress(self, agent_id: int, task_id: str, 
                       message: str = "") -> Progress:
        """Create a new progress entry."""
        with self.get_session() as session:
            progress = Progress(
                agent_id=agent_id,
                task_id=task_id,
                status=ProgressStatus.STARTED,
                message=message,
                percentage=0
            )
            session.add(progress)
            session.commit()
            session.refresh(progress)
            return progress
    
    def update_progress(self, task_id: str, percentage: int, 
                       message: str = None) -> bool:
        """Update progress for a task."""
        with self.get_session() as session:
            progress = session.query(Progress).filter(
                Progress.task_id == task_id
            ).order_by(desc(Progress.timestamp)).first()
            
            if progress:
                progress.update_progress(percentage, message)
                if percentage >= 100:
                    progress.status = ProgressStatus.COMPLETED
                elif progress.status == ProgressStatus.STARTED and percentage > 0:
                    progress.status = ProgressStatus.IN_PROGRESS
                session.commit()
                return True
            return False
    
    def get_active_tasks(self) -> List[Progress]:
        """Get all active tasks (not completed or failed)."""
        with self.get_session() as session:
            return session.query(Progress).filter(
                Progress.status.in_([
                    ProgressStatus.STARTED,
                    ProgressStatus.IN_PROGRESS
                ])
            ).order_by(Progress.timestamp).all()
    
    # Aggregate queries
    def get_trial_statistics(self) -> Dict[str, Any]:
        """Get overall trial statistics."""
        with self.get_session() as session:
            total_trials = session.query(func.count(Trial.id)).scalar()
            
            status_counts = session.query(
                Trial.status,
                func.count(Trial.id)
            ).group_by(Trial.status).all()
            
            avg_confidence = session.query(
                func.avg(Result.confidence_score)
            ).scalar() or 0
            
            return {
                "total_trials": total_trials,
                "by_status": {status: count for status, count in status_counts},
                "average_confidence": round(avg_confidence, 3),
                "total_results": session.query(func.count(Result.id)).scalar()
            }
    
    def get_top_performing_trials(self, limit: int = 5) -> List[Tuple[Trial, float]]:
        """Get trials with highest average confidence scores."""
        with self.get_session() as session:
            results = session.query(
                Trial,
                func.avg(Result.confidence_score).label('avg_confidence')
            ).join(
                Result, Trial.id == Result.trial_id
            ).group_by(Trial.id).order_by(
                desc('avg_confidence')
            ).limit(limit).all()
            
            return results
    
    def search_trials(self, keyword: str) -> List[Trial]:
        """Search trials by name or description."""
        with self.get_session() as session:
            pattern = f"%{keyword}%"
            return session.query(Trial).filter(
                or_(
                    Trial.name.like(pattern),
                    Trial.description.like(pattern),
                    Trial.hypothesis.like(pattern)
                )
            ).all()
    
    def get_agent_activity_timeline(self, agent_id: int, 
                                   hours: int = 24) -> Dict[str, Any]:
        """Get agent activity timeline for the last N hours."""
        with self.get_session() as session:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Get trials created
            trials = session.query(Trial).filter(
                and_(
                    Trial.created_by_agent == agent_id,
                    Trial.created_at >= cutoff_time
                )
            ).all()
            
            # Get results generated
            results = session.query(Result).filter(
                and_(
                    Result.agent_id == agent_id,
                    Result.timestamp >= cutoff_time
                )
            ).all()
            
            # Get progress updates
            progress = session.query(Progress).filter(
                and_(
                    Progress.agent_id == agent_id,
                    Progress.timestamp >= cutoff_time
                )
            ).all()
            
            return {
                "agent_id": agent_id,
                "period_hours": hours,
                "trials_created": len(trials),
                "results_generated": len(results),
                "progress_updates": len(progress),
                "timeline": {
                    "trials": [{"id": t.id, "name": t.name, "time": t.created_at} 
                              for t in trials],
                    "results": [{"id": r.id, "trial_id": r.trial_id, "time": r.timestamp} 
                               for r in results],
                    "progress": [{"task": p.task_id, "status": p.status, "time": p.timestamp} 
                                for p in progress]
                }
            }
    
    def cleanup_old_progress(self, days: int = 7) -> int:
        """Clean up old completed progress entries."""
        with self.get_session() as session:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            
            deleted = session.query(Progress).filter(
                and_(
                    Progress.status == ProgressStatus.COMPLETED,
                    Progress.timestamp < cutoff_time
                )
            ).delete()
            
            session.commit()
            return deleted


# Example usage functions
def example_queries():
    """Example usage of the database utilities."""
    db = DatabaseUtils()
    
    # Get active agents
    print("Active Agents:")
    for agent in db.get_active_agents():
        print(f"  - {agent.name} ({agent.type})")
    
    # Get trial statistics
    print("\nTrial Statistics:")
    stats = db.get_trial_statistics()
    print(f"  Total trials: {stats['total_trials']}")
    print(f"  Average confidence: {stats['average_confidence']}")
    
    # Get top performing trials
    print("\nTop Performing Trials:")
    for trial, avg_conf in db.get_top_performing_trials():
        print(f"  - {trial.name}: {avg_conf:.3f}")
    
    # Get agent workload
    print("\nAgent Workload:")
    for workload in db.get_agent_workload():
        print(f"  - {workload['name']}: {workload['trials_created']} trials, "
              f"{workload['results_generated']} results")


if __name__ == "__main__":
    example_queries()