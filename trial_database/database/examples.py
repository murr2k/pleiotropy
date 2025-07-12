#!/usr/bin/env python3
"""
Example queries and operations for the Trial Database.
Demonstrates common use cases and best practices.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from trial_database.database.utils import DatabaseUtils
from trial_database.database.models import (
    AgentType, AgentStatus, TrialStatus, ProgressStatus
)


def example_agent_operations():
    """Demonstrate agent-related operations."""
    print("\n=== Agent Operations ===")
    db = DatabaseUtils()
    
    # Get all active agents
    print("\n1. Active Agents:")
    active_agents = db.get_active_agents()
    for agent in active_agents:
        print(f"   - {agent.name} ({agent.type.value})")
        print(f"     Last seen: {agent.last_heartbeat}")
        print(f"     Tasks completed: {agent.tasks_completed}")
    
    # Update agent heartbeat
    if active_agents:
        agent = active_agents[0]
        print(f"\n2. Updating heartbeat for {agent.name}")
        db.update_agent_heartbeat(agent.id)
        print("   ✓ Heartbeat updated")
    
    # Get agent workload
    print("\n3. Agent Workload Report:")
    workloads = db.get_agent_workload()
    for w in workloads:
        print(f"   - {w['name']}:")
        print(f"     Trials created: {w['trials_created']}")
        print(f"     Results generated: {w['results_generated']}")


def example_trial_operations():
    """Demonstrate trial-related operations."""
    print("\n=== Trial Operations ===")
    db = DatabaseUtils()
    
    # Get recent trials
    print("\n1. Recent Trials:")
    recent_trials = db.get_recent_trials(limit=5)
    for trial in recent_trials:
        print(f"   - [{trial.id}] {trial.name}")
        print(f"     Status: {trial.status}")
        print(f"     Created: {trial.created_at}")
    
    # Search trials
    print("\n2. Searching for 'codon' in trials:")
    search_results = db.search_trials("codon")
    for trial in search_results:
        print(f"   - {trial.name}")
    
    # Get trial statistics
    print("\n3. Trial Statistics:")
    stats = db.get_trial_statistics()
    print(f"   Total trials: {stats['total_trials']}")
    print(f"   By status:")
    for status, count in stats['by_status'].items():
        print(f"     - {status}: {count}")
    print(f"   Average confidence: {stats['average_confidence']}")
    print(f"   Total results: {stats['total_results']}")


def example_result_operations():
    """Demonstrate result-related operations."""
    print("\n=== Result Operations ===")
    db = DatabaseUtils()
    
    # Get high confidence results
    print("\n1. High Confidence Results (>= 0.8):")
    high_conf = db.get_high_confidence_results(min_confidence=0.8)
    for result in high_conf[:5]:  # Limit to 5
        print(f"   - Result {result.id}: confidence = {result.confidence_score}")
        print(f"     Trial: {result.trial.name if result.trial else 'N/A'}")
        if result.metrics:
            print(f"     Metrics: {json.dumps(result.metrics, indent=6)}")
    
    # Get top performing trials
    print("\n2. Top Performing Trials:")
    top_trials = db.get_top_performing_trials(limit=3)
    for trial, avg_conf in top_trials:
        print(f"   - {trial.name}")
        print(f"     Average confidence: {avg_conf:.3f}")
        print(f"     Parameters: {json.dumps(trial.parameters, indent=6)}")


def example_progress_operations():
    """Demonstrate progress tracking operations."""
    print("\n=== Progress Operations ===")
    db = DatabaseUtils()
    
    # Get active tasks
    print("\n1. Active Tasks:")
    active_tasks = db.get_active_tasks()
    for progress in active_tasks:
        print(f"   - Task: {progress.task_id}")
        print(f"     Agent: {progress.agent.name if progress.agent else 'N/A'}")
        print(f"     Status: {progress.status}")
        print(f"     Progress: {progress.percentage}%")
        print(f"     Message: {progress.message}")
    
    # Simulate progress update
    if active_tasks:
        task = active_tasks[0]
        print(f"\n2. Updating progress for task: {task.task_id}")
        new_percentage = min(task.percentage + 10, 100)
        db.update_progress(
            task.task_id, 
            new_percentage,
            f"Progress update: now at {new_percentage}%"
        )
        print(f"   ✓ Updated to {new_percentage}%")


def example_complex_queries():
    """Demonstrate more complex query patterns."""
    print("\n=== Complex Query Examples ===")
    db = DatabaseUtils()
    
    # Get agent activity timeline
    print("\n1. Agent Activity Timeline (last 24 hours):")
    active_agents = db.get_active_agents()
    if active_agents:
        agent = active_agents[0]
        timeline = db.get_agent_activity_timeline(agent.id, hours=24)
        print(f"   Agent: {agent.name}")
        print(f"   Period: Last {timeline['period_hours']} hours")
        print(f"   - Trials created: {timeline['trials_created']}")
        print(f"   - Results generated: {timeline['results_generated']}")
        print(f"   - Progress updates: {timeline['progress_updates']}")
    
    # Custom query example using session
    print("\n2. Custom Query - Trials with Multiple Results:")
    with db.get_session() as session:
        from sqlalchemy import func
        
        # Find trials with more than one result
        subquery = session.query(
            Result.trial_id,
            func.count(Result.id).label('result_count')
        ).group_by(Result.trial_id).subquery()
        
        trials_with_multiple_results = session.query(Trial).join(
            subquery, Trial.id == subquery.c.trial_id
        ).filter(subquery.c.result_count > 1).all()
        
        for trial in trials_with_multiple_results:
            result_count = session.query(Result).filter(
                Result.trial_id == trial.id
            ).count()
            print(f"   - {trial.name}: {result_count} results")


def example_data_export():
    """Demonstrate exporting data for analysis."""
    print("\n=== Data Export Example ===")
    db = DatabaseUtils()
    
    # Export trial data as JSON
    print("\n1. Exporting completed trials as JSON:")
    completed_trials = db.get_trials_by_status(TrialStatus.COMPLETED)
    
    export_data = []
    for trial in completed_trials:
        trial_data = {
            "id": trial.id,
            "name": trial.name,
            "description": trial.description,
            "hypothesis": trial.hypothesis,
            "parameters": trial.parameters,
            "created_at": trial.created_at.isoformat() if trial.created_at else None,
            "results": []
        }
        
        # Get results for this trial
        results = db.get_trial_results(trial.id)
        for result in results:
            trial_data["results"].append({
                "metrics": result.metrics,
                "confidence_score": result.confidence_score,
                "timestamp": result.timestamp.isoformat() if result.timestamp else None
            })
        
        export_data.append(trial_data)
    
    # Pretty print a sample
    if export_data:
        print(json.dumps(export_data[0], indent=2))
        print(f"\n   Total trials exported: {len(export_data)}")


def main():
    """Run all examples."""
    print("Trial Database Examples")
    print("=" * 50)
    
    try:
        example_agent_operations()
        example_trial_operations()
        example_result_operations()
        example_progress_operations()
        example_complex_queries()
        example_data_export()
        
        print("\n" + "=" * 50)
        print("Examples completed successfully!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure the database is initialized with seed data:")
        print("  python trial_database/database/init_db.py --seed")


if __name__ == "__main__":
    main()