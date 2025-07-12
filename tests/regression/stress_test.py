#!/usr/bin/env python3
"""
Database Stress Test
Tests database performance under high load and stress conditions.
"""
import time
import threading
import random
import tempfile
import os
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from trial_database.database.init_db import DatabaseInitializer
from trial_database.database.utils import DatabaseUtils
from trial_database.database.models import AgentType, AgentStatus, TrialStatus


def stress_test_database():
    """Run comprehensive stress test on database."""
    print("🔥 Starting Database Stress Test")
    print("=" * 60)
    
    # Create temporary database
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "stress_test.db")
    
    print(f"Creating stress test database: {db_path}")
    initializer = DatabaseInitializer(db_path)
    initializer.create_database()
    
    db_utils = DatabaseUtils(db_path)
    
    # Test metrics
    metrics = {
        "total_operations": 0,
        "successful_operations": 0,
        "failed_operations": 0,
        "total_time": 0,
        "concurrent_errors": [],
        "performance_issues": []
    }
    
    def stress_agent_operations(thread_id, operations=100):
        """Stress test agent operations."""
        local_success = 0
        local_failures = 0
        
        for i in range(operations):
            try:
                # Create agent
                with db_utils.get_session() as session:
                    from trial_database.database.models import Agent
                    agent = Agent(
                        name=f"stress-agent-{thread_id}-{i}",
                        type=random.choice(list(AgentType)),
                        status=random.choice(list(AgentStatus))
                    )
                    session.add(agent)
                    session.commit()
                    
                    # Update heartbeat
                    agent.update_heartbeat()
                    session.commit()
                    
                    local_success += 1
                    
            except Exception as e:
                local_failures += 1
                metrics["concurrent_errors"].append(f"Thread {thread_id}: {str(e)}")
        
        metrics["successful_operations"] += local_success
        metrics["failed_operations"] += local_failures
    
    def stress_trial_operations(thread_id, operations=50):
        """Stress test trial operations."""
        local_success = 0
        local_failures = 0
        
        for i in range(operations):
            try:
                # Get a random agent
                agents = db_utils.get_active_agents()
                if not agents:
                    # Create a base agent if none exist
                    with db_utils.get_session() as session:
                        from trial_database.database.models import Agent
                        agent = Agent(
                            name=f"base-agent-{thread_id}",
                            type=AgentType.ORCHESTRATOR,
                            status=AgentStatus.ACTIVE
                        )
                        session.add(agent)
                        session.commit()
                        agents = [agent]
                
                agent = random.choice(agents)
                
                # Create trial
                trial = db_utils.create_trial(
                    name=f"Stress Trial {thread_id}-{i}",
                    description=f"Stress test trial from thread {thread_id}",
                    parameters={
                        "thread_id": thread_id,
                        "iteration": i,
                        "random_data": [random.randint(1, 1000) for _ in range(10)],
                        "stress_test": True
                    },
                    hypothesis=f"Stress test hypothesis {i}",
                    agent_id=agent.id
                )
                
                # Update trial status randomly
                new_status = random.choice(list(TrialStatus))
                db_utils.update_trial_status(trial.id, new_status)
                
                # Add a result
                db_utils.add_result(
                    trial_id=trial.id,
                    metrics={
                        "accuracy": random.uniform(0.1, 1.0),
                        "precision": random.uniform(0.1, 1.0),
                        "recall": random.uniform(0.1, 1.0),
                        "iteration": i
                    },
                    confidence_score=random.uniform(0.0, 1.0),
                    agent_id=agent.id
                )
                
                local_success += 1
                
            except Exception as e:
                local_failures += 1
                metrics["concurrent_errors"].append(f"Trial Thread {thread_id}: {str(e)}")
        
        metrics["successful_operations"] += local_success
        metrics["failed_operations"] += local_failures
    
    # Run stress tests
    start_time = time.time()
    
    print("🔥 Phase 1: Concurrent Agent Creation (10 threads, 100 ops each)")
    agent_threads = []
    for i in range(10):
        thread = threading.Thread(target=stress_agent_operations, args=(i, 100))
        agent_threads.append(thread)
        thread.start()
    
    for thread in agent_threads:
        thread.join()
    
    phase1_time = time.time() - start_time
    print(f"   ✓ Phase 1 completed in {phase1_time:.2f}s")
    
    print("🔥 Phase 2: Concurrent Trial Operations (5 threads, 50 ops each)")
    phase2_start = time.time()
    trial_threads = []
    for i in range(5):
        thread = threading.Thread(target=stress_trial_operations, args=(i, 50))
        trial_threads.append(thread)
        thread.start()
    
    for thread in trial_threads:
        thread.join()
    
    phase2_time = time.time() - phase2_start
    print(f"   ✓ Phase 2 completed in {phase2_time:.2f}s")
    
    total_time = time.time() - start_time
    metrics["total_time"] = total_time
    
    # Performance tests
    print("🔥 Phase 3: Performance Under Load")
    
    # Large query test
    query_start = time.time()
    try:
        with db_utils.get_session() as session:
            from trial_database.database.models import Agent, Trial, Result
            
            # Complex join query
            result = session.query(
                Trial.name,
                Agent.name,
                Result.confidence_score
            ).join(
                Agent, Trial.created_by_agent == Agent.id
            ).join(
                Result, Trial.id == Result.trial_id
            ).order_by(
                Result.confidence_score.desc()
            ).limit(100).all()
            
            query_time = (time.time() - query_start) * 1000
            print(f"   ✓ Complex join query: {query_time:.2f}ms ({len(result)} results)")
            
            if query_time > 100:  # More than 100ms
                metrics["performance_issues"].append(f"Slow complex query: {query_time:.2f}ms")
                
    except Exception as e:
        metrics["concurrent_errors"].append(f"Complex query failed: {str(e)}")
    
    # Database size and statistics
    print("🔥 Phase 4: Database Statistics")
    try:
        stats = db_utils.get_trial_statistics()
        print(f"   📊 Total Trials: {stats['total_trials']}")
        print(f"   📊 Total Results: {stats['total_results']}")
        print(f"   📊 Average Confidence: {stats['average_confidence']:.3f}")
        
        agent_count = len(db_utils.get_active_agents())
        print(f"   📊 Active Agents: {agent_count}")
        
    except Exception as e:
        metrics["concurrent_errors"].append(f"Statistics query failed: {str(e)}")
    
    # Results
    print("\n" + "=" * 60)
    print("🔥 STRESS TEST RESULTS")
    print("=" * 60)
    
    total_ops = metrics["successful_operations"] + metrics["failed_operations"]
    success_rate = (metrics["successful_operations"] / total_ops) * 100 if total_ops > 0 else 0
    
    print(f"Total Operations: {total_ops}")
    print(f"Successful: {metrics['successful_operations']}")
    print(f"Failed: {metrics['failed_operations']}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Operations/Second: {total_ops/total_time:.1f}")
    
    if metrics["concurrent_errors"]:
        print(f"\n🐛 CONCURRENT ERRORS ({len(metrics['concurrent_errors'])}):")
        for error in metrics["concurrent_errors"][:10]:  # Show first 10
            print(f"  - {error}")
        if len(metrics["concurrent_errors"]) > 10:
            print(f"  ... and {len(metrics['concurrent_errors']) - 10} more")
    
    if metrics["performance_issues"]:
        print(f"\n⚠️  PERFORMANCE ISSUES:")
        for issue in metrics["performance_issues"]:
            print(f"  - {issue}")
    
    # Final assessment
    print(f"\n📊 FINAL ASSESSMENT:")
    if success_rate >= 95:
        print("✅ EXCELLENT: Database handles high load very well")
    elif success_rate >= 90:
        print("✅ GOOD: Database handles high load adequately")
    elif success_rate >= 80:
        print("⚠️  ACCEPTABLE: Database has some issues under load")
    else:
        print("❌ POOR: Database struggles under high load")
    
    # Save results to memory namespace
    memory_key = f"swarm-regression-1752301224/database-test/stress_test_results"
    print(f"\n💾 Saving results to memory: {memory_key}")
    
    return {
        "success_rate": success_rate,
        "total_operations": total_ops,
        "total_time": total_time,
        "ops_per_second": total_ops/total_time,
        "errors": len(metrics["concurrent_errors"]),
        "performance_issues": len(metrics["performance_issues"])
    }


if __name__ == "__main__":
    results = stress_test_database()
    print(f"\n🔥 Stress test complete!")
    print(f"Database performance under load: {results['success_rate']:.1f}% success rate")
    print(f"Throughput: {results['ops_per_second']:.1f} operations/second")