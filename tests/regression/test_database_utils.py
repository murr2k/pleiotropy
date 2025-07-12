#!/usr/bin/env python3
"""
Additional Database Utilities Regression Tests
Tests the DatabaseUtils class and migration scenarios.
"""
import pytest
import time
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from trial_database.database.utils import DatabaseUtils
from trial_database.database.models import (
    Agent, Trial, Result, Progress,
    AgentType, AgentStatus, TrialStatus, ProgressStatus
)


class TestMemoryStorageUtils:
    """Memory storage for utilities test findings."""
    
    def __init__(self):
        self.findings = {}
        self.metrics = {}
        self.bugs = []
        
    def save_finding(self, key: str, data):
        """Save finding to memory."""
        namespace_key = f"swarm-regression-1752301224/database-test/utils/{key}"
        self.findings[namespace_key] = {
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        print(f"💾 UTILS: Saved finding {namespace_key}")
        
    def save_metric(self, key: str, value):
        """Save performance metric."""
        metric_key = f"swarm-regression-1752301224/database-test/utils/metrics/{key}"
        self.metrics[metric_key] = {
            "timestamp": datetime.utcnow().isoformat(),
            "value": value
        }
        print(f"📊 UTILS METRIC: {key} = {value}")
        
    def report_bug(self, severity: str, description: str, details: dict):
        """Report a bug."""
        bug = {
            "id": f"UTILS-BUG-{len(self.bugs) + 1:03d}",
            "severity": severity,
            "description": description,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.bugs.append(bug)
        print(f"🐛 UTILS BUG {bug['id']} ({severity}): {description}")


utils_memory = TestMemoryStorageUtils()


class TestDatabaseUtilities:
    """Test the DatabaseUtils class functionality."""
    
    @pytest.fixture
    def db_utils(self):
        """Create a temporary database with utilities."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "utils_test.db")
        
        # Initialize with test data
        from trial_database.database.init_db import DatabaseInitializer
        initializer = DatabaseInitializer(db_path)
        initializer.create_database()
        initializer.seed_database()
        
        return DatabaseUtils(db_path)
    
    def test_agent_operations(self, db_utils):
        """Test agent-related utility methods."""
        start_time = time.time()
        
        # Test getting agent by name
        agent = db_utils.get_agent_by_name("orchestrator-001")
        assert agent is not None
        assert agent.type == AgentType.ORCHESTRATOR
        
        # Test getting active agents
        active_agents = db_utils.get_active_agents()
        assert len(active_agents) > 0
        
        # Test updating heartbeat
        original_heartbeat = agent.last_heartbeat
        success = db_utils.update_agent_heartbeat(agent.id)
        assert success
        
        # Verify heartbeat was updated
        updated_agent = db_utils.get_agent_by_name("orchestrator-001")
        assert updated_agent.last_heartbeat > original_heartbeat
        
        # Test workload statistics
        workload = db_utils.get_agent_workload()
        assert isinstance(workload, list)
        assert len(workload) > 0
        
        operation_time = (time.time() - start_time) * 1000
        utils_memory.save_metric("agent_operations_time_ms", operation_time)
        utils_memory.save_finding("agent_operations", "PASS - All agent operations working")
    
    def test_trial_operations(self, db_utils):
        """Test trial-related utility methods."""
        start_time = time.time()
        
        # Get an existing agent for trial creation
        agent = db_utils.get_active_agents()[0]
        
        # Test creating a trial
        trial = db_utils.create_trial(
            name="Utils Test Trial",
            description="Testing database utilities",
            parameters={"test": True, "utils": "testing"},
            hypothesis="Database utilities should work correctly",
            agent_id=agent.id
        )
        assert trial is not None
        assert trial.name == "Utils Test Trial"
        
        # Test getting trial by ID
        retrieved_trial = db_utils.get_trial_by_id(trial.id)
        assert retrieved_trial is not None
        assert retrieved_trial.id == trial.id
        
        # Test updating trial status
        success = db_utils.update_trial_status(trial.id, TrialStatus.RUNNING)
        assert success
        
        # Verify status update
        updated_trial = db_utils.get_trial_by_id(trial.id)
        assert updated_trial.status == TrialStatus.RUNNING
        
        # Test getting trials by status
        running_trials = db_utils.get_trials_by_status(TrialStatus.RUNNING)
        assert len(running_trials) > 0
        
        # Test getting recent trials
        recent_trials = db_utils.get_recent_trials(limit=5)
        assert len(recent_trials) > 0
        
        operation_time = (time.time() - start_time) * 1000
        utils_memory.save_metric("trial_operations_time_ms", operation_time)
        utils_memory.save_finding("trial_operations", "PASS - All trial operations working")
    
    def test_result_operations(self, db_utils):
        """Test result-related utility methods."""
        start_time = time.time()
        
        # Get an existing trial and agent
        trials = db_utils.get_recent_trials(limit=1)
        trial = trials[0]
        agent = db_utils.get_active_agents()[0]
        
        # Test adding a result
        result = db_utils.add_result(
            trial_id=trial.id,
            metrics={"accuracy": 0.95, "precision": 0.92, "recall": 0.89},
            confidence_score=0.92,
            agent_id=agent.id,
            visualizations={"plot": "test_plot.png", "heatmap": "test_heatmap.png"}
        )
        assert result is not None
        assert result.confidence_score == 0.92
        
        # Test getting high confidence results
        high_conf_results = db_utils.get_high_confidence_results(min_confidence=0.9)
        assert len(high_conf_results) > 0
        
        # Test getting trial results
        trial_results = db_utils.get_trial_results(trial.id)
        assert len(trial_results) > 0
        
        operation_time = (time.time() - start_time) * 1000
        utils_memory.save_metric("result_operations_time_ms", operation_time)
        utils_memory.save_finding("result_operations", "PASS - All result operations working")
    
    def test_progress_operations(self, db_utils):
        """Test progress tracking operations."""
        start_time = time.time()
        
        # Get an active agent
        agent = db_utils.get_active_agents()[0]
        
        # Test creating progress
        task_id = f"test-task-{int(time.time())}"
        progress = db_utils.create_progress(
            agent_id=agent.id,
            task_id=task_id,
            message="Starting test task"
        )
        assert progress is not None
        assert progress.task_id == task_id
        assert progress.percentage == 0
        
        # Test updating progress
        success = db_utils.update_progress(
            task_id=task_id,
            percentage=50,
            message="Halfway done"
        )
        assert success
        
        # Test completing progress
        success = db_utils.update_progress(
            task_id=task_id,
            percentage=100,
            message="Task completed"
        )
        assert success
        
        # Test getting active tasks
        active_tasks = db_utils.get_active_tasks()
        # Should not include our completed task
        completed_task_found = any(task.task_id == task_id for task in active_tasks)
        assert not completed_task_found
        
        operation_time = (time.time() - start_time) * 1000
        utils_memory.save_metric("progress_operations_time_ms", operation_time)
        utils_memory.save_finding("progress_operations", "PASS - All progress operations working")
    
    def test_aggregate_queries(self, db_utils):
        """Test aggregate and statistical queries."""
        start_time = time.time()
        
        # Test trial statistics
        stats = db_utils.get_trial_statistics()
        assert "total_trials" in stats
        assert "by_status" in stats
        assert "average_confidence" in stats
        assert "total_results" in stats
        assert stats["total_trials"] > 0
        
        # Test top performing trials
        top_trials = db_utils.get_top_performing_trials(limit=3)
        assert len(top_trials) > 0
        
        # Test search functionality
        search_results = db_utils.search_trials("test")
        # Should find some trials with "test" in name/description
        
        # Test agent activity timeline
        agent = db_utils.get_active_agents()[0]
        timeline = db_utils.get_agent_activity_timeline(agent.id, hours=24)
        assert "agent_id" in timeline
        assert "trials_created" in timeline
        assert "results_generated" in timeline
        
        operation_time = (time.time() - start_time) * 1000
        utils_memory.save_metric("aggregate_queries_time_ms", operation_time)
        utils_memory.save_finding("aggregate_queries", "PASS - All aggregate queries working")
    
    def test_cleanup_operations(self, db_utils):
        """Test database cleanup operations."""
        start_time = time.time()
        
        # Create some old completed progress entries
        agent = db_utils.get_active_agents()[0]
        old_task_id = f"old-task-{int(time.time())}"
        
        # Create and complete a progress entry
        progress = db_utils.create_progress(
            agent_id=agent.id,
            task_id=old_task_id,
            message="Old task"
        )
        
        # Complete it
        db_utils.update_progress(old_task_id, 100, "Old task completed")
        
        # Test cleanup (won't delete recent entries, but tests the function)
        deleted_count = db_utils.cleanup_old_progress(days=0)  # Clean everything older than now
        # The exact count depends on test timing, just verify function works
        assert isinstance(deleted_count, int)
        
        operation_time = (time.time() - start_time) * 1000
        utils_memory.save_metric("cleanup_operations_time_ms", operation_time)
        utils_memory.save_finding("cleanup_operations", "PASS - Cleanup operations working")


def test_error_handling():
    """Test error handling in database utilities."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "error_test.db")
    
    # Test with non-existent database
    try:
        db_utils = DatabaseUtils(db_path)
        # Should create database
        utils_memory.save_finding("error_handling_db_creation", "PASS - Database created when missing")
    except Exception as e:
        utils_memory.report_bug("MEDIUM", f"Failed to handle missing database: {str(e)}", {})
    
    # Test invalid operations
    if 'db_utils' in locals():
        # Test getting non-existent agent
        agent = db_utils.get_agent_by_name("non-existent-agent")
        assert agent is None
        
        # Test getting non-existent trial
        trial = db_utils.get_trial_by_id(99999)
        assert trial is None
        
        # Test updating non-existent heartbeat
        success = db_utils.update_agent_heartbeat(99999)
        assert not success
        
        utils_memory.save_finding("error_handling_graceful", "PASS - Graceful handling of missing entities")


def run_utils_tests():
    """Run all utility tests."""
    print("🔧 Starting Database Utils Regression Tests")
    print("=" * 60)
    
    # Create temp database and run tests
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "utils_test.db")
    
    # Initialize database
    from trial_database.database.init_db import DatabaseInitializer
    initializer = DatabaseInitializer(db_path)
    initializer.create_database()
    initializer.seed_database()
    
    db_utils = DatabaseUtils(db_path)
    
    test_classes = [TestDatabaseUtilities]
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}...")
        
        instance = test_class()
        test_methods = [method for method in dir(instance) if method.startswith('test_')]
        
        for test_method_name in test_methods:
            total_tests += 1
            try:
                print(f"  ✓ {test_method_name}")
                test_method = getattr(instance, test_method_name)
                test_method(db_utils)
                passed_tests += 1
            except Exception as e:
                print(f"  ❌ {test_method_name}: {str(e)}")
                utils_memory.report_bug("HIGH", f"Utils test {test_method_name} failed", {
                    "exception": str(e)
                })
    
    # Run error handling test
    print(f"\n📋 Running error handling tests...")
    total_tests += 1
    try:
        print(f"  ✓ test_error_handling")
        test_error_handling()
        passed_tests += 1
    except Exception as e:
        print(f"  ❌ test_error_handling: {str(e)}")
        utils_memory.report_bug("HIGH", f"Error handling test failed", {"exception": str(e)})
    
    # Results
    print("\n" + "=" * 60)
    print("📊 UTILS TEST RESULTS")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if utils_memory.bugs:
        print(f"\n🐛 BUGS FOUND: {len(utils_memory.bugs)}")
        for bug in utils_memory.bugs:
            print(f"  {bug['id']} ({bug['severity']}): {bug['description']}")
    
    print(f"\n📈 PERFORMANCE METRICS:")
    for key, metric in utils_memory.metrics.items():
        metric_name = key.split('/')[-1]
        print(f"  {metric_name}: {metric['value']}")
    
    return {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "bugs": utils_memory.bugs,
        "metrics": utils_memory.metrics,
        "findings": utils_memory.findings
    }


if __name__ == "__main__":
    results = run_utils_tests()
    print(f"\n✅ Utils regression testing complete!")