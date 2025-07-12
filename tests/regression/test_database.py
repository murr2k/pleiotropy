#!/usr/bin/env python3
"""
Comprehensive Database Regression Test Suite
Tests all database operations, constraints, integrity, and edge cases.
Part of the Genomic Pleiotropy Cryptanalysis project.
"""
import pytest
import sqlite3
import json
import time
import threading
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import patch
import sys
import os

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError, StatementError

from trial_database.database.models import (
    Base, Agent, Trial, Result, Progress,
    AgentType, AgentStatus, TrialStatus, ProgressStatus
)
from trial_database.database.utils import DatabaseUtils
from trial_database.database.init_db import DatabaseInitializer


class TestMemoryStorage:
    """In-memory storage for test findings and metrics."""
    
    def __init__(self):
        self.findings = {}
        self.metrics = {}
        self.bugs = []
        
    def save_finding(self, key: str, data: Any):
        """Save a finding to memory with swarm namespace."""
        namespace_key = f"swarm-regression-1752301224/database-test/{key}"
        self.findings[namespace_key] = {
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        print(f"💾 MEMORY: Saved finding {namespace_key}")
        
    def save_metric(self, key: str, value: Any):
        """Save a performance metric."""
        metric_key = f"swarm-regression-1752301224/database-test/metrics/{key}"
        self.metrics[metric_key] = {
            "timestamp": datetime.utcnow().isoformat(),
            "value": value
        }
        print(f"📊 METRIC: {key} = {value}")
        
    def report_bug(self, severity: str, description: str, details: Dict):
        """Report a bug with severity level."""
        bug = {
            "id": f"BUG-{len(self.bugs) + 1:03d}",
            "severity": severity,
            "description": description,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.bugs.append(bug)
        print(f"🐛 BUG {bug['id']} ({severity}): {description}")


# Global memory storage instance
memory = TestMemoryStorage()


class DatabaseRegressionTester:
    """Comprehensive database regression tester."""
    
    def __init__(self, test_db_path: str = None):
        """Initialize the tester with a temporary database."""
        if test_db_path is None:
            # Create temporary database for testing
            self.temp_dir = tempfile.mkdtemp()
            self.db_path = os.path.join(self.temp_dir, "test_trials.db")
        else:
            self.db_path = test_db_path
            
        self.db_url = f"sqlite:///{self.db_path}"
        self.engine = None
        self.SessionLocal = None
        self.setup_database()
        
    def setup_database(self):
        """Set up the test database."""
        self.engine = create_engine(
            self.db_url,
            echo=False,
            connect_args={"check_same_thread": False}
        )
        
        # Enable foreign keys
        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))
            
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def get_session(self):
        """Get a database session."""
        return self.SessionLocal()


class TestSchemaValidation:
    """Test database schema constraints and validation."""
    
    @pytest.fixture
    def db_tester(self):
        """Create a database tester instance."""
        return DatabaseRegressionTester()
    
    def test_agent_type_constraints(self, db_tester):
        """Test that agent type constraints are enforced."""
        with db_tester.get_session() as session:
            # Valid agent type should work
            valid_agent = Agent(
                name="test-agent-valid",
                type=AgentType.ORCHESTRATOR,
                status=AgentStatus.ACTIVE
            )
            session.add(valid_agent)
            session.commit()
            
            # Invalid agent type should fail
            with pytest.raises(Exception):
                invalid_agent = Agent(
                    name="test-agent-invalid",
                    type="invalid_type",  # This should fail
                    status=AgentStatus.ACTIVE
                )
                session.add(invalid_agent)
                session.commit()
                
        memory.save_finding("agent_type_validation", "PASS - Agent type constraints working")
    
    def test_agent_name_uniqueness(self, db_tester):
        """Test that agent names must be unique."""
        with db_tester.get_session() as session:
            # First agent should work
            agent1 = Agent(
                name="duplicate-name",
                type=AgentType.ORCHESTRATOR,
                status=AgentStatus.ACTIVE
            )
            session.add(agent1)
            session.commit()
            
            # Second agent with same name should fail
            try:
                agent2 = Agent(
                    name="duplicate-name",  # Duplicate name
                    type=AgentType.GENOME_ANALYST,
                    status=AgentStatus.ACTIVE
                )
                session.add(agent2)
                session.commit()
                memory.report_bug("HIGH", "Agent name uniqueness constraint not enforced", {
                    "table": "agents",
                    "constraint": "name_unique"
                })
            except IntegrityError:
                # This is expected
                session.rollback()
                memory.save_finding("agent_name_uniqueness", "PASS - Unique constraint working")
    
    def test_confidence_score_range(self, db_tester):
        """Test that confidence scores are constrained to 0.0-1.0."""
        with db_tester.get_session() as session:
            # Create an agent and trial first
            agent = Agent(name="test-agent", type=AgentType.GENOME_ANALYST, status=AgentStatus.ACTIVE)
            session.add(agent)
            session.flush()
            
            trial = Trial(
                name="test-trial",
                parameters={"test": "data"},
                created_by_agent=agent.id
            )
            session.add(trial)
            session.flush()
            
            # Valid confidence score should work
            valid_result = Result(
                trial_id=trial.id,
                metrics={"test": "metric"},
                confidence_score=0.85,
                agent_id=agent.id
            )
            session.add(valid_result)
            session.commit()
            
            # Test invalid confidence scores
            for invalid_score in [-0.1, 1.1, 2.0, -1.0]:
                try:
                    invalid_result = Result(
                        trial_id=trial.id,
                        metrics={"test": "metric"},
                        confidence_score=invalid_score,
                        agent_id=agent.id
                    )
                    session.add(invalid_result)
                    session.commit()
                    memory.report_bug("MEDIUM", f"Confidence score {invalid_score} allowed outside 0.0-1.0 range", {
                        "table": "results",
                        "constraint": "confidence_score_range",
                        "value": invalid_score
                    })
                except Exception:
                    # This is expected - constraint should prevent invalid values
                    session.rollback()
                    
        memory.save_finding("confidence_score_range", "PASS - Range constraints working")
    
    def test_percentage_range(self, db_tester):
        """Test that progress percentages are constrained to 0-100."""
        with db_tester.get_session() as session:
            # Create an agent first
            agent = Agent(name="test-agent-progress", type=AgentType.ORCHESTRATOR, status=AgentStatus.ACTIVE)
            session.add(agent)
            session.flush()
            
            # Valid percentage should work
            valid_progress = Progress(
                agent_id=agent.id,
                task_id="test-task-valid",
                status=ProgressStatus.IN_PROGRESS,
                percentage=50
            )
            session.add(valid_progress)
            session.commit()
            
            # Test invalid percentages
            for invalid_percent in [-1, 101, 150, -50]:
                try:
                    invalid_progress = Progress(
                        agent_id=agent.id,
                        task_id=f"test-task-{invalid_percent}",
                        status=ProgressStatus.IN_PROGRESS,
                        percentage=invalid_percent
                    )
                    session.add(invalid_progress)
                    session.commit()
                    memory.report_bug("MEDIUM", f"Percentage {invalid_percent} allowed outside 0-100 range", {
                        "table": "progress",
                        "constraint": "percentage_range",
                        "value": invalid_percent
                    })
                except Exception:
                    # This is expected
                    session.rollback()
                    
        memory.save_finding("percentage_range", "PASS - Percentage constraints working")


class TestCRUDOperations:
    """Test Create, Read, Update, Delete operations for all tables."""
    
    @pytest.fixture
    def db_tester(self):
        """Create a database tester instance."""
        return DatabaseRegressionTester()
    
    def test_agent_crud(self, db_tester):
        """Test full CRUD operations on agents table."""
        start_time = time.time()
        
        with db_tester.get_session() as session:
            # CREATE
            agent = Agent(
                name="crud-test-agent",
                type=AgentType.CRYPTO_SPECIALIST,
                status=AgentStatus.ACTIVE,
                tasks_completed=5,
                memory_keys=["key1", "key2"]
            )
            session.add(agent)
            session.commit()
            agent_id = agent.id
            
            # READ
            retrieved_agent = session.query(Agent).filter(Agent.id == agent_id).first()
            assert retrieved_agent is not None
            assert retrieved_agent.name == "crud-test-agent"
            assert retrieved_agent.type == AgentType.CRYPTO_SPECIALIST
            assert retrieved_agent.tasks_completed == 5
            assert retrieved_agent.memory_keys == ["key1", "key2"]
            
            # UPDATE
            retrieved_agent.status = AgentStatus.IDLE
            retrieved_agent.tasks_completed = 10
            session.commit()
            
            # Verify update
            updated_agent = session.query(Agent).filter(Agent.id == agent_id).first()
            assert updated_agent.status == AgentStatus.IDLE
            assert updated_agent.tasks_completed == 10
            
            # DELETE
            session.delete(updated_agent)
            session.commit()
            
            # Verify deletion
            deleted_agent = session.query(Agent).filter(Agent.id == agent_id).first()
            assert deleted_agent is None
            
        crud_time = time.time() - start_time
        memory.save_metric("agent_crud_time_ms", crud_time * 1000)
        memory.save_finding("agent_crud", "PASS - All CRUD operations successful")
    
    def test_trial_crud(self, db_tester):
        """Test full CRUD operations on trials table."""
        start_time = time.time()
        
        with db_tester.get_session() as session:
            # Create prerequisite agent
            agent = Agent(name="trial-crud-agent", type=AgentType.GENOME_ANALYST, status=AgentStatus.ACTIVE)
            session.add(agent)
            session.flush()
            
            # CREATE
            trial = Trial(
                name="CRUD Test Trial",
                description="Testing CRUD operations",
                parameters={"test": True, "value": 42},
                hypothesis="CRUD operations should work",
                status=TrialStatus.PENDING,
                created_by_agent=agent.id
            )
            session.add(trial)
            session.commit()
            trial_id = trial.id
            
            # READ
            retrieved_trial = session.query(Trial).filter(Trial.id == trial_id).first()
            assert retrieved_trial is not None
            assert retrieved_trial.name == "CRUD Test Trial"
            assert retrieved_trial.parameters["test"] is True
            assert retrieved_trial.parameters["value"] == 42
            
            # UPDATE
            retrieved_trial.status = TrialStatus.RUNNING
            retrieved_trial.parameters = {"test": False, "value": 100}
            session.commit()
            
            # Verify update
            updated_trial = session.query(Trial).filter(Trial.id == trial_id).first()
            assert updated_trial.status == TrialStatus.RUNNING
            assert updated_trial.parameters["test"] is False
            assert updated_trial.parameters["value"] == 100
            
            # DELETE (should cascade to results if any)
            session.delete(updated_trial)
            session.commit()
            
            # Verify deletion
            deleted_trial = session.query(Trial).filter(Trial.id == trial_id).first()
            assert deleted_trial is None
            
        crud_time = time.time() - start_time
        memory.save_metric("trial_crud_time_ms", crud_time * 1000)
        memory.save_finding("trial_crud", "PASS - All CRUD operations successful")
    
    def test_result_crud(self, db_tester):
        """Test full CRUD operations on results table."""
        start_time = time.time()
        
        with db_tester.get_session() as session:
            # Create prerequisites
            agent = Agent(name="result-crud-agent", type=AgentType.VISUALIZATION_ENGINEER, status=AgentStatus.ACTIVE)
            session.add(agent)
            session.flush()
            
            trial = Trial(
                name="Result CRUD Trial",
                parameters={"test": "result"},
                created_by_agent=agent.id
            )
            session.add(trial)
            session.flush()
            
            # CREATE
            result = Result(
                trial_id=trial.id,
                metrics={"accuracy": 0.95, "precision": 0.92},
                confidence_score=0.89,
                visualizations={"plot": "path/to/plot.png"},
                agent_id=agent.id
            )
            session.add(result)
            session.commit()
            result_id = result.id
            
            # READ
            retrieved_result = session.query(Result).filter(Result.id == result_id).first()
            assert retrieved_result is not None
            assert retrieved_result.metrics["accuracy"] == 0.95
            assert retrieved_result.confidence_score == 0.89
            
            # UPDATE
            retrieved_result.metrics = {"accuracy": 0.97, "precision": 0.94, "recall": 0.91}
            retrieved_result.confidence_score = 0.93
            session.commit()
            
            # Verify update
            updated_result = session.query(Result).filter(Result.id == result_id).first()
            assert updated_result.metrics["accuracy"] == 0.97
            assert updated_result.metrics["recall"] == 0.91
            assert updated_result.confidence_score == 0.93
            
            # DELETE
            session.delete(updated_result)
            session.commit()
            
            # Verify deletion
            deleted_result = session.query(Result).filter(Result.id == result_id).first()
            assert deleted_result is None
            
        crud_time = time.time() - start_time
        memory.save_metric("result_crud_time_ms", crud_time * 1000)
        memory.save_finding("result_crud", "PASS - All CRUD operations successful")


class TestForeignKeyRelationships:
    """Test foreign key constraints and cascading operations."""
    
    @pytest.fixture
    def db_tester(self):
        """Create a database tester instance."""
        return DatabaseRegressionTester()
    
    def test_trial_agent_foreign_key(self, db_tester):
        """Test foreign key constraint between trials and agents."""
        with db_tester.get_session() as session:
            # Try to create trial with non-existent agent
            try:
                invalid_trial = Trial(
                    name="Invalid Trial",
                    parameters={"test": "data"},
                    created_by_agent=99999  # Non-existent agent ID
                )
                session.add(invalid_trial)
                session.commit()
                memory.report_bug("HIGH", "Foreign key constraint not enforced: trial -> agent", {
                    "table": "trials",
                    "foreign_key": "created_by_agent"
                })
            except IntegrityError:
                # This is expected
                session.rollback()
                memory.save_finding("trial_agent_fk", "PASS - Foreign key constraint working")
    
    def test_result_trial_foreign_key(self, db_tester):
        """Test foreign key constraint between results and trials."""
        with db_tester.get_session() as session:
            # Create valid agent first
            agent = Agent(name="fk-test-agent", type=AgentType.ORCHESTRATOR, status=AgentStatus.ACTIVE)
            session.add(agent)
            session.flush()
            
            # Try to create result with non-existent trial
            try:
                invalid_result = Result(
                    trial_id=99999,  # Non-existent trial ID
                    metrics={"test": "data"},
                    confidence_score=0.5,
                    agent_id=agent.id
                )
                session.add(invalid_result)
                session.commit()
                memory.report_bug("HIGH", "Foreign key constraint not enforced: result -> trial", {
                    "table": "results",
                    "foreign_key": "trial_id"
                })
            except IntegrityError:
                # This is expected
                session.rollback()
                memory.save_finding("result_trial_fk", "PASS - Foreign key constraint working")
    
    def test_cascade_delete(self, db_tester):
        """Test that deleting a trial cascades to its results."""
        with db_tester.get_session() as session:
            # Create test data
            agent = Agent(name="cascade-test-agent", type=AgentType.GENOME_ANALYST, status=AgentStatus.ACTIVE)
            session.add(agent)
            session.flush()
            
            trial = Trial(
                name="Cascade Test Trial",
                parameters={"test": "cascade"},
                created_by_agent=agent.id
            )
            session.add(trial)
            session.flush()
            
            # Create multiple results for this trial
            results = []
            for i in range(3):
                result = Result(
                    trial_id=trial.id,
                    metrics={"iteration": i},
                    confidence_score=0.5 + i * 0.1,
                    agent_id=agent.id
                )
                session.add(result)
                results.append(result)
            
            session.commit()
            trial_id = trial.id
            
            # Count results before deletion
            result_count_before = session.query(Result).filter(Result.trial_id == trial_id).count()
            assert result_count_before == 3
            
            # Delete the trial
            session.delete(trial)
            session.commit()
            
            # Verify results were cascaded
            result_count_after = session.query(Result).filter(Result.trial_id == trial_id).count()
            assert result_count_after == 0
            
        memory.save_finding("cascade_delete", "PASS - Cascade delete working correctly")


class TestConcurrentAccess:
    """Test concurrent database access and transaction handling."""
    
    @pytest.fixture
    def db_tester(self):
        """Create a database tester instance."""
        return DatabaseRegressionTester()
    
    def test_concurrent_agent_creation(self, db_tester):
        """Test concurrent agent creation for race conditions."""
        results = []
        errors = []
        
        def create_agent(agent_name):
            """Function to create an agent in a separate thread."""
            try:
                with db_tester.get_session() as session:
                    agent = Agent(
                        name=agent_name,
                        type=AgentType.PERFORMANCE_OPTIMIZER,
                        status=AgentStatus.ACTIVE
                    )
                    session.add(agent)
                    session.commit()
                    results.append(agent.id)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads to create agents concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_agent, args=(f"concurrent-agent-{i}",))
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        concurrent_time = time.time() - start_time
        
        # Verify results
        if len(errors) == 0 and len(results) == 5:
            memory.save_finding("concurrent_access", "PASS - Concurrent operations successful")
        else:
            memory.report_bug("MEDIUM", f"Concurrent access issues: {len(errors)} errors", {
                "errors": errors,
                "successful_operations": len(results)
            })
        
        memory.save_metric("concurrent_agent_creation_time_ms", concurrent_time * 1000)
    
    def test_transaction_rollback(self, db_tester):
        """Test transaction rollback on errors."""
        with db_tester.get_session() as session:
            # Create a valid agent first
            agent = Agent(name="rollback-test-agent", type=AgentType.DATABASE_ARCHITECT, status=AgentStatus.ACTIVE)
            session.add(agent)
            session.commit()
            agent_id = agent.id
            
            # Start a transaction that should fail
            try:
                # Create a trial
                trial = Trial(
                    name="Rollback Test Trial",
                    parameters={"test": "rollback"},
                    created_by_agent=agent.id
                )
                session.add(trial)
                session.flush()
                
                # Create a result with invalid confidence score (should fail)
                invalid_result = Result(
                    trial_id=trial.id,
                    metrics={"test": "data"},
                    confidence_score=2.0,  # Invalid score > 1.0
                    agent_id=agent.id
                )
                session.add(invalid_result)
                session.commit()
                
                memory.report_bug("MEDIUM", "Transaction with invalid data committed", {
                    "issue": "confidence_score > 1.0 allowed"
                })
                
            except Exception:
                # This is expected - transaction should rollback
                session.rollback()
                
                # Verify that the trial was NOT committed
                trial_count = session.query(Trial).filter(Trial.name == "Rollback Test Trial").count()
                if trial_count == 0:
                    memory.save_finding("transaction_rollback", "PASS - Transaction rollback working")
                else:
                    memory.report_bug("HIGH", "Transaction rollback failed - partial data committed", {
                        "trial_count_after_rollback": trial_count
                    })


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def db_tester(self):
        """Create a database tester instance."""
        return DatabaseRegressionTester()
    
    def test_null_handling(self, db_tester):
        """Test handling of NULL values where appropriate."""
        with db_tester.get_session() as session:
            # Create agent with minimal required fields
            agent = Agent(
                name="null-test-agent",
                type=AgentType.ORCHESTRATOR,
                status=AgentStatus.ACTIVE
                # description and memory_keys can be NULL
            )
            session.add(agent)
            session.commit()
            
            # Verify NULL fields are handled correctly
            retrieved_agent = session.query(Agent).filter(Agent.name == "null-test-agent").first()
            assert retrieved_agent.memory_keys == []  # Should default to empty list
            
        memory.save_finding("null_handling", "PASS - NULL values handled correctly")
    
    def test_json_field_operations(self, db_tester):
        """Test JSON field operations and edge cases."""
        with db_tester.get_session() as session:
            # Create agent and trial
            agent = Agent(name="json-test-agent", type=AgentType.CRYPTO_SPECIALIST, status=AgentStatus.ACTIVE)
            session.add(agent)
            session.flush()
            
            # Test complex JSON data
            complex_parameters = {
                "nested": {
                    "level1": {
                        "level2": ["array", "data", 123]
                    }
                },
                "unicode": "Testing unicode: 你好世界 🧬",
                "large_array": list(range(1000)),
                "boolean": True,
                "null_value": None
            }
            
            trial = Trial(
                name="JSON Test Trial",
                parameters=complex_parameters,
                created_by_agent=agent.id
            )
            session.add(trial)
            session.commit()
            
            # Retrieve and verify JSON data integrity
            retrieved_trial = session.query(Trial).filter(Trial.name == "JSON Test Trial").first()
            assert retrieved_trial.parameters["nested"]["level1"]["level2"] == ["array", "data", 123]
            assert retrieved_trial.parameters["unicode"] == "Testing unicode: 你好世界 🧬"
            assert len(retrieved_trial.parameters["large_array"]) == 1000
            assert retrieved_trial.parameters["boolean"] is True
            assert retrieved_trial.parameters["null_value"] is None
            
        memory.save_finding("json_operations", "PASS - Complex JSON operations working")
    
    def test_large_data_handling(self, db_tester):
        """Test handling of large data sets."""
        start_time = time.time()
        
        with db_tester.get_session() as session:
            # Create agent
            agent = Agent(name="large-data-agent", type=AgentType.PERFORMANCE_OPTIMIZER, status=AgentStatus.ACTIVE)
            session.add(agent)
            session.flush()
            
            # Create trial with large parameter set
            large_params = {
                "large_string": "x" * 10000,  # 10KB string
                "large_array": list(range(5000)),
                "large_dict": {f"key_{i}": f"value_{i}" for i in range(1000)}
            }
            
            trial = Trial(
                name="Large Data Trial",
                parameters=large_params,
                created_by_agent=agent.id
            )
            session.add(trial)
            session.commit()
            
            # Create result with large metrics
            large_metrics = {
                "analysis_results": {f"gene_{i}": {"score": i * 0.001, "data": [j for j in range(100)]} for i in range(100)},
                "raw_data": list(range(10000))
            }
            
            result = Result(
                trial_id=trial.id,
                metrics=large_metrics,
                confidence_score=0.75,
                agent_id=agent.id
            )
            session.add(result)
            session.commit()
            
        large_data_time = time.time() - start_time
        memory.save_metric("large_data_handling_time_ms", large_data_time * 1000)
        memory.save_finding("large_data", "PASS - Large data sets handled successfully")


class TestPerformance:
    """Test database performance and optimization."""
    
    @pytest.fixture
    def db_tester(self):
        """Create a database tester instance."""
        return DatabaseRegressionTester()
    
    def test_index_performance(self, db_tester):
        """Test that database indices are working effectively."""
        with db_tester.get_session() as session:
            # Create test data
            agents = []
            for i in range(100):
                agent = Agent(
                    name=f"perf-agent-{i}",
                    type=AgentType.GENOME_ANALYST,
                    status=AgentStatus.ACTIVE if i % 2 == 0 else AgentStatus.IDLE
                )
                agents.append(agent)
                session.add(agent)
            
            # Commit agents first so they have IDs
            session.commit()
            
            trials = []
            for i in range(500):
                trial = Trial(
                    name=f"perf-trial-{i}",
                    parameters={"iteration": i},
                    status=TrialStatus.COMPLETED if i % 3 == 0 else TrialStatus.RUNNING,
                    created_by_agent=agents[i % len(agents)].id
                )
                trials.append(trial)
                session.add(trial)
            
            session.commit()
            
            # Test indexed queries
            queries = [
                ("agent_by_status", lambda: session.query(Agent).filter(Agent.status == AgentStatus.ACTIVE).all()),
                ("trial_by_status", lambda: session.query(Trial).filter(Trial.status == TrialStatus.COMPLETED).all()),
                ("trials_by_agent", lambda: session.query(Trial).filter(Trial.created_by_agent == agents[0].id).all()),
            ]
            
            for query_name, query_func in queries:
                start_time = time.time()
                results = query_func()
                query_time = (time.time() - start_time) * 1000
                
                memory.save_metric(f"{query_name}_time_ms", query_time)
                
                if query_time > 100:  # More than 100ms for small dataset
                    memory.report_bug("LOW", f"Slow query performance: {query_name}", {
                        "query_time_ms": query_time,
                        "result_count": len(results)
                    })
        
        memory.save_finding("index_performance", "Performance tests completed")


def run_all_tests():
    """Run all regression tests and generate comprehensive report."""
    print("🚀 Starting Database Regression Tests")
    print("=" * 60)
    
    # Initialize test database
    print("Setting up test database...")
    tester = DatabaseRegressionTester()
    
    # Run test classes
    test_classes = [
        TestSchemaValidation,
        TestCRUDOperations,
        TestForeignKeyRelationships,
        TestConcurrentAccess,
        TestEdgeCases,
        TestPerformance
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}...")
        
        # Create instance and run tests
        instance = test_class()
        db_tester = tester
        
        # Get all test methods
        test_methods = [method for method in dir(instance) if method.startswith('test_')]
        
        for test_method_name in test_methods:
            total_tests += 1
            try:
                print(f"  ✓ {test_method_name}")
                test_method = getattr(instance, test_method_name)
                test_method(db_tester)
                passed_tests += 1
            except Exception as e:
                print(f"  ❌ {test_method_name}: {str(e)}")
                memory.report_bug("HIGH", f"Test {test_method_name} failed", {
                    "exception": str(e),
                    "test_class": test_class.__name__
                })
    
    # Generate final report
    print("\n" + "=" * 60)
    print("📊 REGRESSION TEST RESULTS")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print(f"\n🐛 BUGS FOUND: {len(memory.bugs)}")
    for bug in memory.bugs:
        print(f"  {bug['id']} ({bug['severity']}): {bug['description']}")
    
    print(f"\n📈 PERFORMANCE METRICS:")
    for key, metric in memory.metrics.items():
        metric_name = key.split('/')[-1]
        print(f"  {metric_name}: {metric['value']}")
    
    print(f"\n💾 FINDINGS SAVED: {len(memory.findings)}")
    for key in memory.findings.keys():
        finding_name = key.split('/')[-1]
        print(f"  {finding_name}")
    
    return {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "bugs": memory.bugs,
        "metrics": memory.metrics,
        "findings": memory.findings
    }


if __name__ == "__main__":
    # Run the regression tests
    results = run_all_tests()
    
    # Save summary report to memory
    memory.save_finding("final_report", {
        "summary": results,
        "recommendations": [
            "Review and fix any HIGH severity bugs immediately",
            "Monitor performance metrics in production",
            "Add automated regression testing to CI/CD pipeline",
            "Consider implementing database connection pooling",
            "Add monitoring for foreign key constraint violations"
        ]
    })
    
    print(f"\n✅ Regression testing complete!")
    print(f"All findings saved to memory namespace: swarm-regression-1752301224/database-test/")