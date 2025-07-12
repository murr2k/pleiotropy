"""
SQLAlchemy models for the Genomic Pleiotropy Cryptanalysis trial database.
"""
from datetime import datetime
from enum import Enum
import json
from typing import Optional, Dict, List, Any

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float, 
    DateTime, ForeignKey, JSON, CheckConstraint, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func

Base = declarative_base()


class AgentType(str, Enum):
    """Types of agents in the system."""
    ORCHESTRATOR = "orchestrator"
    DATABASE_ARCHITECT = "database_architect"
    GENOME_ANALYST = "genome_analyst"
    CRYPTO_SPECIALIST = "crypto_specialist"
    VISUALIZATION_ENGINEER = "visualization_engineer"
    PERFORMANCE_OPTIMIZER = "performance_optimizer"


class AgentStatus(str, Enum):
    """Status of an agent."""
    ACTIVE = "active"
    IDLE = "idle"
    OFFLINE = "offline"


class TrialStatus(str, Enum):
    """Status of a trial."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProgressStatus(str, Enum):
    """Status of a progress entry."""
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Agent(Base):
    """Model for tracking AI agents."""
    __tablename__ = 'agents'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)
    type = Column(String, nullable=False)
    status = Column(String, nullable=False, default=AgentStatus.ACTIVE)
    last_heartbeat = Column(DateTime, default=datetime.utcnow)
    tasks_completed = Column(Integer, default=0)
    memory_keys = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    created_trials = relationship("Trial", back_populates="creator")
    results = relationship("Result", back_populates="agent")
    progress_entries = relationship("Progress", back_populates="agent")
    
    __table_args__ = (
        CheckConstraint(
            type.in_([t.value for t in AgentType]),
            name='check_agent_type'
        ),
        CheckConstraint(
            status.in_([s.value for s in AgentStatus]),
            name='check_agent_status'
        ),
        Index('idx_agents_type', 'type'),
        Index('idx_agents_status', 'status'),
    )
    
    def update_heartbeat(self):
        """Update the last heartbeat timestamp."""
        self.last_heartbeat = datetime.utcnow()
    
    def increment_tasks(self):
        """Increment the tasks completed counter."""
        self.tasks_completed += 1
    
    def add_memory_key(self, key: str):
        """Add a memory key to the agent's list."""
        if self.memory_keys is None:
            self.memory_keys = []
        if key not in self.memory_keys:
            self.memory_keys = self.memory_keys + [key]


class Trial(Base):
    """Model for experimental trials."""
    __tablename__ = 'trials'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    parameters = Column(JSON, nullable=False)
    hypothesis = Column(Text)
    status = Column(String, nullable=False, default=TrialStatus.PENDING)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by_agent = Column(Integer, ForeignKey('agents.id'), nullable=False)
    
    # Relationships
    creator = relationship("Agent", back_populates="created_trials")
    results = relationship("Result", back_populates="trial", cascade="all, delete-orphan")
    
    __table_args__ = (
        CheckConstraint(
            status.in_([s.value for s in TrialStatus]),
            name='check_trial_status'
        ),
        Index('idx_trials_status', 'status'),
        Index('idx_trials_created_by', 'created_by_agent'),
    )
    
    def set_status(self, status: TrialStatus):
        """Update the trial status."""
        self.status = status
    
    def add_parameter(self, key: str, value: Any):
        """Add or update a parameter."""
        if self.parameters is None:
            self.parameters = {}
        self.parameters[key] = value


class Result(Base):
    """Model for trial results."""
    __tablename__ = 'results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trial_id = Column(Integer, ForeignKey('trials.id', ondelete='CASCADE'), nullable=False)
    metrics = Column(JSON, nullable=False)
    confidence_score = Column(Float)
    visualizations = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=datetime.utcnow)
    agent_id = Column(Integer, ForeignKey('agents.id'), nullable=False)
    
    # Relationships
    trial = relationship("Trial", back_populates="results")
    agent = relationship("Agent", back_populates="results")
    
    __table_args__ = (
        CheckConstraint(
            (confidence_score >= 0.0) & (confidence_score <= 1.0),
            name='check_confidence_score_range'
        ),
        Index('idx_results_trial', 'trial_id'),
        Index('idx_results_confidence', 'confidence_score'),
    )
    
    def add_metric(self, key: str, value: Any):
        """Add or update a metric."""
        if self.metrics is None:
            self.metrics = {}
        self.metrics[key] = value
    
    def add_visualization(self, name: str, path: str):
        """Add a visualization reference."""
        if self.visualizations is None:
            self.visualizations = {}
        self.visualizations[name] = path


class Progress(Base):
    """Model for tracking task progress."""
    __tablename__ = 'progress'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(Integer, ForeignKey('agents.id'), nullable=False)
    task_id = Column(String, nullable=False)
    status = Column(String, nullable=False)
    message = Column(Text)
    percentage = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    agent = relationship("Agent", back_populates="progress_entries")
    
    __table_args__ = (
        CheckConstraint(
            status.in_([s.value for s in ProgressStatus]),
            name='check_progress_status'
        ),
        CheckConstraint(
            (percentage >= 0) & (percentage <= 100),
            name='check_percentage_range'
        ),
        Index('idx_progress_agent', 'agent_id'),
        Index('idx_progress_task', 'task_id'),
    )
    
    def update_progress(self, percentage: int, message: Optional[str] = None):
        """Update progress percentage and optionally the message."""
        self.percentage = min(100, max(0, percentage))
        if message:
            self.message = message
        if percentage >= 100:
            self.status = ProgressStatus.COMPLETED