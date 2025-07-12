"""
SQLAlchemy database models
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, JSON, Float, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.database import Base


class Agent(Base):
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    role = Column(String(50), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    capabilities = Column(JSON, default=list)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_active = Column(DateTime(timezone=True))
    
    # Relationships
    created_trials = relationship("Trial", back_populates="creator")
    validated_results = relationship("Result", back_populates="validator")


class Trial(Base):
    __tablename__ = "trials"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    organism = Column(String(100), nullable=False, index=True)
    genome_file = Column(String(500), nullable=False)
    parameters = Column(JSON, nullable=False)
    status = Column(String(50), nullable=False, default="pending", index=True)
    created_by = Column(Integer, ForeignKey("agents.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
    
    # Relationships
    creator = relationship("Agent", back_populates="created_trials")
    results = relationship("Result", back_populates="trial", cascade="all, delete-orphan")
    progress_updates = relationship("Progress", back_populates="trial", cascade="all, delete-orphan")


class Result(Base):
    __tablename__ = "results"
    
    id = Column(Integer, primary_key=True, index=True)
    trial_id = Column(Integer, ForeignKey("trials.id"), nullable=False, index=True)
    gene_id = Column(String(100), nullable=False, index=True)
    traits = Column(JSON, nullable=False)  # List of trait names
    confidence_scores = Column(JSON, nullable=False)  # Dict of trait: score
    codon_usage_bias = Column(JSON, nullable=False)  # Dict of codon: frequency
    regulatory_context = Column(JSON)  # Additional regulatory information
    validated = Column(Boolean, default=False, index=True)
    validation_notes = Column(Text)
    validated_by = Column(Integer, ForeignKey("agents.id"))
    validated_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    trial = relationship("Trial", back_populates="results")
    validator = relationship("Agent", back_populates="validated_results")


class Progress(Base):
    __tablename__ = "progress"
    
    id = Column(Integer, primary_key=True, index=True)
    trial_id = Column(Integer, ForeignKey("trials.id"), nullable=False, index=True)
    stage = Column(String(100), nullable=False)
    progress_percentage = Column(Float, nullable=False, default=0.0)
    current_task = Column(String(500), nullable=False)
    genes_processed = Column(Integer, default=0)
    total_genes = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    estimated_completion = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    trial = relationship("Trial", back_populates="progress_updates")