"""
Pydantic schemas for API models
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# Enums
class TrialStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentRole(str, Enum):
    ANALYZER = "analyzer"
    VALIDATOR = "validator"
    REPORTER = "reporter"
    COORDINATOR = "coordinator"


# Base models
class BaseSchema(BaseModel):
    """Base schema with common configuration"""
    class Config:
        from_attributes = True
        use_enum_values = True


# Agent models
class AgentBase(BaseSchema):
    name: str = Field(..., min_length=1, max_length=100)
    role: AgentRole
    capabilities: List[str] = Field(default_factory=list)
    active: bool = True


class AgentCreate(AgentBase):
    password: str = Field(..., min_length=8)


class AgentUpdate(BaseSchema):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    role: Optional[AgentRole] = None
    capabilities: Optional[List[str]] = None
    active: Optional[bool] = None


class Agent(AgentBase):
    id: int
    created_at: datetime
    updated_at: datetime
    last_active: Optional[datetime] = None


class AgentWithToken(Agent):
    access_token: str
    token_type: str = "bearer"


# Trial models
class TrialBase(BaseSchema):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    organism: str = Field(..., min_length=1, max_length=100)
    genome_file: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('parameters')
    def validate_parameters(cls, v):
        required_params = ['window_size', 'min_confidence', 'trait_count']
        for param in required_params:
            if param not in v:
                raise ValueError(f"Missing required parameter: {param}")
        return v


class TrialCreate(TrialBase):
    created_by: int  # Agent ID


class TrialUpdate(BaseSchema):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    status: Optional[TrialStatus] = None
    parameters: Optional[Dict[str, Any]] = None


class Trial(TrialBase):
    id: int
    status: TrialStatus
    created_by: int
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


# Result models
class ResultBase(BaseSchema):
    trial_id: int
    gene_id: str
    traits: List[str]
    confidence_scores: Dict[str, float]
    codon_usage_bias: Dict[str, float]
    regulatory_context: Optional[Dict[str, Any]] = None


class ResultCreate(ResultBase):
    pass


class ResultUpdate(BaseSchema):
    confidence_scores: Optional[Dict[str, float]] = None
    regulatory_context: Optional[Dict[str, Any]] = None
    validated: Optional[bool] = None
    validation_notes: Optional[str] = None


class Result(ResultBase):
    id: int
    created_at: datetime
    validated: bool = False
    validation_notes: Optional[str] = None
    validated_by: Optional[int] = None
    validated_at: Optional[datetime] = None


# Progress models
class ProgressBase(BaseSchema):
    trial_id: int
    stage: str
    progress_percentage: float = Field(..., ge=0, le=100)
    current_task: str
    genes_processed: int = 0
    total_genes: int = 0


class ProgressCreate(ProgressBase):
    pass


class ProgressUpdate(BaseSchema):
    progress_percentage: Optional[float] = Field(None, ge=0, le=100)
    current_task: Optional[str] = None
    genes_processed: Optional[int] = None
    error_count: Optional[int] = None


class Progress(ProgressBase):
    id: int
    created_at: datetime
    updated_at: datetime
    error_count: int = 0
    estimated_completion: Optional[datetime] = None


# Batch operation models
class BatchOperation(BaseSchema):
    operation: str
    items: List[Dict[str, Any]]
    
    @validator('items')
    def validate_batch_size(cls, v):
        if len(v) > 1000:  # Max batch size
            raise ValueError("Batch size cannot exceed 1000 items")
        return v


class BatchResult(BaseSchema):
    success_count: int
    error_count: int
    results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]


# WebSocket models
class WSMessage(BaseSchema):
    type: str  # 'trial_update', 'progress_update', 'result_added', etc.
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Authentication models
class Token(BaseSchema):
    access_token: str
    token_type: str


class TokenData(BaseSchema):
    agent_id: Optional[int] = None


class AgentLogin(BaseSchema):
    name: str
    password: str


# Response models
class PaginatedResponse(BaseSchema):
    items: List[Any]
    total: int
    page: int
    page_size: int
    pages: int