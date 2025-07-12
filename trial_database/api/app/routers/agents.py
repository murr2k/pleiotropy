"""
Agents API router
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from datetime import datetime, timedelta
from app.db.database import get_db
from app.models import database as db_models
from app.models import schemas
from app.core.auth import (
    get_current_active_agent,
    get_password_hash,
    verify_password,
    create_access_token,
    require_role
)
from app.core.config import settings
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/register", response_model=schemas.Agent, status_code=status.HTTP_201_CREATED)
async def register_agent(
    agent: schemas.AgentCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new agent"""
    # Check if agent name already exists
    result = await db.execute(
        select(db_models.Agent).where(db_models.Agent.name == agent.name)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent name already registered"
        )
    
    # Create new agent
    hashed_password = get_password_hash(agent.password)
    db_agent = db_models.Agent(
        **agent.dict(exclude={"password"}),
        hashed_password=hashed_password
    )
    
    db.add(db_agent)
    await db.commit()
    await db.refresh(db_agent)
    
    logger.info(f"New agent registered: {db_agent.name} with role {db_agent.role}")
    return db_agent


@router.post("/login", response_model=schemas.AgentWithToken)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Authenticate an agent and return access token"""
    # Find agent by name
    result = await db.execute(
        select(db_models.Agent).where(db_models.Agent.name == form_data.username)
    )
    agent = result.scalar_one_or_none()
    
    if not agent or not verify_password(form_data.password, agent.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect agent name or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not agent.active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Agent account is deactivated"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(agent.id)},
        expires_delta=access_token_expires
    )
    
    # Update last active
    agent.last_active = datetime.utcnow()
    await db.commit()
    
    logger.info(f"Agent {agent.name} logged in")
    
    # Convert to response model
    agent_dict = {
        "id": agent.id,
        "name": agent.name,
        "role": agent.role,
        "capabilities": agent.capabilities,
        "active": agent.active,
        "created_at": agent.created_at,
        "updated_at": agent.updated_at,
        "last_active": agent.last_active,
        "access_token": access_token,
        "token_type": "bearer"
    }
    
    return schemas.AgentWithToken(**agent_dict)


@router.get("/me", response_model=schemas.Agent)
async def get_current_agent_info(
    current_agent: db_models.Agent = Depends(get_current_active_agent)
):
    """Get current agent information"""
    return current_agent


@router.get("/", response_model=List[schemas.Agent])
async def list_agents(
    db: AsyncSession = Depends(get_db),
    current_agent: db_models.Agent = Depends(require_role("coordinator"))
):
    """List all agents (coordinator role required)"""
    result = await db.execute(
        select(db_models.Agent).order_by(db_models.Agent.created_at.desc())
    )
    agents = result.scalars().all()
    return agents


@router.get("/{agent_id}", response_model=schemas.Agent)
async def get_agent(
    agent_id: int,
    db: AsyncSession = Depends(get_db),
    current_agent: db_models.Agent = Depends(get_current_active_agent)
):
    """Get agent information"""
    result = await db.execute(
        select(db_models.Agent).where(db_models.Agent.id == agent_id)
    )
    agent = result.scalar_one_or_none()
    
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )
    
    return agent


@router.patch("/{agent_id}", response_model=schemas.Agent)
async def update_agent(
    agent_id: int,
    agent_update: schemas.AgentUpdate,
    db: AsyncSession = Depends(get_db),
    current_agent: db_models.Agent = Depends(get_current_active_agent)
):
    """Update agent information"""
    # Only coordinators can update other agents
    if current_agent.id != agent_id and current_agent.role != "coordinator":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update your own information"
        )
    
    result = await db.execute(
        select(db_models.Agent).where(db_models.Agent.id == agent_id)
    )
    agent = result.scalar_one_or_none()
    
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )
    
    # Update fields
    update_data = agent_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(agent, field, value)
    
    agent.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(agent)
    
    logger.info(f"Agent {agent_id} updated by {current_agent.name}")
    return agent


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def deactivate_agent(
    agent_id: int,
    db: AsyncSession = Depends(get_db),
    current_agent: db_models.Agent = Depends(require_role("coordinator"))
):
    """Deactivate an agent (coordinator role required)"""
    if current_agent.id == agent_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account"
        )
    
    result = await db.execute(
        select(db_models.Agent).where(db_models.Agent.id == agent_id)
    )
    agent = result.scalar_one_or_none()
    
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )
    
    agent.active = False
    agent.updated_at = datetime.utcnow()
    
    await db.commit()
    
    logger.info(f"Agent {agent_id} deactivated by {current_agent.name}")


@router.get("/stats/active")
async def get_active_agents_stats(
    db: AsyncSession = Depends(get_db)
):
    """Get statistics about active agents"""
    # Get agents active in last 5 minutes
    five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
    
    result = await db.execute(
        select(db_models.Agent).where(
            db_models.Agent.active == True,
            db_models.Agent.last_active > five_minutes_ago
        )
    )
    active_agents = result.scalars().all()
    
    # Group by role
    role_counts = {}
    for agent in active_agents:
        role_counts[agent.role] = role_counts.get(agent.role, 0) + 1
    
    return {
        "total_active": len(active_agents),
        "by_role": role_counts,
        "agents": [
            {
                "id": agent.id,
                "name": agent.name,
                "role": agent.role,
                "last_active": agent.last_active
            }
            for agent in active_agents
        ]
    }