"""
Trials API router
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
from datetime import datetime
from app.db.database import get_db
from app.models import database as db_models
from app.models import schemas
from app.core.auth import get_current_active_agent
from app.websocket.manager import manager
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=schemas.Trial, status_code=status.HTTP_201_CREATED)
async def create_trial(
    trial: schemas.TrialCreate,
    db: AsyncSession = Depends(get_db),
    current_agent: db_models.Agent = Depends(get_current_active_agent)
):
    """Create a new trial"""
    db_trial = db_models.Trial(
        **trial.dict(exclude={"created_by"}),
        created_by=current_agent.id,
        status=schemas.TrialStatus.PENDING
    )
    
    db.add(db_trial)
    await db.commit()
    await db.refresh(db_trial)
    
    # Send WebSocket notification
    await manager.broadcast({
        "type": "trial_created",
        "data": {
            "trial_id": db_trial.id,
            "name": db_trial.name,
            "created_by": current_agent.name
        }
    })
    
    logger.info(f"Trial {db_trial.id} created by agent {current_agent.name}")
    return db_trial


@router.get("/", response_model=schemas.PaginatedResponse)
async def list_trials(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[schemas.TrialStatus] = None,
    organism: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """List trials with pagination and filtering"""
    # Build query
    query = select(db_models.Trial)
    
    if status:
        query = query.where(db_models.Trial.status == status)
    if organism:
        query = query.where(db_models.Trial.organism == organism)
    
    # Get total count
    count_query = select(func.count()).select_from(db_models.Trial)
    if status:
        count_query = count_query.where(db_models.Trial.status == status)
    if organism:
        count_query = count_query.where(db_models.Trial.organism == organism)
    
    total_result = await db.execute(count_query)
    total = total_result.scalar_one()
    
    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(db_models.Trial.created_at.desc())
    
    result = await db.execute(query)
    trials = result.scalars().all()
    
    return {
        "items": trials,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size
    }


@router.get("/{trial_id}", response_model=schemas.Trial)
async def get_trial(
    trial_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific trial"""
    result = await db.execute(
        select(db_models.Trial).where(db_models.Trial.id == trial_id)
    )
    trial = result.scalar_one_or_none()
    
    if not trial:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trial {trial_id} not found"
        )
    
    return trial


@router.patch("/{trial_id}", response_model=schemas.Trial)
async def update_trial(
    trial_id: int,
    trial_update: schemas.TrialUpdate,
    db: AsyncSession = Depends(get_db),
    current_agent: db_models.Agent = Depends(get_current_active_agent)
):
    """Update a trial"""
    result = await db.execute(
        select(db_models.Trial).where(db_models.Trial.id == trial_id)
    )
    trial = result.scalar_one_or_none()
    
    if not trial:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trial {trial_id} not found"
        )
    
    # Update fields
    update_data = trial_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(trial, field, value)
    
    # Handle status transitions
    if "status" in update_data:
        if update_data["status"] == schemas.TrialStatus.RUNNING:
            trial.started_at = datetime.utcnow()
        elif update_data["status"] in [schemas.TrialStatus.COMPLETED, schemas.TrialStatus.FAILED]:
            trial.completed_at = datetime.utcnow()
    
    trial.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(trial)
    
    # Send WebSocket notification
    await manager.broadcast({
        "type": "trial_updated",
        "data": {
            "trial_id": trial.id,
            "status": trial.status,
            "updated_by": current_agent.name
        }
    })
    
    logger.info(f"Trial {trial_id} updated by agent {current_agent.name}")
    return trial


@router.delete("/{trial_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_trial(
    trial_id: int,
    db: AsyncSession = Depends(get_db),
    current_agent: db_models.Agent = Depends(get_current_active_agent)
):
    """Delete a trial and all associated data"""
    result = await db.execute(
        select(db_models.Trial).where(db_models.Trial.id == trial_id)
    )
    trial = result.scalar_one_or_none()
    
    if not trial:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trial {trial_id} not found"
        )
    
    # Only allow deletion of pending or failed trials
    if trial.status not in [schemas.TrialStatus.PENDING, schemas.TrialStatus.FAILED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only delete pending or failed trials"
        )
    
    await db.delete(trial)
    await db.commit()
    
    # Send WebSocket notification
    await manager.broadcast({
        "type": "trial_deleted",
        "data": {
            "trial_id": trial_id,
            "deleted_by": current_agent.name
        }
    })
    
    logger.info(f"Trial {trial_id} deleted by agent {current_agent.name}")


@router.post("/batch", response_model=schemas.BatchResult)
async def batch_create_trials(
    batch: schemas.BatchOperation,
    db: AsyncSession = Depends(get_db),
    current_agent: db_models.Agent = Depends(get_current_active_agent)
):
    """Create multiple trials in a single request"""
    if batch.operation != "create":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only 'create' operation is supported for trials"
        )
    
    success_count = 0
    error_count = 0
    results = []
    errors = []
    
    for item in batch.items:
        try:
            trial_data = schemas.TrialCreate(**item)
            db_trial = db_models.Trial(
                **trial_data.dict(exclude={"created_by"}),
                created_by=current_agent.id,
                status=schemas.TrialStatus.PENDING
            )
            db.add(db_trial)
            await db.flush()  # Get ID without committing
            
            results.append({
                "id": db_trial.id,
                "name": db_trial.name,
                "status": "created"
            })
            success_count += 1
            
        except Exception as e:
            errors.append({
                "item": item,
                "error": str(e)
            })
            error_count += 1
    
    await db.commit()
    
    # Send WebSocket notification
    await manager.broadcast({
        "type": "batch_trials_created",
        "data": {
            "count": success_count,
            "created_by": current_agent.name
        }
    })
    
    logger.info(f"Batch created {success_count} trials by agent {current_agent.name}")
    
    return {
        "success_count": success_count,
        "error_count": error_count,
        "results": results,
        "errors": errors
    }