"""
Progress API router
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import List, Optional
from datetime import datetime, timedelta
from app.db.database import get_db
from app.models import database as db_models
from app.models import schemas
from app.core.auth import get_current_active_agent
from app.websocket.manager import manager
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=schemas.Progress, status_code=status.HTTP_201_CREATED)
async def create_progress_update(
    progress: schemas.ProgressCreate,
    db: AsyncSession = Depends(get_db),
    current_agent: db_models.Agent = Depends(get_current_active_agent)
):
    """Create a new progress update"""
    # Verify trial exists
    trial_result = await db.execute(
        select(db_models.Trial).where(db_models.Trial.id == progress.trial_id)
    )
    trial = trial_result.scalar_one_or_none()
    
    if not trial:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trial {progress.trial_id} not found"
        )
    
    # Calculate estimated completion if processing genes
    estimated_completion = None
    if progress.genes_processed > 0 and progress.total_genes > 0:
        # Get the oldest progress update to calculate rate
        first_update = await db.execute(
            select(db_models.Progress)
            .where(db_models.Progress.trial_id == progress.trial_id)
            .order_by(db_models.Progress.created_at.asc())
            .limit(1)
        )
        first = first_update.scalar_one_or_none()
        
        if first and first.genes_processed > 0:
            time_elapsed = (datetime.utcnow() - first.created_at).total_seconds()
            genes_per_second = (progress.genes_processed - first.genes_processed) / time_elapsed
            if genes_per_second > 0:
                remaining_genes = progress.total_genes - progress.genes_processed
                remaining_seconds = remaining_genes / genes_per_second
                estimated_completion = datetime.utcnow() + timedelta(seconds=remaining_seconds)
    
    db_progress = db_models.Progress(
        **progress.dict(),
        estimated_completion=estimated_completion
    )
    
    db.add(db_progress)
    await db.commit()
    await db.refresh(db_progress)
    
    # Send WebSocket notification
    await manager.broadcast({
        "type": "progress_update",
        "data": {
            "trial_id": progress.trial_id,
            "stage": progress.stage,
            "progress_percentage": progress.progress_percentage,
            "genes_processed": progress.genes_processed,
            "total_genes": progress.total_genes,
            "estimated_completion": estimated_completion.isoformat() if estimated_completion else None
        }
    })
    
    logger.info(f"Progress update for trial {progress.trial_id}: {progress.progress_percentage}%")
    return db_progress


@router.get("/trial/{trial_id}", response_model=List[schemas.Progress])
async def get_trial_progress(
    trial_id: int,
    limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Get progress updates for a specific trial"""
    # Verify trial exists
    trial_result = await db.execute(
        select(db_models.Trial).where(db_models.Trial.id == trial_id)
    )
    trial = trial_result.scalar_one_or_none()
    
    if not trial:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trial {trial_id} not found"
        )
    
    # Get progress updates
    result = await db.execute(
        select(db_models.Progress)
        .where(db_models.Progress.trial_id == trial_id)
        .order_by(db_models.Progress.created_at.desc())
        .limit(limit)
    )
    progress_updates = result.scalars().all()
    
    return progress_updates


@router.get("/trial/{trial_id}/latest", response_model=schemas.Progress)
async def get_latest_progress(
    trial_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get the latest progress update for a trial"""
    # Verify trial exists
    trial_result = await db.execute(
        select(db_models.Trial).where(db_models.Trial.id == trial_id)
    )
    trial = trial_result.scalar_one_or_none()
    
    if not trial:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trial {trial_id} not found"
        )
    
    # Get latest progress
    result = await db.execute(
        select(db_models.Progress)
        .where(db_models.Progress.trial_id == trial_id)
        .order_by(db_models.Progress.created_at.desc())
        .limit(1)
    )
    latest_progress = result.scalar_one_or_none()
    
    if not latest_progress:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No progress updates found for trial {trial_id}"
        )
    
    return latest_progress


@router.patch("/{progress_id}", response_model=schemas.Progress)
async def update_progress(
    progress_id: int,
    progress_update: schemas.ProgressUpdate,
    db: AsyncSession = Depends(get_db),
    current_agent: db_models.Agent = Depends(get_current_active_agent)
):
    """Update a progress entry"""
    result = await db.execute(
        select(db_models.Progress).where(db_models.Progress.id == progress_id)
    )
    db_progress = result.scalar_one_or_none()
    
    if not db_progress:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Progress {progress_id} not found"
        )
    
    # Update fields
    update_data = progress_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_progress, field, value)
    
    # Recalculate estimated completion if genes data changed
    if "genes_processed" in update_data and db_progress.total_genes > 0:
        # Calculate based on current rate
        first_update = await db.execute(
            select(db_models.Progress)
            .where(db_models.Progress.trial_id == db_progress.trial_id)
            .order_by(db_models.Progress.created_at.asc())
            .limit(1)
        )
        first = first_update.scalar_one_or_none()
        
        if first and first.genes_processed > 0:
            time_elapsed = (datetime.utcnow() - first.created_at).total_seconds()
            genes_per_second = (db_progress.genes_processed - first.genes_processed) / time_elapsed
            if genes_per_second > 0:
                remaining_genes = db_progress.total_genes - db_progress.genes_processed
                remaining_seconds = remaining_genes / genes_per_second
                db_progress.estimated_completion = datetime.utcnow() + timedelta(seconds=remaining_seconds)
    
    db_progress.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(db_progress)
    
    # Send WebSocket notification
    await manager.broadcast({
        "type": "progress_update",
        "data": {
            "trial_id": db_progress.trial_id,
            "progress_id": progress_id,
            "progress_percentage": db_progress.progress_percentage,
            "current_task": db_progress.current_task
        }
    })
    
    logger.info(f"Progress {progress_id} updated by agent {current_agent.name}")
    return db_progress


@router.get("/active", response_model=List[schemas.Progress])
async def get_active_trials_progress(
    db: AsyncSession = Depends(get_db)
):
    """Get latest progress for all active (running) trials"""
    # Get all running trials
    trials_result = await db.execute(
        select(db_models.Trial)
        .where(db_models.Trial.status == schemas.TrialStatus.RUNNING)
    )
    running_trials = trials_result.scalars().all()
    
    if not running_trials:
        return []
    
    # Get latest progress for each trial
    trial_ids = [trial.id for trial in running_trials]
    
    # Use a subquery to get the latest progress per trial
    from sqlalchemy import func
    
    latest_progress = []
    for trial_id in trial_ids:
        result = await db.execute(
            select(db_models.Progress)
            .where(db_models.Progress.trial_id == trial_id)
            .order_by(db_models.Progress.created_at.desc())
            .limit(1)
        )
        progress = result.scalar_one_or_none()
        if progress:
            latest_progress.append(progress)
    
    return latest_progress


@router.post("/batch", response_model=schemas.BatchResult)
async def batch_create_progress(
    batch: schemas.BatchOperation,
    db: AsyncSession = Depends(get_db),
    current_agent: db_models.Agent = Depends(get_current_active_agent)
):
    """Create multiple progress updates in a single request"""
    if batch.operation != "create":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only 'create' operation is supported for progress updates"
        )
    
    success_count = 0
    error_count = 0
    results = []
    errors = []
    
    # Verify all trial IDs exist
    trial_ids = set(item.get("trial_id") for item in batch.items if item.get("trial_id"))
    if trial_ids:
        trials_result = await db.execute(
            select(db_models.Trial.id).where(db_models.Trial.id.in_(trial_ids))
        )
        existing_trial_ids = set(row[0] for row in trials_result)
        
        for item in batch.items:
            try:
                if item.get("trial_id") not in existing_trial_ids:
                    raise ValueError(f"Trial {item.get('trial_id')} not found")
                
                progress_data = schemas.ProgressCreate(**item)
                db_progress = db_models.Progress(**progress_data.dict())
                db.add(db_progress)
                await db.flush()
                
                results.append({
                    "id": db_progress.id,
                    "trial_id": db_progress.trial_id,
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
    
    # Send WebSocket notification for batch update
    if success_count > 0:
        await manager.broadcast({
            "type": "batch_progress_created",
            "data": {
                "count": success_count,
                "trials_updated": list(trial_ids)
            }
        })
    
    logger.info(f"Batch created {success_count} progress updates by agent {current_agent.name}")
    
    return {
        "success_count": success_count,
        "error_count": error_count,
        "results": results,
        "errors": errors
    }