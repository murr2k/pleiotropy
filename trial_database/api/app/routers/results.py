"""
Results API router
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
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


@router.post("/", response_model=schemas.Result, status_code=status.HTTP_201_CREATED)
async def create_result(
    result: schemas.ResultCreate,
    db: AsyncSession = Depends(get_db),
    current_agent: db_models.Agent = Depends(get_current_active_agent)
):
    """Create a new result"""
    # Verify trial exists
    trial_result = await db.execute(
        select(db_models.Trial).where(db_models.Trial.id == result.trial_id)
    )
    trial = trial_result.scalar_one_or_none()
    
    if not trial:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trial {result.trial_id} not found"
        )
    
    db_result = db_models.Result(**result.dict())
    db.add(db_result)
    await db.commit()
    await db.refresh(db_result)
    
    # Send WebSocket notification
    await manager.broadcast({
        "type": "result_added",
        "data": {
            "trial_id": result.trial_id,
            "gene_id": result.gene_id,
            "traits": result.traits
        }
    })
    
    logger.info(f"Result created for gene {result.gene_id} in trial {result.trial_id}")
    return db_result


@router.get("/", response_model=schemas.PaginatedResponse)
async def list_results(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    trial_id: Optional[int] = None,
    gene_id: Optional[str] = None,
    validated: Optional[bool] = None,
    min_confidence: Optional[float] = Query(None, ge=0, le=1),
    db: AsyncSession = Depends(get_db)
):
    """List results with pagination and filtering"""
    # Build query
    query = select(db_models.Result)
    count_query = select(func.count()).select_from(db_models.Result)
    
    # Apply filters
    filters = []
    if trial_id is not None:
        filters.append(db_models.Result.trial_id == trial_id)
    if gene_id is not None:
        filters.append(db_models.Result.gene_id == gene_id)
    if validated is not None:
        filters.append(db_models.Result.validated == validated)
    
    if filters:
        query = query.where(and_(*filters))
        count_query = count_query.where(and_(*filters))
    
    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar_one()
    
    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(db_models.Result.created_at.desc())
    
    result = await db.execute(query)
    results = result.scalars().all()
    
    # Filter by confidence if specified (post-query filtering)
    if min_confidence is not None:
        filtered_results = []
        for res in results:
            max_confidence = max(res.confidence_scores.values()) if res.confidence_scores else 0
            if max_confidence >= min_confidence:
                filtered_results.append(res)
        results = filtered_results
    
    return {
        "items": results,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size
    }


@router.get("/{result_id}", response_model=schemas.Result)
async def get_result(
    result_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific result"""
    result = await db.execute(
        select(db_models.Result).where(db_models.Result.id == result_id)
    )
    db_result = result.scalar_one_or_none()
    
    if not db_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result {result_id} not found"
        )
    
    return db_result


@router.patch("/{result_id}", response_model=schemas.Result)
async def update_result(
    result_id: int,
    result_update: schemas.ResultUpdate,
    db: AsyncSession = Depends(get_db),
    current_agent: db_models.Agent = Depends(get_current_active_agent)
):
    """Update a result (mainly for validation)"""
    result = await db.execute(
        select(db_models.Result).where(db_models.Result.id == result_id)
    )
    db_result = result.scalar_one_or_none()
    
    if not db_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result {result_id} not found"
        )
    
    # Update fields
    update_data = result_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_result, field, value)
    
    # Handle validation
    if "validated" in update_data and update_data["validated"]:
        db_result.validated_by = current_agent.id
        db_result.validated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(db_result)
    
    # Send WebSocket notification
    await manager.broadcast({
        "type": "result_validated",
        "data": {
            "result_id": result_id,
            "validated": db_result.validated,
            "validated_by": current_agent.name
        }
    })
    
    logger.info(f"Result {result_id} updated by agent {current_agent.name}")
    return db_result


@router.delete("/{result_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_result(
    result_id: int,
    db: AsyncSession = Depends(get_db),
    current_agent: db_models.Agent = Depends(get_current_active_agent)
):
    """Delete a result"""
    result = await db.execute(
        select(db_models.Result).where(db_models.Result.id == result_id)
    )
    db_result = result.scalar_one_or_none()
    
    if not db_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result {result_id} not found"
        )
    
    await db.delete(db_result)
    await db.commit()
    
    logger.info(f"Result {result_id} deleted by agent {current_agent.name}")


@router.post("/batch", response_model=schemas.BatchResult)
async def batch_create_results(
    batch: schemas.BatchOperation,
    db: AsyncSession = Depends(get_db),
    current_agent: db_models.Agent = Depends(get_current_active_agent)
):
    """Create multiple results in a single request"""
    if batch.operation != "create":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only 'create' operation is supported for results"
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
                
                result_data = schemas.ResultCreate(**item)
                db_result = db_models.Result(**result_data.dict())
                db.add(db_result)
                await db.flush()
                
                results.append({
                    "id": db_result.id,
                    "gene_id": db_result.gene_id,
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
    if success_count > 0:
        await manager.broadcast({
            "type": "batch_results_created",
            "data": {
                "count": success_count,
                "created_by": current_agent.name
            }
        })
    
    logger.info(f"Batch created {success_count} results by agent {current_agent.name}")
    
    return {
        "success_count": success_count,
        "error_count": error_count,
        "results": results,
        "errors": errors
    }


@router.get("/trial/{trial_id}/summary")
async def get_trial_results_summary(
    trial_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get summary statistics for a trial's results"""
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
    
    # Get results statistics
    results = await db.execute(
        select(db_models.Result).where(db_models.Result.trial_id == trial_id)
    )
    all_results = results.scalars().all()
    
    # Calculate summary
    total_results = len(all_results)
    validated_count = sum(1 for r in all_results if r.validated)
    
    # Aggregate traits
    all_traits = set()
    trait_counts = {}
    for result in all_results:
        for trait in result.traits:
            all_traits.add(trait)
            trait_counts[trait] = trait_counts.get(trait, 0) + 1
    
    # Calculate average confidence scores
    avg_confidence = {}
    if all_results:
        for trait in all_traits:
            scores = []
            for result in all_results:
                if trait in result.confidence_scores:
                    scores.append(result.confidence_scores[trait])
            if scores:
                avg_confidence[trait] = sum(scores) / len(scores)
    
    return {
        "trial_id": trial_id,
        "total_results": total_results,
        "validated_count": validated_count,
        "validation_percentage": (validated_count / total_results * 100) if total_results > 0 else 0,
        "unique_traits": len(all_traits),
        "trait_counts": trait_counts,
        "average_confidence_scores": avg_confidence
    }