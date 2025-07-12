"""
FastAPI backend for Genomic Pleiotropy Trial Tracking System
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from app.core.config import settings
from app.db.database import init_db, close_db
from app.routers import trials, results, agents, progress
from app.websocket.manager import websocket_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("Starting up trial tracking API...")
    await init_db()
    yield
    logger.info("Shutting down trial tracking API...")
    await close_db()


# Create FastAPI app
app = FastAPI(
    title="Genomic Pleiotropy Trial Tracking API",
    description="API for tracking and managing genomic pleiotropy analysis trials",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    trials.router,
    prefix="/api/v1/trials",
    tags=["trials"]
)

app.include_router(
    results.router,
    prefix="/api/v1/results",
    tags=["results"]
)

app.include_router(
    agents.router,
    prefix="/api/v1/agents",
    tags=["agents"]
)

app.include_router(
    progress.router,
    prefix="/api/v1/progress",
    tags=["progress"]
)

# Include WebSocket router
app.include_router(
    websocket_router,
    prefix="/ws",
    tags=["websocket"]
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Genomic Pleiotropy Trial Tracking API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}