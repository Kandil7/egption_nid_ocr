"""
Egyptian ID OCR API - Main Application
FastAPI server for extracting information from Egyptian National ID cards.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.services.pipeline import IDExtractionPipeline
from app.core.config import settings
from app.core.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Loads models at startup.
    """
    # Startup
    logger.info("Starting Egyptian ID OCR API...")
    logger.info(f"Environment: {settings.APP_ENV}")
    logger.info(f"Loading models...")

    try:
        IDExtractionPipeline.initialize()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")

    yield

    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Egyptian ID OCR API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.APP_ENV == "development",
        workers=1 if settings.APP_ENV == "development" else settings.APP_WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    )
