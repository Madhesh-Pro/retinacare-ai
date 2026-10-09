"""
RetinaCare-AI - FastAPI Production Screening Server
===================================================
Serves real AI-powered Diabetic Retinopathy screening and Grad-CAM explainability.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure project root is in sys.path when running 'python backend/main.py'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router
from backend.services.model_service import ModelService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RetinaCare-Main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Pre-load PyTorch model into memory
    logger.info("Initializing RetinaCare-AI Deep Learning Service...")
    try:
        ModelService.get_predictor()
    except Exception as e:
        logger.warning(f"Could not pre-load model at startup: {e}. Model will load on first request.")
    yield
    # Shutdown
    logger.info("Shutting down RetinaCare-AI Backend Service.")


app = FastAPI(
    title="RetinaCare-AI Screening API",
    description="Real AI-Powered Diabetic Retinopathy Screening & Explainability Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Endpoints
app.include_router(router)

# Mount results directory if exists
os.makedirs("results", exist_ok=True)
app.mount("/results", StaticFiles(directory="results"), name="results")


@app.get("/")
def root():
    return {
        "service": "RetinaCare-AI Clinical Screening API",
        "status": "online",
        "docs_url": "/docs",
        "api_endpoints": {
            "health": "/api/health",
            "predict": "/api/predict",
            "classes": "/api/classes",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
