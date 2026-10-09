"""
Model Service Singleton for FastAPI Backend
============================================
Maintains the loaded PyTorch model in GPU/CPU memory for sub-second inference latency.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.inference.predictor import DRPredictor

logger = logging.getLogger("RetinaCare-Backend")


class ModelService:
    _instance: Optional[DRPredictor] = None

    @classmethod
    def get_predictor(cls, checkpoint_path: Optional[str] = None) -> DRPredictor:
        if cls._instance is None:
            logger.info("Initializing ModelService and loading EfficientNet-B2...")
            cls._instance = DRPredictor(checkpoint_path=checkpoint_path)
            logger.info(
                f"ModelService ready (Device: {cls._instance.device}, "
                f"Checkpoint Loaded: {cls._instance.checkpoint_loaded})"
            )
        return cls._instance

    @classmethod
    def reload_checkpoint(cls, checkpoint_path: Optional[str] = None) -> DRPredictor:
        logger.info(f"Reloading checkpoint from {checkpoint_path}...")
        cls._instance = DRPredictor(checkpoint_path=checkpoint_path)
        return cls._instance
