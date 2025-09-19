"""Model loading and caching for Whisper and Pyannote models."""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import whisper
from pyannote.audio import Pipeline

from app.config import settings
from app.hardware_detector import get_optimal_device

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and caching of ML models."""

    def __init__(self):
        self.whisper_model: Optional[whisper.Whisper] = None
        self.diarization_pipeline: Optional[Pipeline] = None
        self.device = get_optimal_device()
        self.cache_dir = Path(settings.model.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_whisper_model(self) -> whisper.Whisper:
        """Load Whisper model with caching."""
        if self.whisper_model is None:
            logger.info(f"Loading Whisper model: {settings.model.whisper_model}")

            try:
                # Try to load from cache first
                model_path = self.cache_dir / "whisper" / settings.model.whisper_model
                if model_path.exists():
                    logger.info(f"Loading Whisper model from cache: {model_path}")
                    self.whisper_model = whisper.load_model(
                        settings.model.whisper_model,
                        device=self.device,
                        download_root=str(self.cache_dir)
                    )
                else:
                    logger.info("Downloading and loading Whisper model...")
                    self.whisper_model = whisper.load_model(
                        settings.model.whisper_model,
                        device=self.device,
                        download_root=str(self.cache_dir)
                    )
                    logger.info(f"Whisper model cached at: {model_path}")

            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                # Fallback to CPU if GPU loading fails
                logger.info("Falling back to CPU for Whisper model")
                self.whisper_model = whisper.load_model(
                    settings.model.whisper_model,
                    device="cpu",
                    download_root=str(self.cache_dir)
                )
                self.device = "cpu"

            logger.info(f"Whisper model loaded successfully on {self.device}")

        return self.whisper_model

    def load_diarization_pipeline(self) -> Pipeline:
        """Load Pyannote diarization pipeline."""
        if self.diarization_pipeline is None:
            logger.info(f"Loading Pyannote pipeline: {settings.model.pyannote_model}")

            try:
                # Set Hugging Face token if provided
                if settings.huggingface_token:
                    os.environ["HF_TOKEN"] = settings.huggingface_token

                self.diarization_pipeline = Pipeline.from_pretrained(
                    settings.model.pyannote_model,
                    cache_dir=str(self.cache_dir / "pyannote")
                )

                # Move pipeline to appropriate device
                if self.device.startswith("cuda"):
                    self.diarization_pipeline = self.diarization_pipeline.to(torch.device(self.device))

                logger.info("Pyannote diarization pipeline loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load Pyannote pipeline: {e}")
                raise RuntimeError(f"Could not load diarization pipeline: {e}")

        return self.diarization_pipeline

    def get_models(self) -> Tuple[whisper.Whisper, Pipeline]:
        """Get both Whisper model and diarization pipeline."""
        whisper_model = self.load_whisper_model()
        diarization_pipeline = self.load_diarization_pipeline()
        return whisper_model, diarization_pipeline

    def unload_models(self) -> None:
        """Unload models to free memory."""
        if self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None

        if self.diarization_pipeline is not None:
            del self.diarization_pipeline
            self.diarization_pipeline = None

        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Models unloaded and GPU cache cleared")

    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        return {
            "whisper_model": settings.model.whisper_model,
            "pyannote_model": settings.model.pyannote_model,
            "device": self.device,
            "cache_dir": str(self.cache_dir),
            "whisper_loaded": self.whisper_model is not None,
            "diarization_loaded": self.diarization_pipeline is not None
        }


# Global model loader instance
model_loader = ModelLoader()


def get_whisper_model() -> whisper.Whisper:
    """Get Whisper model instance."""
    return model_loader.load_whisper_model()


def get_diarization_pipeline() -> Pipeline:
    """Get Pyannote diarization pipeline instance."""
    return model_loader.load_diarization_pipeline()


def get_models() -> Tuple[whisper.Whisper, Pipeline]:
    """Get both model instances."""
    return model_loader.get_models()


def unload_models() -> None:
    """Unload all models."""
    model_loader.unload_models()


def get_model_info() -> Dict:
    """Get model information."""
    return model_loader.get_model_info()