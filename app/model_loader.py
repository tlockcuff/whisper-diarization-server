"""
Robust model loading with hardware-aware fallbacks and error recovery
"""

import torch
import logging
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from typing import Optional, Dict, Any, Callable
import time

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Custom exception for model loading failures"""
    pass


class ModelLoader:
    """Robust model loader with hardware detection and fallback strategies"""

    def __init__(self, hardware_detector):
        self.hardware = hardware_detector
        self.asr_model = None
        self.diarization_pipeline = None
        self.models_loaded = False

    def load_models(self, asr_model_name: str, diarization_model_name: str, hf_token: Optional[str] = None) -> bool:
        """Load all models with comprehensive error handling and fallbacks"""
        logger.info("🔄 Starting model loading process...")

        # First, run hardware detection
        self.hardware.print_hardware_info()

        # Load ASR model
        asr_success = self._load_asr_model(asr_model_name)

        # Load diarization model
        diarization_success = self._load_diarization_model(diarization_model_name, hf_token)

        self.models_loaded = asr_success and diarization_success

        if self.models_loaded:
            logger.info("✅ All models loaded successfully!")
        else:
            logger.warning("⚠️ Some models failed to load - service may have limited functionality")

        return self.models_loaded

    def _load_asr_model(self, model_name: str) -> bool:
        """Load ASR model with device-aware fallbacks"""
        logger.info(f"🔄 Loading ASR model: {model_name}")

        devices_to_try = []
        if self.hardware.cuda_available:
            devices_to_try.append(("cuda", "float16"))
            devices_to_try.append(("cuda", "int8"))
        devices_to_try.append(("cpu", "int8"))

        for device, compute_type in devices_to_try:
            try:
                logger.info(f"  Attempting to load ASR model on {device} with {compute_type}")

                # Set memory fraction for GPU to avoid OOM
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.set_per_process_memory_fraction(0.8, 0)

                self.asr_model = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type,
                    download_root="/app/cache/whisper"
                )

                # Test the model by checking if it loaded successfully
                # We can test by checking the model attributes instead of requiring a file
                if hasattr(self.asr_model, 'model') and self.asr_model.model is not None:
                    logger.info(f"✅ ASR model loaded successfully on {device}")
                    return True
                else:
                    raise ModelLoadError(f"ASR model failed to initialize on {device}")

            except Exception as e:
                logger.warning(f"⚠️ Failed to load ASR model on {device}: {e}")
                continue

        logger.error("❌ Failed to load ASR model on any device")
        return False

    def _load_diarization_model(self, model_name: str, hf_token: Optional[str] = None) -> bool:
        """Load diarization model with GPU/CPU fallbacks"""
        logger.info(f"🔄 Loading diarization model: {model_name}")

        # Try GPU first if available and compatible
        if self.hardware.cuda_available:
            try:
                logger.info("  Attempting GPU load for diarization...")

                if hf_token:
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        model_name, use_auth_token=hf_token
                    )
                else:
                    self.diarization_pipeline = Pipeline.from_pretrained(model_name)

                # Try to move to GPU
                self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cuda"))

                # Test the pipeline by checking if it loaded successfully
                # We can test by checking the pipeline attributes instead of requiring a file
                if hasattr(self.diarization_pipeline, '_pipeline') and self.diarization_pipeline._pipeline is not None:
                    logger.info("✅ Diarization model loaded successfully on GPU")
                    return True
                else:
                    raise ModelLoadError("Diarization model failed to initialize on GPU")

            except Exception as e:
                logger.warning(f"⚠️ GPU diarization failed: {e}")
                logger.info("  Falling back to CPU for diarization...")

        # CPU fallback
        try:
            logger.info("  Loading diarization model on CPU...")

            if hf_token:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    model_name, use_auth_token=hf_token
                )
            else:
                self.diarization_pipeline = Pipeline.from_pretrained(model_name)

            # Test the pipeline by checking if it loaded successfully
            # We can test by checking the pipeline attributes instead of requiring a file
            if hasattr(self.diarization_pipeline, '_pipeline') and self.diarization_pipeline._pipeline is not None:
                logger.info("✅ Diarization model loaded successfully on CPU")
                return True
            else:
                raise ModelLoadError("Diarization model failed to initialize on CPU")

        except Exception as e:
            logger.error(f"❌ Failed to load diarization model: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "asr_loaded": self.asr_model is not None,
            "diarization_loaded": self.diarization_pipeline is not None,
            "hardware_info": self.hardware.detect_hardware()
        }

    def is_ready(self) -> bool:
        """Check if all models are ready for inference"""
        return self.models_loaded and self.asr_model is not None and self.diarization_pipeline is not None


# Global model loader instance
def create_model_loader():
    """Factory function to create model loader with hardware detection"""
    from .hardware_detector import hardware_detector
    return ModelLoader(hardware_detector)
