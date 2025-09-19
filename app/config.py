"""
Configuration management for whisper-diarization-server
"""

import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration management with environment variable support"""

    def __init__(self):
        # Model configurations
        self.asr_model = os.getenv("ASR_MODEL", "base")
        self.diarization_model = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization@2.1")
        self.hf_token = os.getenv("HF_TOKEN")

        # Cache configuration
        self.cache_dir = os.getenv("CACHE_DIR", "/app/cache")
        self.whisper_cache = os.path.join(self.cache_dir, "whisper")
        self.hf_cache = os.path.join(self.cache_dir, "huggingface")

        # Hardware configuration
        self.force_cpu = os.getenv("FORCE_CPU", "false").lower() == "true"
        self.max_gpu_memory_fraction = float(os.getenv("MAX_GPU_MEMORY_FRACTION", "0.8"))
        self.cuda_device = os.getenv("CUDA_DEVICE", "0")

        # Performance configuration
        self.batch_size = int(os.getenv("BATCH_SIZE", "1"))
        self.max_audio_length = int(os.getenv("MAX_AUDIO_LENGTH", "300"))  # seconds
        self.enable_streaming = os.getenv("ENABLE_STREAMING", "true").lower() == "true"

        # Server configuration
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.workers = int(os.getenv("WORKERS", "1"))

        # Logging configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_file = os.getenv("LOG_FILE")

    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return any issues"""
        issues = []

        # Check required directories
        for dir_path, name in [(self.whisper_cache, "Whisper cache"),
                              (self.hf_cache, "HuggingFace cache")]:
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    logger.info(f"Created {name} directory: {dir_path}")
                except Exception as e:
                    issues.append(f"Failed to create {name} directory {dir_path}: {e}")

        # Validate model names
        valid_asr_models = ["tiny", "base", "small", "medium", "large"]
        if self.asr_model not in valid_asr_models:
            issues.append(f"Invalid ASR model '{self.asr_model}'. Valid options: {valid_asr_models}")

        # Validate GPU memory fraction
        if not 0.1 <= self.max_gpu_memory_fraction <= 1.0:
            issues.append("MAX_GPU_MEMORY_FRACTION must be between 0.1 and 1.0")

        return {"valid": len(issues) == 0, "issues": issues}

    def print_config(self):
        """Print current configuration (without sensitive data)"""
        logger.info("🔧 Configuration:")
        logger.info(f"  ASR Model: {self.asr_model}")
        logger.info(f"  Diarization Model: {self.diarization_model}")
        logger.info(f"  HF Token: {'✅ Provided' if self.hf_token else '❌ Not provided'}")
        logger.info(f"  Cache Directory: {self.cache_dir}")
        logger.info(f"  Force CPU: {self.force_cpu}")
        logger.info(f"  Max GPU Memory Fraction: {self.max_gpu_memory_fraction}")
        logger.info(f"  Enable Streaming: {self.enable_streaming}")
        logger.info(f"  Server: {self.host}:{self.port}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "asr_model": self.asr_model,
            "diarization_model": self.diarization_model,
            "hf_token_provided": self.hf_token is not None,
            "cache_dir": self.cache_dir,
            "force_cpu": self.force_cpu,
            "max_gpu_memory_fraction": self.max_gpu_memory_fraction,
            "enable_streaming": self.enable_streaming,
            "host": self.host,
            "port": self.port,
        }


# Global configuration instance
config = Config()
