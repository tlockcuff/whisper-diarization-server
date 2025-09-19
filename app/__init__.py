"""
Whisper Diarization Server - Main Application Package
"""

from .config import config
from .hardware_detector import hardware_detector
from .model_loader import create_model_loader

__version__ = "2.0.0"
__all__ = ["config", "hardware_detector", "create_model_loader"]
