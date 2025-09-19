#!/usr/bin/env python3
"""Script to download and cache Whisper and Pyannote models."""

import argparse
import logging
import os
from pathlib import Path

import torch
import whisper
from pyannote.audio import Pipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_cache_dir(cache_dir: str = "./cache") -> Path:
    """Setup cache directory structure."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (cache_path / "whisper").mkdir(exist_ok=True)
    (cache_path / "pyannote").mkdir(exist_ok=True)
    (cache_path / "huggingface").mkdir(exist_ok=True)

    return cache_path


def download_whisper_model(model_name: str, cache_dir: Path) -> None:
    """Download and cache Whisper model."""
    logger.info(f"Downloading Whisper model: {model_name}")

    try:
        # Set cache directory
        os.environ["WHISPER_CACHE_DIR"] = str(cache_dir / "whisper")

        # Download model
        model = whisper.load_model(model_name, download_root=str(cache_dir))

        # Test model loading
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        logger.info(f"✓ Whisper model {model_name} downloaded and cached successfully")
        logger.info(f"  Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f"}M parameters")

    except Exception as e:
        logger.error(f"✗ Failed to download Whisper model {model_name}: {e}")
        raise


def download_pyannote_model(model_name: str, cache_dir: Path, hf_token: str = None) -> None:
    """Download and cache Pyannote model."""
    logger.info(f"Downloading Pyannote model: {model_name}")

    try:
        # Set Hugging Face token if provided
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        # Set cache directory
        os.environ["PYANNOTE_CACHE"] = str(cache_dir / "pyannote")
        os.environ["HF_HOME"] = str(cache_dir / "huggingface")

        # Download pipeline
        pipeline = Pipeline.from_pretrained(model_name)

        # Test pipeline loading
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            pipeline = pipeline.to(torch.device(device))

        logger.info(f"✓ Pyannote model {model_name} downloaded and cached successfully")

    except Exception as e:
        logger.error(f"✗ Failed to download Pyannote model {model_name}: {e}")
        raise


def detect_hardware() -> dict:
    """Detect available hardware."""
    hardware_info = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cpu_cores": os.cpu_count(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    if hardware_info["gpu_available"]:
        hardware_info["gpu_name"] = torch.cuda.get_device_name(0)
        hardware_info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB

    logger.info(f"Detected hardware: {hardware_info}")
    return hardware_info


def main():
    """Main function to download models."""
    parser = argparse.ArgumentParser(description="Download Whisper and Pyannote models")
    parser.add_argument(
        "--whisper-model",
        default="large-v2",
        help="Whisper model to download (default: large-v2)"
    )
    parser.add_argument(
        "--pyannote-model",
        default="pyannote/speaker-diarization-3.1",
        help="Pyannote model to download (default: pyannote/speaker-diarization-3.1)"
    )
    parser.add_argument(
        "--cache-dir",
        default="./cache",
        help="Cache directory for models (default: ./cache)"
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token for Pyannote models"
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Force device for testing (default: auto-detect)"
    )

    args = parser.parse_args()

    # Setup cache directory
    cache_dir = setup_cache_dir(args.cache_dir)
    logger.info(f"Cache directory: {cache_dir}")

    # Detect hardware
    hardware = detect_hardware()
    device = args.device or hardware["device"]

    # Set device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    logger.info(f"Using device: {device}")

    # Download Whisper model
    try:
        download_whisper_model(args.whisper_model, cache_dir)
    except Exception as e:
        logger.error(f"Failed to download Whisper model: {e}")
        return 1

    # Download Pyannote model
    try:
        download_pyannote_model(args.pyannote_model, cache_dir, args.hf_token)
    except Exception as e:
        logger.error(f"Failed to download Pyannote model: {e}")
        return 1

    logger.info("✓ All models downloaded successfully!")
    logger.info(f"Models cached in: {cache_dir}")
    logger.info("You can now start the server with: python app/main.py")

    return 0


if __name__ == "__main__":
    exit(main())
