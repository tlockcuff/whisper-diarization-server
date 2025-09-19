#!/usr/bin/env python3
"""
Script to pre-download and cache models locally
"""
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_whisper_models():
    """Download Whisper models to local cache"""
    try:
        from faster_whisper import WhisperModel
        
        cache_dir = Path("cache/whisper")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        models = ["base", "small", "medium"]  # Common models
        
        for model_name in models:
            logger.info(f"📥 Downloading Whisper model: {model_name}")
            try:
                model = WhisperModel(model_name, download_root=str(cache_dir))
                logger.info(f"✅ Downloaded Whisper model: {model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to download {model_name}: {e}")
                
    except ImportError:
        logger.error("❌ faster-whisper not installed")

def download_diarization_models():
    """Download diarization models to local cache"""
    try:
        from pyannote.audio import Pipeline
        
        cache_dir = Path("cache/huggingface")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set HuggingFace cache directory
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
        
        models = [
            "pyannote/speaker-diarization@2.1",
            "pyannote/speaker-diarization-3.1"
        ]
        
        hf_token = os.getenv("HF_TOKEN")
        
        for model_name in models:
            logger.info(f"📥 Downloading diarization model: {model_name}")
            try:
                if hf_token:
                    pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
                else:
                    pipeline = Pipeline.from_pretrained(model_name)
                logger.info(f"✅ Downloaded diarization model: {model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to download {model_name}: {e}")
                if "authentication" in str(e).lower():
                    logger.warning("💡 Some models require HF_TOKEN environment variable")
                
    except ImportError:
        logger.error("❌ pyannote.audio not installed")

def download_pip_cache():
    """Cache pip packages locally"""
    try:
        import subprocess
        
        cache_dir = Path("cache/pip")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("📥 Caching pip packages...")
        
        # Download packages to cache
        with open("requirements.txt", "r") as f:
            packages = f.read().strip().split("\n")
        
        for package in packages:
            if package.strip() and not package.startswith("#"):
                try:
                    subprocess.run([
                        sys.executable, "-m", "pip", "download", 
                        "--dest", str(cache_dir), 
                        "--no-deps", package.strip()
                    ], check=True, capture_output=True)
                    logger.info(f"✅ Cached package: {package}")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"⚠️ Failed to cache {package}: {e}")
        
    except Exception as e:
        logger.error(f"❌ Failed to cache pip packages: {e}")

def main():
    """Main function to download all models and cache"""
    logger.info("🚀 Starting model download and caching...")
    
    # Create cache directories
    for cache_type in ["models", "huggingface", "whisper", "pip"]:
        Path(f"cache/{cache_type}").mkdir(parents=True, exist_ok=True)
    
    # Download models
    logger.info("📥 Downloading Whisper models...")
    download_whisper_models()
    
    logger.info("📥 Downloading diarization models...")
    download_diarization_models()
    
    logger.info("📥 Caching pip packages...")
    download_pip_cache()
    
    logger.info("🎉 Model download and caching completed!")
    
    # Print cache size
    try:
        cache_size = sum(f.stat().st_size for f in Path("cache").rglob("*") if f.is_file())
        logger.info(f"📊 Total cache size: {cache_size / 1024**3:.2f} GB")
    except Exception:
        pass

if __name__ == "__main__":
    main()
