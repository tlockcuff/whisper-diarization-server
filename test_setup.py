#!/usr/bin/env python3
"""
Test script to verify CUDA/cuDNN setup and model loading
"""
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cuda_setup():
    """Test CUDA and cuDNN availability"""
    logger.info("🧪 Testing CUDA setup...")
    
    # Check PyTorch and CUDA
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
    else:
        logger.warning("⚠️ CUDA not available, will use CPU")

def test_model_loading():
    """Test model loading with fallback"""
    logger.info("🧪 Testing model loading...")
    
    try:
        from faster_whisper import WhisperModel
        
        # Test ASR model loading
        try:
            logger.info("Testing ASR model on GPU...")
            asr_model = WhisperModel("base", device="cuda", compute_type="float16")
            logger.info("✅ ASR model loaded successfully (GPU)")
        except Exception as e:
            logger.warning(f"⚠️ GPU failed: {e}")
            logger.info("Testing ASR model on CPU...")
            asr_model = WhisperModel("base", device="cpu", compute_type="int8")
            logger.info("✅ ASR model loaded successfully (CPU)")
        
        # Test diarization pipeline
        try:
            from pyannote.audio import Pipeline
            logger.info("Testing diarization pipeline...")
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1")
            logger.info("✅ Diarization pipeline loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load diarization pipeline: {e}")
            logger.info("💡 Note: This may require HuggingFace token for some models")
        
    except ImportError as e:
        logger.error(f"❌ Failed to import models: {e}")

if __name__ == "__main__":
    test_cuda_setup()
    test_model_loading()
    logger.info("🎉 Test completed!")
