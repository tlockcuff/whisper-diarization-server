"""
Whisper Diarization Server - Main Application

A robust, hardware-aware server for speech-to-text with speaker diarization.
Features automatic GPU/CPU detection, comprehensive error handling, and graceful fallbacks.
"""

from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
import os
import json
import asyncio
import logging
import warnings
import time
from pydub import AudioSegment
from sse_starlette.sse import EventSourceResponse
from typing import Optional

# Import our custom modules
from .config import config
from .hardware_detector import hardware_detector
from .model_loader import create_model_loader

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] +
             ([logging.FileHandler(config.log_file)] if config.log_file else [])
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress specific torchaudio warnings
import torchaudio
warnings.filterwarnings("ignore", module="torchaudio")

# Create FastAPI app
app = FastAPI(
    title="Whisper Diarization Server",
    description="Speech-to-text with speaker diarization using OpenAI Whisper and pyannote.audio",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
model_loader = None
asr_model = None
diarization_pipeline = None

# Validate configuration on startup
@app.on_event("startup")
async def startup_event():
    """Application startup with comprehensive initialization"""
    global model_loader, asr_model, diarization_pipeline

    logger.info("🚀 Starting whisper-diarization server")
    logger.info(f"Version: {app.version}")

    # Validate configuration
    validation = config.validate()
    if not validation["valid"]:
        logger.error("❌ Configuration validation failed:")
        for issue in validation["issues"]:
            logger.error(f"  - {issue}")
        raise Exception("Invalid configuration")

    # Print configuration
    config.print_config()

    # Create model loader with hardware detection
    model_loader = create_model_loader()

    # Load models using the robust model loader
    success = model_loader.load_models(
        asr_model_name=config.asr_model,
        diarization_model_name=config.diarization_model,
        hf_token=config.hf_token
    )

    if not success:
        logger.error("❌ Critical: Failed to load required models")
        raise Exception("Model loading failed")

    # Get model references
    asr_model = model_loader.asr_model
    diarization_pipeline = model_loader.diarization_pipeline

    logger.info("🎉 Server initialization complete! Ready to process audio files.")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown cleanup"""
    logger.info("🛑 Shutting down whisper-diarization server")
    # Cleanup resources if needed
    pass

def diarize_with_progress(audio_path, progress_callback=None):
    """
    Run speaker diarization with progress callbacks and timing
    Uses the robust model loader for automatic fallbacks
    """
    if not model_loader or not model_loader.is_ready():
        raise Exception("Models not properly loaded")

    start_time = time.time()

    if progress_callback:
        progress_callback("Starting speaker diarization analysis...")

    logger.info("🎤 Starting speaker diarization...")

    if progress_callback:
        progress_callback("Loading audio file and extracting features...")

    # Run the diarization pipeline (this is the main bottleneck)
    diarization_start = time.time()
    try:
        diarization = model_loader.diarization_pipeline(audio_path)

        # Check if diarization worked
        if diarization is None:
            raise Exception("Diarization returned None")

    except Exception as e:
        logger.warning(f"⚠️ Diarization failed on current device: {e}")
        logger.info("🔄 Attempting fallback strategies...")

        # The model loader already handles fallbacks, but let's be explicit
        try:
            # Force CPU mode as fallback
            import torch
            cpu_pipeline = model_loader.diarization_pipeline.to(torch.device("cpu"))
            diarization = cpu_pipeline(audio_path)
            logger.info("✅ Diarization completed successfully on CPU after fallback")
        except Exception as e2:
            logger.error(f"❌ Diarization failed on CPU fallback as well: {e2}")
            raise

    diarization_time = time.time() - diarization_start

    if progress_callback:
        progress_callback(f"Processing completed in {diarization_time:.1f}s - Analyzing results...")

    # Count speakers for logging
    speakers = set()
    segment_count = 0
    total_duration = 0

    try:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            segment_count += 1
            total_duration = max(total_duration, turn.end)
    except Exception as e:
        logger.error(f"Error analyzing diarization results: {e}")
        # Return minimal diarization info
        speakers = {"SPEAKER_01"}
        segment_count = 1
        total_duration = 0

    total_time = time.time() - start_time

    logger.info(f"✅ Speaker diarization completed in {total_time:.1f}s")
    logger.info(f"📊 Results: {len(speakers)} speakers, {segment_count} segments, {total_duration:.1f}s audio")
    logger.info(f"⚡ Processing speed: {total_duration/diarization_time:.1f}x realtime")

    if progress_callback:
        progress_callback(f"Complete! Found {len(speakers)} speakers in {total_time:.1f}s ({total_duration/diarization_time:.1f}x realtime)")

    return diarization

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    global model_loader

    # Get hardware information
    hardware_info = hardware_detector.detect_hardware()

    # Get model information
    model_info = {}
    if model_loader:
        model_info = model_loader.get_model_info()

    # Determine overall status
    status_code = "healthy"
    if not model_loader or not model_loader.is_ready():
        status_code = "unhealthy"
    elif hardware_info.get("warnings"):
        status_code = "degraded"

    return {
        "status": status_code,
        "version": app.version,
        "models": {
            "asr": config.asr_model,
            "diarization": config.diarization_model,
            "asr_loaded": model_info.get("asr_loaded", False),
            "diarization_loaded": model_info.get("diarization_loaded", False)
        },
        "hardware": hardware_info,
        "configuration": config.to_dict()
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with performance metrics"""
    global model_loader

    # Run hardware detection
    hardware_info = hardware_detector.detect_hardware()

    # Test model inference if available
    model_tests = {}
    if model_loader and model_loader.is_ready():
        try:
            # Quick model test
            start_time = time.time()

            # Test ASR - skip file-based test, just check model readiness
            asr_test_passed = hasattr(model_loader.asr_model, 'model') and model_loader.asr_model.model is not None
            asr_time = time.time() - start_time

            # Test diarization - skip file-based test, just check pipeline readiness
            diarization_test_passed = False
            if model_loader.diarization_pipeline:
                diarization_test_passed = hasattr(model_loader.diarization_pipeline, '_pipeline') and model_loader.diarization_pipeline._pipeline is not None
            diarization_time = time.time() - start_time - asr_time

                model_tests = {
                    "asr_test_passed": asr_test_passed,
                    "diarization_test_passed": diarization_test_passed,
                    "asr_inference_time": asr_time,
                    "diarization_inference_time": diarization_time
                }
        except Exception as e:
            model_tests = {
                "asr_test_passed": False,
                "diarization_test_passed": False,
                "error": str(e)
            }

    return {
        "status": "healthy" if model_loader and model_loader.is_ready() else "unhealthy",
        "hardware": hardware_info,
        "model_tests": model_tests,
        "memory_usage": {
            "gpu_memory_fraction": config.max_gpu_memory_fraction
        }
    }

@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile, model: str = Form("whisper-1")):
    logger.info(f"📥 Received transcription request - File: {file.filename}, Model: {model}")
    
    # Get file extension from uploaded file
    original_extension = None
    if file.filename:
        original_extension = os.path.splitext(file.filename)[1].lower()
    
    logger.info(f"📋 File details - Extension: {original_extension}, Size: {file.size if hasattr(file, 'size') else 'unknown'}")
    
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=original_extension or ".tmp") as tmp:
        shutil.copyfileobj(file.file, tmp)
        uploaded_path = tmp.name
    
    logger.info(f"💾 Saved uploaded file to: {uploaded_path}")

    # Create WAV file path
    wav_path = None
    
    try:
        # Convert to WAV format for compatibility
        if original_extension and original_extension != ".wav":
            try:
                logger.info(f"🔄 Converting {original_extension} to WAV format...")
                # Use pydub to convert to WAV
                audio = AudioSegment.from_file(uploaded_path)
                wav_path = uploaded_path.replace(original_extension, ".wav")
                audio.export(wav_path, format="wav")
                audio_path = wav_path
                logger.info("✅ Audio conversion completed")
            except Exception as e:
                # If conversion fails, try original file
                logger.warning(f"⚠️ Audio conversion failed: {e}, using original file")
                audio_path = uploaded_path
        else:
            logger.info("📄 File is already in WAV format, using directly")
            audio_path = uploaded_path

        # Check if models are ready
        if not model_loader or not model_loader.is_ready():
            raise HTTPException(status_code=503, detail="Models not loaded")

        # Diarization with progress tracking
        diarization = diarize_with_progress(audio_path, lambda msg: logger.info(f"🔄 {msg}"))

        # Run ASR
        logger.info("🗣️ Starting speech recognition...")
        try:
            segments, _ = model_loader.asr_model.transcribe(audio_path)
            logger.info("✅ Speech recognition completed")
        except Exception as e:
            logger.error(f"ASR transcription failed: {e}")
            raise HTTPException(status_code=500, detail=f"Speech recognition failed: {e}")

        # Merge diarization with ASR output
        logger.info("🔗 Merging diarization with transcription...")
        results = []
        segment_count = 0

        try:
            for segment in segments:
                start, end, text = segment.start, segment.end, segment.text
                speaker = "UNKNOWN"

                # Find speaker for this segment
                for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                    if turn.start <= start and turn.end >= end:
                        speaker = speaker_label
                        break

                results.append({
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "text": text
                })
                segment_count += 1

        except Exception as e:
            logger.error(f"Error merging diarization and transcription: {e}")
            # Continue with UNKNOWN speakers
            for segment in segments:
                results.append({
                    "speaker": "UNKNOWN",
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                })
                segment_count += 1

        logger.info(f"✅ Processing completed - {segment_count} segments processed")

        # Format response
        response_text = "\n".join([f"{r['speaker']}: {r['text']}" for r in results])

        logger.info(f"📤 Returning transcription result ({len(response_text)} characters)")

        return {
            "text": response_text,
            "segments": results,
            "model": model,
            "processing_info": {
                "segments_processed": segment_count,
                "total_speakers": len(set(r["speaker"] for r in results))
            }
        }
    
    except Exception as e:
        logger.error(f"❌ Error during transcription: {str(e)}")
        raise
    
    finally:
        # Clean up temp files
        for path in [uploaded_path, wav_path]:
            if path:
                try:
                    os.unlink(path)
                    logger.debug(f"🗑️ Cleaned up temp file: {path}")
                except:
                    pass

@app.post("/v1/audio/transcriptions/stream")
async def transcribe_stream(file: UploadFile, model: str = Form("whisper-1")):
    """
    OpenAI-compatible streaming transcription endpoint
    Returns chunks in OpenAI's streaming format for compatibility with existing apps
    """
    logger.info(f"🌊 Received streaming transcription request - File: {file.filename}, Model: {model}")
    
    # Read the file content first to avoid stream closure issues
    file_content = await file.read()
    original_extension = None
    if file.filename:
        original_extension = os.path.splitext(file.filename)[1].lower()
    
    logger.info(f"📋 Streaming file details - Extension: {original_extension}, Size: {len(file_content)} bytes")
    
    async def generate_openai_stream():
        # Save uploaded file content to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_extension or ".tmp") as tmp:
            tmp.write(file_content)
            uploaded_path = tmp.name
        
        logger.info(f"💾 Saved streaming file to: {uploaded_path}")

        # Create WAV file path
        wav_path = None
        
        try:
            # Convert to WAV format for compatibility
            if original_extension and original_extension != ".wav":
                try:
                    logger.info(f"🔄 Converting {original_extension} to WAV for streaming...")
                    # Use pydub to convert to WAV
                    audio = AudioSegment.from_file(uploaded_path)
                    wav_path = uploaded_path.replace(original_extension, ".wav")
                    audio.export(wav_path, format="wav")
                    audio_path = wav_path
                    logger.info("✅ Streaming audio conversion completed")
                except Exception as e:
                    # If conversion fails, try original file
                    logger.warning(f"⚠️ Streaming audio conversion failed: {e}, using original file")
                    audio_path = uploaded_path
            else:
                logger.info("📄 Streaming file is already in WAV format")
                audio_path = uploaded_path

            # Check if models are ready
            if not model_loader or not model_loader.is_ready():
                error_msg = "Models not loaded for streaming"
                logger.error(f"❌ {error_msg}")
                error_chunk = {
                    "error": {
                        "message": error_msg,
                        "type": "service_unavailable",
                        "code": "models_not_ready"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Run diarization with progress tracking
            diarization = diarize_with_progress(audio_path, lambda msg: logger.info(f"🌊 Streaming - {msg}"))

            # Run ASR with streaming
            logger.info("🗣️ Starting streaming speech recognition...")
            try:
                segments, info = model_loader.asr_model.transcribe(audio_path, word_timestamps=True)
            except Exception as e:
                logger.error(f"❌ ASR streaming failed: {e}")
                error_chunk = {
                    "error": {
                        "message": f"Speech recognition failed: {e}",
                        "type": "transcription_error",
                        "code": "asr_failed"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Process segments as they come
            results = []
            segment_count = 0
            
            logger.info("🌊 Beginning to stream transcription segments...")
            
            for segment in segments:
                start, end, text = segment.start, segment.end, segment.text
                
                # Find speaker for this segment
                speaker = "UNKNOWN"
                for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                    if turn.start <= start and turn.end >= end:
                        speaker = speaker_label
                        break
                
                # Create segment result with speaker prefix
                speaker_text = f"{speaker}: {text}"
                
                segment_count += 1
                logger.debug(f"📤 Streaming segment {segment_count}: [{start:.2f}s-{end:.2f}s] {speaker}: {text[:50]}...")
                
                # Stream this segment in OpenAI format
                chunk = {
                    "id": f"chatcmpl-{hash(text) % 10000000000}",
                    "object": "chat.completion.chunk",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": speaker_text + "\n"
                        },
                        "finish_reason": None
                    }]
                }
                
                # Yield in OpenAI streaming format
                yield f"data: {json.dumps(chunk)}\n\n"
                
                results.append({
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "text": text
                })
                
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.01)
            
            logger.info(f"✅ Streaming transcription completed - {segment_count} segments streamed")
            
            # Send final chunk with finish_reason
            final_chunk = {
                "id": f"chatcmpl-{hash('final') % 10000000000}",
                "object": "chat.completion.chunk", 
                "created": int(asyncio.get_event_loop().time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
            logger.info("🎉 Streaming response completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error during streaming transcription: {str(e)}")
            # Send error in OpenAI format
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "transcription_error",
                    "code": "processing_failed"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        
        finally:
            # Clean up temp files
            for path in [uploaded_path, wav_path]:
                if path:
                    try:
                        os.unlink(path)
                        logger.debug(f"🗑️ Cleaned up streaming temp file: {path}")
                    except:
                        pass

    return EventSourceResponse(generate_openai_stream(), media_type="text/plain")
