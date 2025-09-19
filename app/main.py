"""Main FastAPI application for Whisper Diarization Server."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

from .config import settings
from .hardware_detector import detect_hardware, get_device_info
from .model_loader import get_models, unload_models, get_model_info


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.logging.level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if settings.logging.format == "text"
    else "%(asctime)s %(name)s %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Whisper Diarization Server...")

    # Detect hardware
    hardware_info = detect_hardware()
    logger.info(f"Hardware detected: {hardware_info}")

    # Load models
    try:
        logger.info("Loading models...")
        get_models()
        model_info = get_model_info()
        logger.info(f"Models loaded: {model_info}")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down server...")
    unload_models()


# Create FastAPI app
app = FastAPI(
    title="Whisper Diarization Server",
    description="OpenAI-compatible API for speech recognition with speaker diarization",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Whisper Diarization Server",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "transcriptions": "/v1/audio/transcriptions",
            "health": "/health",
            "models": "/models"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "device": get_device_info()
    }


@app.get("/models")
async def get_models_info():
    """Get information about loaded models."""
    return {
        "models": get_model_info(),
        "hardware": get_device_info()
    }


async def process_audio_streaming(
    file: UploadFile,
    model: str = "whisper-1",
    response_format: str = "json",
    temperature: float = 0.0,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """Process audio with streaming response."""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Check file format
        allowed_formats = settings.audio.supported_formats
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in allowed_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed: {allowed_formats}"
            )

        # Read file content
        content = await file.read()

        # Load models
        whisper_model, diarization_pipeline = get_models()

        # Process audio (this would be implemented in the endpoint handler)
        # For now, return a mock streaming response
        yield '{"status": "processing", "progress": 0}\n'

        # Simulate processing
        await asyncio.sleep(1)

        yield '{"status": "processing", "progress": 50}\n'

        await asyncio.sleep(1)

        yield '{"status": "completed", "progress": 100}\n'

        # Mock result
        result = {
            "text": "This is a mock transcription result.",
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "This is a mock segment.",
                    "speaker": "Speaker 1"
                }
            ]
        }

        yield f'data: {{"result": {result}}}\n\n'

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        yield f'data: {{"error": "{str(e)}"}}\n\n'


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    language: Optional[str] = Form(default=None),
    stream: bool = Form(default=False),
):
    """OpenAI-compatible transcription endpoint."""
    try:
        if stream and settings.enable_streaming:
            return StreamingResponse(
                process_audio_streaming(
                    file=file,
                    model=model,
                    response_format=response_format,
                    temperature=temperature,
                    language=language,
                    prompt=prompt,
                ),
                media_type="text/plain"
            )

        # Non-streaming response
        content = await file.read()

        # Here you would implement the actual transcription logic
        # For now, return a mock response
        return {
            "text": "This is a mock transcription result.",
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "This is a mock segment.",
                    "speaker": "Speaker 1"
                }
            ]
        }

    except Exception as e:
        logger.error(f"Error in transcription endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.debug,
        log_level=settings.logging.level.lower()
    )