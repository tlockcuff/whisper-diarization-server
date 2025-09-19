from fastapi import FastAPI, UploadFile, Form
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import tempfile
import shutil
import os
import json
import asyncio
import logging
import warnings
from pydub import AudioSegment
from sse_starlette.sse import EventSourceResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress specific torchaudio warnings
import torchaudio
warnings.filterwarnings("ignore", module="torchaudio")

app = FastAPI()

# Load models at startup using environment variables
asr_model_name = os.getenv("ASR_MODEL", "base")
diarization_model_name = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
hf_token = os.getenv("HF_TOKEN")

logger.info(f"🚀 Starting whisper-diarization server")
logger.info(f"📝 ASR Model: {asr_model_name}")
logger.info(f"🎤 Diarization Model: {diarization_model_name}")
logger.info(f"🔐 HF Token: {'✅ Provided' if hf_token else '❌ Not provided'}")

logger.info("🔄 Loading ASR model...")
asr_model = WhisperModel(asr_model_name, device="cuda", compute_type="float16")
logger.info("✅ ASR model loaded successfully")

logger.info("🔄 Loading diarization pipeline...")
if hf_token:
    diarization_pipeline = Pipeline.from_pretrained(diarization_model_name, use_auth_token=hf_token)
else:
    diarization_pipeline = Pipeline.from_pretrained(diarization_model_name)
logger.info("✅ Diarization pipeline loaded successfully")

logger.info("🎉 Server initialization complete! Ready to process audio files.")

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

        # Diarization
        logger.info("🎤 Starting speaker diarization...")
        diarization = diarization_pipeline(audio_path)
        logger.info("✅ Speaker diarization completed")

        # Run ASR
        logger.info("🗣️ Starting speech recognition...")
        segments, _ = asr_model.transcribe(audio_path)
        logger.info("✅ Speech recognition completed")

        # Merge diarization with ASR output
        logger.info("🔗 Merging diarization with transcription...")
        results = []
        segment_count = 0
        for segment in segments:
            start, end, text = segment.start, segment.end, segment.text
            speaker = "UNKNOWN"
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

        logger.info(f"✅ Processing completed - {segment_count} segments processed")

        # Format like OpenAI Whisper
        response_text = "\n".join([f"{r['speaker']}: {r['text']}" for r in results])
        
        logger.info(f"📤 Returning transcription result ({len(response_text)} characters)")
        return {
            "text": response_text,
            "segments": results
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

            # Run diarization (this happens first and takes time)
            logger.info("🎤 Starting streaming speaker diarization...")
            diarization = diarization_pipeline(audio_path)
            logger.info("✅ Streaming speaker diarization completed")

            # Run ASR with streaming
            logger.info("🗣️ Starting streaming speech recognition...")
            segments, info = asr_model.transcribe(audio_path, word_timestamps=True)
            
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
