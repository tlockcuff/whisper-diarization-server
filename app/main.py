from fastapi import FastAPI, UploadFile, Form
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import tempfile
import shutil
import os

app = FastAPI()

# Load models at startup using environment variables
asr_model_name = os.getenv("ASR_MODEL", "base")
diarization_model_name = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
hf_token = os.getenv("HF_TOKEN")

asr_model = WhisperModel(asr_model_name, device="cuda", compute_type="float16")
if hf_token:
    diarization_pipeline = Pipeline.from_pretrained(diarization_model_name, use_auth_token=hf_token)
else:
    diarization_pipeline = Pipeline.from_pretrained(diarization_model_name)

@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile, model: str = Form("whisper-1")):
    # Get file extension from uploaded file
    file_extension = None
    if file.filename:
        file_extension = os.path.splitext(file.filename)[1]
    if not file_extension:
        file_extension = ".wav"  # Default fallback
    
    # Save upload to temp file with proper extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        shutil.copyfileobj(file.file, tmp)
        audio_path = tmp.name

    try:
        # Diarization
        diarization = diarization_pipeline(audio_path)

        # Run ASR
        segments, _ = asr_model.transcribe(audio_path)

        # Merge diarization with ASR output
        results = []
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

        # Format like OpenAI Whisper
        response_text = "\n".join([f"{r['speaker']}: {r['text']}" for r in results])
        return {
            "text": response_text,
            "segments": results
        }
    
    finally:
        # Clean up temp file
        try:
            os.unlink(audio_path)
        except:
            pass
