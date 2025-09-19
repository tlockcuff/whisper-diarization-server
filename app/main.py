from fastapi import FastAPI, UploadFile, Form
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import tempfile
import shutil
import os

app = FastAPI()

# Load models at startup
asr_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile, model: str = Form("whisper-1")):
    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        audio_path = tmp.name

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
