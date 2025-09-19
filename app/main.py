from fastapi import FastAPI, UploadFile, Form
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import tempfile
import shutil
import os
from pydub import AudioSegment

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
    original_extension = None
    if file.filename:
        original_extension = os.path.splitext(file.filename)[1].lower()
    
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=original_extension or ".tmp") as tmp:
        shutil.copyfileobj(file.file, tmp)
        uploaded_path = tmp.name

    # Create WAV file path
    wav_path = None
    
    try:
        # Convert to WAV format for compatibility
        if original_extension and original_extension != ".wav":
            try:
                # Use pydub to convert to WAV
                audio = AudioSegment.from_file(uploaded_path)
                wav_path = uploaded_path.replace(original_extension, ".wav")
                audio.export(wav_path, format="wav")
                audio_path = wav_path
            except Exception as e:
                # If conversion fails, try original file
                print(f"Audio conversion failed: {e}, using original file")
                audio_path = uploaded_path
        else:
            audio_path = uploaded_path

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
        # Clean up temp files
        for path in [uploaded_path, wav_path]:
            if path:
                try:
                    os.unlink(path)
                except:
                    pass
