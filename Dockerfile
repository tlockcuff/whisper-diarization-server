FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    git python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app/ ./app

# Pre-download Whisper + Pyannote models (optional for full offline)
# Example for Whisper large-v3
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', download_root='/models')"
# Example for Pyannote diarization model
RUN python3 -c "from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding; PretrainedSpeakerEmbedding('speechbrain/spkrec-ecapa-voxceleb', device='cpu')"

ENV TRANSFORMERS_CACHE=/models
ENV HF_HOME=/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
