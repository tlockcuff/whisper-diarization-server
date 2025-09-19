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

# Set up environment variables with defaults
ARG ASR_MODEL=base
ARG DIARIZATION_MODEL=pyannote/speaker-diarization-3.1
ARG HF_TOKEN

# Pre-download Whisper + Pyannote models (optional for full offline)
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('$ASR_MODEL', download_root='/models')"
RUN if [ -n "$HF_TOKEN" ]; then python3 -c "from pyannote.audio import Pipeline; Pipeline.from_pretrained('$DIARIZATION_MODEL', use_auth_token='$HF_TOKEN')"; fi

# Set environment variables for runtime
ENV TRANSFORMERS_CACHE=/models
ENV HF_HOME=/models
ENV ASR_MODEL=${ASR_MODEL}
ENV DIARIZATION_MODEL=${DIARIZATION_MODEL}
ENV HF_TOKEN=${HF_TOKEN}

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
