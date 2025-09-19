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
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('${ASR_MODEL}', download_root='/models')"
RUN python3 -c "from pyannote.audio import Pipeline; Pipeline.from_pretrained('${DIARIZATION_MODEL}', use_auth_token='${HF_TOKEN}')"

ENV TRANSFORMERS_CACHE=/models
ENV HF_HOME=/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
