# Use PyTorch official image which includes properly configured CUDA + cuDNN
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# System deps
RUN apt-get update && apt-get install -y \
    git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
WORKDIR /app

# Create cache directories
RUN mkdir -p /app/cache/models /app/cache/huggingface /app/cache/whisper /app/cache/pip

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Use local pip cache if available, otherwise download
COPY cache/pip/ /tmp/pip-cache/ 2>/dev/null || echo "No local pip cache found"
RUN pip install --cache-dir /tmp/pip-cache --find-links /tmp/pip-cache -r requirements.txt

# Copy application code
COPY app/ ./app

# Copy model download script
COPY download_models.py .

# Copy cached models if they exist (this will be skipped if cache doesn't exist)
COPY cache/ /app/cache/ 2>/dev/null || echo "No local model cache found"

# Set up environment variables with defaults
ARG ASR_MODEL=base
ARG DIARIZATION_MODEL=pyannote/speaker-diarization@2.1
ARG HF_TOKEN

# Set environment variables for runtime (point to cache directories)
ENV TRANSFORMERS_CACHE=/app/cache/huggingface
ENV HF_HOME=/app/cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/app/cache/huggingface
ENV ASR_MODEL=${ASR_MODEL}
ENV DIARIZATION_MODEL=${DIARIZATION_MODEL}
ENV HF_TOKEN=${HF_TOKEN}

# Pre-download models if not cached, or use cached versions
RUN python -c "
import os
from pathlib import Path
from faster_whisper import WhisperModel

# Use cached Whisper model if available
cache_dir = '/app/cache/whisper'
try:
    model = WhisperModel('$ASR_MODEL', download_root=cache_dir)
    print(f'✅ Loaded Whisper model: $ASR_MODEL')
except Exception as e:
    print(f'⚠️ Failed to load Whisper model: {e}')
"

RUN if [ -n "$HF_TOKEN" ]; then python -c "
import os
from pyannote.audio import Pipeline

try:
    pipeline = Pipeline.from_pretrained('$DIARIZATION_MODEL', use_auth_token='$HF_TOKEN')
    print(f'✅ Loaded diarization model: $DIARIZATION_MODEL')
except Exception as e:
    print(f'⚠️ Failed to load diarization model: {e}')
"; fi

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
