# Use PyTorch official image with CUDA 12.x and cuDNN 9.x for latest package compatibility
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Configure timezone to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# CUDA/cuDNN environment variables for better compatibility
ENV CUDA_MODULE_LOADING=LAZY
ENV CUDNN_LOGINFO_DBG=0
ENV CUDNN_LOGERR_DBG=0

# System deps
RUN apt-get update && apt-get install -y \
    tzdata \
    git ffmpeg \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
WORKDIR /app

# Create cache directories
RUN mkdir -p /app/cache/models /app/cache/huggingface /app/cache/whisper /app/cache/pip

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python packages (will use cache if available during build)
RUN pip install -r requirements.txt

# Copy application code
COPY app/ ./app

# Copy model download script
COPY download_models.py .

# Copy model preload script
COPY preload_models.py .

# Copy all cache directories (Docker will use .dockerignore to handle missing files)
COPY cache/ /app/cache/

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
RUN python preload_models.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
