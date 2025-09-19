# Build custom PyTorch with RTX 5060 Ti support (compute capability 8.9)
ARG CUDA_IMAGE_TAG=12.4.1-cudnn-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_IMAGE_TAG}

# Configure timezone to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# CUDA/cuDNN environment variables for better compatibility
ENV CUDA_MODULE_LOADING=LAZY
ENV CUDNN_LOGINFO_DBG=0
ENV CUDNN_LOGERR_DBG=0
# Force CUDA architecture compatibility for RTX 5060 Ti (compute capability 8.9)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV CUDA_LAUNCH_BLOCKING=1

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    tzdata \
    build-essential \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
WORKDIR /app

# Create cache directories
RUN mkdir -p /app/cache/models /app/cache/huggingface /app/cache/whisper /app/cache/pip

# Install PyTorch with CUDA support (configurable CUDA wheel channel)
ARG TORCH_CUDA_TAG=cu124
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG}

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
RUN python3 preload_models.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
