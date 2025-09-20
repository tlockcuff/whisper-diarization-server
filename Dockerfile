FROM nvidia/cuda:12.4.1-cudnn9-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    cython3 \
    sox \
    libsox-fmt-all \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip3 install --cache-dir /root/.cache/pip --no-cache-dir -r requirements.txt

# Clone and install whisper-diarization
RUN git clone https://github.com/MahmoudAshraf97/whisper-diarization.git /whisper-diarization
WORKDIR /whisper-diarization
# Install CUDA runtime dependencies
RUN pip3 install --cache-dir /root/.cache/pip nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12
RUN pip3 install --cache-dir /root/.cache/pip numpy
RUN pip3 install --cache-dir /root/.cache/pip -c constraints.txt -r requirements.txt
# Clone ctc-forced-aligner
RUN git clone https://github.com/MahmoudAshraf97/ctc-forced-aligner.git /ctc-forced-aligner
WORKDIR /ctc-forced-aligner
# Install dependencies for ctc_forced_aligner
RUN pip3 install --cache-dir /root/.cache/pip -r requirements.txt
# Build and install local ctc_forced_aligner from source
RUN python3 setup.py build_ext --inplace
RUN pip3 install --cache-dir /root/.cache/pip .
WORKDIR /whisper-diarization
# Download models on build if possible, but may need runtime
RUN python3 -c "import whisper; whisper.load_model('base')" || true
# Test the import to ensure it works
RUN PYTHONPATH=/ctc-forced-aligner:$PYTHONPATH python3 -c "from ctc_forced_aligner import generate_emissions; print('ctc_forced_aligner import successful')" || exit 1
WORKDIR /app

# Copy app code
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Run the server
CMD ["sh", "-c", "export PYTHONPATH=/whisper-diarization:/ctc-forced-aligner:${PYTHONPATH}; uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"]
