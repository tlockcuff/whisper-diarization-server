FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    cython3 \
    sox \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip3 install --cache-dir /root/.cache/pip --no-cache-dir -r requirements.txt

# Clone and install whisper-diarization
RUN git clone https://github.com/MahmoudAshraf97/whisper-diarization.git /whisper-diarization
WORKDIR /whisper-diarization
RUN pip3 install --cache-dir /root/.cache/pip numpy
RUN pip3 install --cache-dir /root/.cache/pip -c constraints.txt -r requirements.txt
# Download models on build if possible, but may need runtime
RUN python3 -c "import whisper; whisper.load_model('base')" || true
WORKDIR /app

# Copy app code
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
