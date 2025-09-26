FROM onerahmet/openai-whisper-asr-webservice:latest-gpu

# Install compatible PyTorch for RTX 5060 Ti (sm_120, CUDA 12.8)
RUN pip uninstall -y torch torchaudio torchvision
RUN pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128

# Ensure pyannote.audio is compatible (downgrade to avoid version mismatch warnings)
RUN pip install pyannote.audio==2.1.1

# Clean up pip cache to reduce image size
RUN pip cache purge

# Set environment variables for runtime
ENV ASR_MODEL=large-v3
ENV ASR_ENGINE=whisperx
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=true
ENV ASR_DEVICE=cuda