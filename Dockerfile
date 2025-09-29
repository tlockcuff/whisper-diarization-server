ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04

##############################
# Build custom PyTorch wheels #
##############################

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS pytorch-builder

ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=v2.4.0
ARG TORCHAUDIO_VERSION=v2.4.0

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    build-essential \
    libffi-dev \
    libssl-dev \
    libsndfile1 \
    libsox-dev \
    libsox-fmt-all \
    cmake \
    ninja-build \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m pip install --upgrade pip setuptools wheel

ENV TORCH_CUDA_ARCH_LIST="12.0" \
    USE_CUDA=1 \
    USE_CUDNN=1 \
    USE_FFMPEG=1 \
    BUILD_TEST=0 \
    MAX_JOBS=8

# Build PyTorch
WORKDIR /opt/src
RUN git clone --recursive --branch ${PYTORCH_VERSION} --depth 1 https://github.com/pytorch/pytorch.git
WORKDIR /opt/src/pytorch
RUN python${PYTHON_VERSION} -m pip install -r requirements.txt
RUN python${PYTHON_VERSION} setup.py bdist_wheel

# Build torchaudio to match PyTorch
WORKDIR /opt/src
RUN git clone --recursive --branch ${TORCHAUDIO_VERSION} --depth 1 https://github.com/pytorch/audio.git torchaudio
WORKDIR /opt/src/torchaudio
ENV BUILD_SOX=1
RUN python${PYTHON_VERSION} -m pip install -r requirements.txt
RUN python${PYTHON_VERSION} setup.py bdist_wheel

#########################
# Runtime application    #
#########################

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m pip install --upgrade pip

ENV APP_HOME=/app
WORKDIR ${APP_HOME}

# Create virtual environment for application
RUN python${PYTHON_VERSION} -m venv .venv
ENV PATH="${APP_HOME}/.venv/bin:${PATH}"

# Copy built wheels from builder
COPY --from=pytorch-builder /opt/src/pytorch/dist/torch-*.whl /tmp/
COPY --from=pytorch-builder /opt/src/torchaudio/dist/torchaudio-*.whl /tmp/

# Install core ML dependencies
RUN pip install /tmp/torch-*.whl && \
    pip install /tmp/torchaudio-*.whl && \
    rm /tmp/torch-*.whl /tmp/torchaudio-*.whl

# Install application dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir whisperx --no-deps

# Copy application source
COPY app ./app

# Ensure models cache persists between runs (optional volume)
ENV TRANSCRIBE_DEVICE=cuda \
    WHISPER_MODEL_SIZE=large-v3 \
    WHISPER_COMPUTE_TYPE=float16

EXPOSE 9000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]

