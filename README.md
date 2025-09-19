# Whisper Diarization Server v2.0

A robust, hardware-aware FastAPI server that provides speaker diarization and speech recognition using OpenAI Whisper and pyannote.audio.

## ✨ New in v2.0
- **Hardware Detection**: Automatic GPU compatibility checking
- **Modular Architecture**: Clean separation of concerns
- **Robust Error Handling**: Graceful fallbacks and recovery
- **Configuration Management**: Flexible environment-based configuration
- **Enhanced Monitoring**: Detailed health checks and diagnostics

## Features

- 🎤 **Speaker Diarization**: Identify different speakers in audio files
- 🗣️ **Speech Recognition**: Transcribe audio with speaker labels
- 🌊 **Streaming Support**: Real-time transcription streaming
- ⚡ **Progress Callbacks**: Real-time progress updates during processing
- 🔧 **CPU/GPU Support**: Automatic fallback to CPU if GPU unavailable
- 📦 **Local Caching**: Cache models locally for faster builds and offline usage
- 🐳 **Docker Support**: Containerized deployment with CUDA support

## Architecture

### Core Modules

1. **Hardware Detection** (`app/hardware_detector.py`)
   - Automatic GPU compatibility checking
   - Compute capability detection
   - Memory and performance analysis
   - Compatibility recommendations

2. **Model Loading** (`app/model_loader.py`)
   - Robust model loading with fallbacks
   - Hardware-aware device selection
   - Automatic retry mechanisms
   - Memory management

3. **Configuration** (`app/config.py`)
   - Environment-based configuration
   - Validation and error checking
   - Flexible parameter management

4. **Main Application** (`app/main.py`)
   - FastAPI server with enhanced endpoints
   - Comprehensive error handling
   - Health monitoring and diagnostics

## Quick Start

### Using Make (Recommended)

```bash
# Download and cache models, build, and run
make start

# Or step by step:
make cache          # Download models to local cache
make build          # Build Docker image
make run            # Start the service
```

### Manual Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   export HF_TOKEN="your_huggingface_token"  # Required for some models
   export ASR_MODEL="base"                   # whisper model size
   export DIARIZATION_MODEL="pyannote/speaker-diarization@2.1"
   ```

3. **Download models (optional but recommended):**
   ```bash
   python download_models.py
   ```

4. **Run with Docker:**
   ```bash
   docker-compose up
   ```

## Caching System

The application implements comprehensive caching to speed up builds and reduce network dependencies:

### Cache Structure
```
cache/
├── huggingface/     # HuggingFace models and tokenizers
├── whisper/         # OpenAI Whisper models
├── models/          # Other model artifacts
└── pip/             # Python packages
```

### Benefits
- **Faster Docker builds**: Models cached locally, no re-download
- **Offline capability**: Run without internet after initial cache
- **Bandwidth savings**: Download models once, reuse across builds
- **Build reproducibility**: Same models across environments

### Cache Management

```bash
# Cache all models and dependencies
make cache

# Check cache size
make cache-size

# Clean cache
make clean-cache

# Force rebuild without cache
make build-no-cache
```

## API Endpoints

### Health Check
```bash
GET /health
```
Returns server status and GPU information.

### Transcription (Non-streaming)
```bash
POST /v1/audio/transcriptions
Content-Type: multipart/form-data

file: <audio_file>
model: whisper-1
```

### Transcription (Streaming)
```bash
POST /v1/audio/transcriptions/stream
Content-Type: multipart/form-data

file: <audio_file>
model: whisper-1
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_MODEL` | `base` | Whisper model size (tiny, base, small, medium, large) |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization@2.1` | Diarization model |
| `HF_TOKEN` | - | HuggingFace authentication token |
| `CACHE_DIR` | `/app/cache` | Directory for model caching |

### Model Selection

**Whisper Models** (speed vs accuracy trade-off):
- `tiny`: Fastest, lowest accuracy
- `base`: Good balance (recommended)
- `small`: Better accuracy, slower
- `medium`: High accuracy, much slower
- `large`: Best accuracy, slowest

**Diarization Models**:
- `pyannote/speaker-diarization@2.1`: Faster, good accuracy (recommended)
- `pyannote/speaker-diarization-3.1`: Better accuracy, slower

## Performance Optimization

1. **Use smaller models** for faster processing:
   ```bash
   export ASR_MODEL="base"
   export DIARIZATION_MODEL="pyannote/speaker-diarization@2.1"
   ```

2. **Pre-cache models** to avoid download delays:
   ```bash
   make cache
   ```

3. **Use GPU** when available (automatic detection with CPU fallback)

4. **Monitor processing** with real-time progress callbacks

## Docker GPU Support

For NVIDIA GPU support:

1. **Install NVIDIA Container Toolkit**
2. **Configure Docker to use GPU**
3. **Use the provided docker-compose.yml** (includes GPU configuration)

The application automatically detects and uses GPU when available, with graceful CPU fallback.

## Development

### Local Development
```bash
# Install dependencies
make install-deps

# Run development server
make dev

# Test setup
make test-local
```

### Docker Development
```bash
# Build and run
docker-compose up --build

# View logs
make logs

# Health check
make health
```

## Troubleshooting

### CUDA/cuDNN Issues
- The Docker image uses PyTorch base image with pre-configured CUDA
- CPU fallback is automatic if GPU unavailable
- Check health endpoint for GPU status

### Model Download Issues
- Ensure `HF_TOKEN` is set for restricted models
- Check internet connectivity
- Use `make cache` to pre-download models

### Performance Issues
- Use smaller models for faster processing
- Ensure GPU is available and properly configured
- Monitor with health endpoint and logs

## License

This project is open source. Please check individual model licenses for commercial usage restrictions.
