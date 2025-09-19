# Whisper Diarization Server v2.0

A robust, FastAPI server that provides speaker diarization and speech recognition using OpenAI Whisper and pyannote.audio.

Supports OpenAI API REST Response Streaming (/v1/audio/transcriptions)

## Features

- 🚀 **OpenAI API Compatible** - Drop-in replacement for OpenAI's transcription API
- 🎯 **Speaker Diarization** - Automatic speaker identification and segmentation
- 📡 **Streaming Support** - Real-time streaming responses for better UX
- 🏃‍♂️ **GPU Acceleration** - Optimized for NVIDIA GPUs with automatic fallback to CPU
- 🐳 **Docker Support** - Easy deployment with Docker and Docker Compose
- 🔧 **Model Caching** - Fast startup times with intelligent model caching
- 📊 **Hardware Detection** - Automatic hardware optimization and configuration

## Hardware Requirements

### Minimum
- CPU: 4-core processor
- RAM: 8GB
- Storage: 10GB for models and cache

### Recommended
- GPU: NVIDIA GPU with 4GB+ VRAM (RTX 3060 or better)
- RAM: 16GB
- Storage: 20GB SSD

### Expected Hardware (as mentioned)
- 2x Nvidia RTX 5060 Ti (for high-performance deployment)

## Quick Start (Docker - Recommended)

### 1. Install Docker

```bash
# Install Docker on Ubuntu 24.04
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo apt install docker-compose-plugin
```

### 2. Clone and Setup

```bash
git clone <repository-url>
cd whisper-diarization-server
```

### 3. Start the Server

```bash
# Start production server (GPU-enabled)
make docker-run

# Or start development server (CPU, live reload)
make docker-dev
```

The server will be available at:
- **Production**: http://localhost:8000
- **Development**: http://localhost:8001

### 4. Download Models (if needed)

```bash
# Download models in container
make download-models-dev
```

### 5. Check Status

```bash
# View logs
make logs

# Health check
curl http://localhost:8000/health
```

## Alternative: Native Installation (Not Recommended)

If you prefer not to use Docker, see the troubleshooting section for Ubuntu 24.04 issues.

**Note:** Makefile commands automatically use the virtual environment, so you don't need to manually activate it for `make` commands.

## API Usage

### OpenAI-Compatible Endpoints

#### Transcribe Audio

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
     -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@audio.mp3" \
     -F "model=whisper-1" \
     -F "response_format=json"
```

**Response:**
```json
{
  "text": "This is the transcribed text with speaker identification.",
  "segments": [
    {
      "start": 0.0,
      "end": 5.0,
      "text": "Hello, this is speaker one.",
      "speaker": "Speaker 1"
    },
    {
      "start": 5.0,
      "end": 10.0,
      "text": "Hi there, this is speaker two.",
      "speaker": "Speaker 2"
    }
  ]
}
```

#### Streaming Transcription

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
     -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@audio.mp3" \
     -F "model=whisper-1" \
     -F "stream=true"
```

**Streaming Response:**
```
{"status": "processing", "progress": 0}
{"status": "processing", "progress": 50}
{"status": "completed", "progress": 100}
data: {"result": {"text": "...", "segments": [...]}}
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Models Information

```bash
curl http://localhost:8000/models
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | `0.0.0.0` | Server host |
| `SERVER_PORT` | `8000` | Server port |
| `WHISPER_MODEL` | `large-v2` | Whisper model to use |
| `PYANNOTE_MODEL` | `pyannote/speaker-diarization-3.1` | Diarization model |
| `USE_GPU` | `true` | Enable GPU acceleration |
| `HUGGINGFACE_TOKEN` | - | Required for Pyannote models |
| `CACHE_DIR` | `./cache` | Model cache directory |

### Dependency Versions
- PyTorch: 2.4.1+
- Python: 3.8+
- CUDA: 11.8+ (for GPU support)

### Models

#### Whisper Models
- `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`

#### Pyannote Models
- `pyannote/speaker-diarization-3.1` (requires Hugging Face token)
- `pyannote/speaker-diarization` (legacy)

## Docker Deployment

### Build and Run

```bash
# Build image
make docker-build

# Run with Docker Compose
make docker-run

# Run with Redis caching
make docker-run-with-redis
```

### Docker Commands

```bash
# View logs
make logs

# Stop containers
make docker-stop

# Open shell
make shell
```

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest pytest-asyncio black flake8

# Run tests
make test

# Format code
black app/
flake8 app/
```

### Makefile Commands

```bash
make help          # Show all available commands
make setup         # Full setup (install + download models)
make clean         # Clean cache and temporary files
make info          # Show system information
make check-deps    # Check dependencies
```

## Performance Optimization

### GPU Setup
- Install NVIDIA drivers and CUDA toolkit
- Set `USE_GPU=true` in environment
- Use `nvidia-smi` to monitor GPU usage

### Model Caching
- Models are cached locally for faster startup
- Use SSD storage for better performance
- Set `CACHE_DIR` to a fast storage location

### Memory Management
- Monitor memory usage with `htop` or `nvtop`
- Adjust `MAX_WORKERS` based on available RAM
- Use `make clean` to free up space

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `MAX_WORKERS` in configuration
   - Use smaller Whisper model (e.g., `medium` instead of `large-v2`)
   - Restart the server to clear GPU cache

2. **Model Download Issues**
   - Check internet connection
   - Verify Hugging Face token for Pyannote models
   - Check available disk space in cache directory

3. **GPU Not Detected**
   - Verify NVIDIA drivers are installed
   - Check `nvidia-smi` command works
   - Set `USE_GPU=false` to force CPU mode

### Common Issues

**Docker not running:**
```bash
# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker
```

**Permission denied with Docker:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Then log out and back in
```

**Out of disk space:**
```bash
# Clean up Docker
make docker-clean
```

**Models not downloading:**
```bash
# Download models manually in container
make download-models-dev
```

**Port already in use:**
```bash
# Check what's using the port
sudo netstat -tulpn | grep :8000
# Or use different ports in docker-compose.yml
```

### Getting Help

```bash
# Check system info
make info

# Test hardware detection
make check-deps

# Test model loading (after activating venv)
source venv/bin/activate
python -c "from app.model_loader import get_models; get_models()"
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section above
- Review the API documentation at `/docs` when server is running