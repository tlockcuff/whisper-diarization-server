.PHONY: help docker-run docker-run-cpu docker-dev docker-stop logs shell docker-clean download-models-dev docker-build docker-build-cpu

# Default target
help:
	@echo "🐳 Whisper Diarization Server - Docker-First Setup"
	@echo "================================================="
	@echo ""
	@echo "🚀 Quick Start (Ubuntu 24.04):"
	@echo "  make docker-run          # Start production server (GPU)"
	@echo "  make docker-run-cpu      # Start CPU-only server"
	@echo "  make docker-dev          # Start development server"
	@echo "  make docker-stop         # Stop all containers"
	@echo ""
	@echo "📦 Docker Commands:"
	@echo "  docker-build             # Build GPU Docker image"
	@echo "  docker-build-cpu         # Build CPU-only image"
	@echo "  docker-run               # Run production server (GPU)"
	@echo "  docker-run-cpu           # Run CPU-only server"
	@echo "  docker-dev               # Run development server"
	@echo "  docker-stop              # Stop all containers"
	@echo "  logs                     # View application logs"
	@echo "  shell                    # Open shell in container"
	@echo "  docker-clean             # Clean Docker containers and images"
	@echo ""
	@echo "🔧 Development:"
	@echo "  download-models-dev      # Download models in container"
	@echo ""
	@echo "💡 Tips:"
	@echo "  - Use docker-run-cpu if you don't have NVIDIA GPU"
	@echo "  - Models are cached in ./cache directory"
	@echo "  - Use 'make logs' to monitor the server"
	@echo "  - Server will be available at http://localhost:8000"
	@echo ""
	@echo "For more information, see README.md"

# Docker commands
docker-run:
	@echo "🐳 Starting Whisper Diarization Server..."
	@echo "📡 Server will be available at: http://localhost:8000"
	@echo "📊 Health check: http://localhost:8000/health"
	@echo "📝 API docs: http://localhost:8000/docs"
	@echo ""
	@echo "🔄 Building and starting containers..."
	@echo "⚠️  If this fails, try: make docker-run-cpu"
	docker compose up -d
	@echo ""
	@echo "✅ Server started! Check logs with: make logs"

docker-dev:
	@echo "🐳 Starting Development Server..."
	@echo "📡 Development server will be at: http://localhost:8001"
	@echo "🔧 Live reload enabled"
	@echo ""
	docker compose --profile dev up -d
	@echo ""
	@echo "✅ Development server started!"

docker-stop:
	@echo "🛑 Stopping all containers..."
	docker compose down

logs:
	@echo "📋 Showing application logs..."
	docker compose logs -f

shell:
	@echo "🐚 Opening shell in container..."
	docker compose exec whisper-diarization-server /bin/bash

download-models-dev:
	@echo "📥 Downloading models in container..."
	docker compose run --rm whisper-diarization-server python download_models.py --whisper-model large-v2 --pyannote-model pyannote/speaker-diarization-3.1

docker-build:
	@echo "🔨 Building Docker image..."
	docker compose build

docker-build-cpu:
	@echo "🔨 Building CPU-only Docker image..."
	docker build -f Dockerfile.fallback -t whisper-diarization-server:cpu .

docker-run-cpu:
	@echo "🐳 Starting CPU-only Whisper Diarization Server..."
	@echo "📡 Server will be available at: http://localhost:8000"
	@echo "⚠️  CPU-only mode (no GPU acceleration)"
	@echo ""
	docker run -d --name whisper-server-cpu \
		-p 8000:8000 \
		-v ./cache:/app/cache \
		-e USE_GPU=false \
		-e WHISPER_MODEL=medium \
		whisper-diarization-server:cpu
	@echo ""
	@echo "✅ CPU server started! Check logs with: docker logs -f whisper-server-cpu"

docker-clean:
	@echo "🧹 Cleaning Docker containers and images..."
	docker compose down --volumes --remove-orphans
	docker system prune -f
	docker image prune -f

# Legacy commands for compatibility (redirect to Docker)
install:
	@echo "⚠️  Please use Docker instead!"
	@echo "Run: make docker-run"
	@exit 1

setup:
	@echo "⚠️  Please use Docker instead!"
	@echo "Run: make docker-run"
	@exit 1

run:
	@echo "⚠️  Please use Docker instead!"
	@echo "Run: make docker-run"
	@exit 1

dev:
	@echo "⚠️  Please use Docker instead!"
	@echo "Run: make docker-dev"
	@exit 1
