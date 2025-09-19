.PHONY: help docker-run docker-dev docker-stop logs shell docker-clean download-models-dev docker-build

# Default target
help:
	@echo "🐳 Whisper Diarization Server - Docker-First Setup"
	@echo "================================================="
	@echo ""
	@echo "🚀 Quick Start (Ubuntu 24.04):"
	@echo "  make docker-run          # Start production server"
	@echo "  make docker-dev          # Start development server"
	@echo "  make docker-stop         # Stop all containers"
	@echo ""
	@echo "📦 Docker Commands:"
	@echo "  docker-build             # Build Docker image"
	@echo "  docker-run               # Run production server"
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
	@echo "  - All commands use Docker (no OS dependencies needed)"
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
	docker-compose up -d
	@echo ""
	@echo "✅ Server started! Check logs with: make logs"

docker-dev:
	@echo "🐳 Starting Development Server..."
	@echo "📡 Development server will be at: http://localhost:8001"
	@echo "🔧 Live reload enabled"
	@echo ""
	docker-compose --profile dev up -d
	@echo ""
	@echo "✅ Development server started!"

docker-stop:
	@echo "🛑 Stopping all containers..."
	docker-compose down

logs:
	@echo "📋 Showing application logs..."
	docker-compose logs -f

shell:
	@echo "🐚 Opening shell in container..."
	docker-compose exec whisper-diarization-server /bin/bash

download-models-dev:
	@echo "📥 Downloading models in container..."
	docker-compose run --rm whisper-diarization-server-dev python download_models.py --whisper-model large-v2 --pyannote-model pyannote/speaker-diarization-3.1

docker-build:
	@echo "🔨 Building Docker image..."
	docker-compose build

docker-clean:
	@echo "🧹 Cleaning Docker containers and images..."
	docker-compose down --volumes --remove-orphans
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
