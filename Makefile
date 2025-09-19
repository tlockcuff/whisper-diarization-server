# Whisper Diarization Server Makefile
# Handles caching, building, and running the application

.PHONY: help cache build run clean install-deps download-models

# Default target
help:
	@echo "Available targets:"
	@echo "  cache          - Download and cache all models locally"
	@echo "  build          - Build Docker image with cached models"
	@echo "  run            - Run the application with Docker Compose"
	@echo "  clean          - Clean Docker images and cache"
	@echo "  install-deps   - Install Python dependencies locally"
	@echo "  download-models - Download models to local cache"
	@echo "  health         - Check application health"
	@echo "  logs           - Show application logs"

# Install Python dependencies locally
install-deps:
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt

# Download models to local cache
download-models: install-deps
	@echo "📥 Downloading models to local cache..."
	@mkdir -p cache/models cache/huggingface cache/whisper cache/pip
	python download_models.py

# Cache everything (models + pip packages)
cache: download-models
	@echo "📦 Caching pip packages..."
	pip download --dest cache/pip -r requirements.txt
	@echo "✅ All caching completed!"

# Build Docker image
build:
	@echo "🐳 Building Docker image..."
	docker-compose build

# Build Docker image without cache (force rebuild)
build-no-cache:
	@echo "🐳 Building Docker image (no cache)..."
	docker-compose build --no-cache

# Run the application
run:
	@echo "🚀 Starting application..."
	docker-compose up -d

# Run the application in foreground
run-fg:
	@echo "🚀 Starting application (foreground)..."
	docker-compose up

# Stop the application
stop:
	@echo "🛑 Stopping application..."
	docker-compose down

# Show logs
logs:
	@echo "📋 Showing application logs..."
	docker-compose logs -f stt-server

# Check application health
health:
	@echo "🏥 Checking application health..."
	@curl -s http://localhost:8000/health | jq . || echo "❌ Health check failed"

# Clean up Docker resources
clean:
	@echo "🧹 Cleaning up Docker resources..."
	docker-compose down -v
	docker system prune -f

# Clean cache directories
clean-cache:
	@echo "🧹 Cleaning local cache..."
	rm -rf cache/

# Full clean (Docker + cache)
clean-all: clean clean-cache

# Show cache size
cache-size:
	@echo "📊 Cache directory sizes:"
	@du -sh cache/* 2>/dev/null || echo "No cache found"

# Test setup (without Docker)
test-local:
	@echo "🧪 Testing local setup..."
	python test_setup.py

# Quick start (cache, build, run)
start: cache build run
	@echo "🎉 Quick start completed!"
	@echo "📍 Application should be available at http://localhost:8000"
	@echo "🏥 Health check: http://localhost:8000/health"

# Development mode (run locally without Docker)
dev: install-deps
	@echo "🔧 Starting development server..."
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
