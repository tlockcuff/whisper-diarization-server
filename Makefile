# Whisper Diarization Server Makefile
# Handles caching, building, and running the application

.PHONY: help cache build run clean install-deps download-models

# Default target
help:
	@echo "Available targets:"
	@echo "  start-docker   - Quick start (Docker only, no local Python needed)"
	@echo "  start          - Full start with local caching (requires Python)"
	@echo "  cache          - Download and cache all models locally"
	@echo "  build          - Build Docker image with cached models"
	@echo "  run            - Run the application with Docker Compose"
	@echo "  clean          - Clean Docker images and cache"
	@echo "  install-deps   - Install Python dependencies locally"
	@echo "  download-models - Download models to local cache"
	@echo "  health         - Check application health"
	@echo "  logs           - Show application logs"

# Install Python dependencies locally (with virtual environment)
install-deps:
	@echo "📦 Setting up virtual environment and installing dependencies..."
	@if [ ! -d "venv" ]; then python3 -m venv venv; fi
	@. venv/bin/activate && pip install -r requirements.txt
	@echo "✅ Dependencies installed in virtual environment"

# Download models to local cache (using virtual environment)
download-models:
	@echo "📥 Downloading models to local cache..."
	@mkdir -p cache/models cache/huggingface cache/whisper cache/pip
	@if [ -d "venv" ]; then \
		echo "Using virtual environment..."; \
		. venv/bin/activate && python download_models.py; \
	else \
		echo "No virtual environment found, trying system Python..."; \
		python3 download_models.py 2>/dev/null || echo "⚠️ Local model download failed - will download in Docker instead"; \
	fi

# Cache everything (models + pip packages)
cache: download-models
	@echo "📦 Caching pip packages..."
	@if [ -d "venv" ]; then \
		. venv/bin/activate && pip download --dest cache/pip -r requirements.txt; \
	else \
		python3 -m pip download --dest cache/pip -r requirements.txt 2>/dev/null || echo "⚠️ Pip cache failed - will use Docker cache instead"; \
	fi
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

# Docker-only quick start (no local Python required)
start-docker: build run
	@echo "🎉 Docker-only start completed!"
	@echo "📍 Application should be available at http://localhost:8000"
	@echo "🏥 Health check: http://localhost:8000/health"
	@echo "💡 Models will be downloaded during first run in container"

# Development mode (run locally without Docker)
dev: install-deps
	@echo "🔧 Starting development server..."
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
