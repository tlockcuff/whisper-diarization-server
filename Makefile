.PHONY: help install setup download-models run dev test clean docker-build docker-run docker-stop logs shell

# Default target
help:
	@echo "Whisper Diarization Server - Available commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  install         Install Python dependencies"
	@echo "  setup           Full setup (install + download models)"
	@echo "  download-models Download and cache ML models"
	@echo ""
	@echo "Development:"
	@echo "  run             Run the server in production mode"
	@echo "  dev             Run the server in development mode"
	@echo "  test            Run tests"
	@echo "  clean           Clean cache and temporary files"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build    Build Docker image"
	@echo "  docker-run      Run with Docker Compose"
	@echo "  docker-stop     Stop Docker containers"
	@echo "  logs            View application logs"
	@echo "  shell           Open shell in running container"
	@echo ""
	@echo "For more information, see README.md"

# Setup and installation
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt

setup: install download-models

download-models:
	@echo "Downloading models..."
	python download_models.py --whisper-model large-v2 --pyannote-model pyannote/speaker-diarization-3.1

# Development commands
run:
	@echo "Starting server in production mode..."
	python app/main.py

dev:
	@echo "Starting server in development mode..."
	python app/main.py --debug

test:
	@echo "Running tests..."
	pytest

# Cleanup
clean:
	@echo "Cleaning cache and temporary files..."
	rm -rf cache/models/*
	rm -rf cache/whisper/*
	rm -rf cache/huggingface/*
	rm -rf cache/pip/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker-compose build

docker-run:
	@echo "Starting services with Docker Compose..."
	docker-compose up -d

docker-run-with-redis:
	@echo "Starting services with Redis using Docker Compose..."
	docker-compose --profile with-redis up -d

docker-stop:
	@echo "Stopping Docker containers..."
	docker-compose down

logs:
	@echo "Viewing application logs..."
	docker-compose logs -f whisper-diarization-server

shell:
	@echo "Opening shell in running container..."
	docker-compose exec whisper-diarization-server /bin/bash

# Health check
health:
	@echo "Checking server health..."
	curl -f http://localhost:8000/health || echo "Server is not running"

# Model management
preload-models: download-models

# Development helpers
check-deps:
	@echo "Checking dependencies..."
	python -c "import torch, whisper, pyannote.audio; print('✓ All dependencies available')"

info:
	@echo "System Information:"
	@echo "Python version: $$(python --version)"
	@echo "PyTorch version: $$(python -c 'import torch; print(torch.__version__)')"
	@echo "CUDA available: $$(python -c 'import torch; print(torch.cuda.is_available())')"
	@echo "GPU count: $$(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"