#!/bin/bash

# Rebuild script for whisper-diarization-server with RTX 5060 Ti support
# Now featuring robust hardware detection and modular architecture!

echo "🔧 Rebuilding whisper-diarization-server v2.0 with RTX 5060 Ti support..."
echo "✨ New features: Hardware detection, modular architecture, better error handling"
echo ""

# Stop and remove existing container
echo "🛑 Stopping existing container..."
docker-compose down

# Remove the old image to force rebuild
echo "🗑️ Removing old image..."
docker rmi whisper-diarization-server-stt-server 2>/dev/null || true

# Clear Docker build cache for this project
echo "🧹 Clearing build cache..."
docker builder prune -f --filter label=project=whisper-diarization-server 2>/dev/null || true

# Rebuild with no cache
echo "🔨 Building new image with CUDA 12.4 and PyTorch 2.5.1..."
echo "📋 This may take a while on first build..."
docker-compose build --no-cache

echo ""
echo "🚀 Starting the updated container..."
docker-compose up -d

echo ""
echo "📊 Checking container status..."
docker-compose ps

echo ""
echo "🔍 Testing hardware detection..."
sleep 3
echo "Health check:"
curl -s http://localhost:8000/health | python3 -m json.tool

echo ""
echo "📋 Following logs (press Ctrl+C to exit)..."
echo "💡 New endpoints available:"
echo "   - GET /health (basic health check)"
echo "   - GET /health/detailed (detailed diagnostics)"
echo "   - POST /v1/audio/transcriptions (transcription)"
echo "   - POST /v1/audio/transcriptions/stream (streaming)"
echo ""
docker-compose logs -f stt-server
