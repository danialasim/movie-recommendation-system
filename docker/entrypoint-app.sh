#!/bin/bash
set -e

echo "🚀 Starting Movie Recommendation API..."

# Wait for Redis to be ready (optional)
echo "⏳ Waiting for Redis to be ready..."
timeout=30
counter=0
while ! nc -z redis 6379 2>/dev/null; do
  if [ $counter -ge $timeout ]; then
    echo "❌ Redis not ready after ${timeout}s, starting anyway..."
    break
  fi
  sleep 1
  counter=$((counter + 1))
done
echo "✅ Redis is ready!"

# Create necessary directories
mkdir -p /app/logs

# Check if models exist
if [ ! -d "/app/models" ] || [ -z "$(ls -A /app/models)" ]; then
    echo "⚠️  Warning: No models found in /app/models directory"
else
    echo "✅ Models directory found"
fi

echo "🎬 Starting FastAPI application..."

# Change to the application directory
cd /app

# Try different ways to start the application
if command -v uvicorn >/dev/null 2>&1; then
    echo "Using uvicorn..."
    exec uvicorn src.api.enhanced_fastapi_app:app --host 0.0.0.0 --port 8001
elif python -m uvicorn --help >/dev/null 2>&1; then
    echo "Using python -m uvicorn..."
    exec python -m uvicorn src.api.enhanced_fastapi_app:app --host 0.0.0.0 --port 8001
else
    echo "Fallback to direct Python execution..."
    exec python src/api/enhanced_fastapi_app.py
fi
