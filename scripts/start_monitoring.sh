#!/bin/bash
# Startup script for Movie Recommendation System Monitoring Stack

set -e

echo "🚀 Starting Movie Recommendation System Monitoring Stack"
echo "========================================================"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not available. Please install docker-compose."
    exit 1
fi

echo "✅ Docker and docker-compose are available"

# Install Python dependencies for metrics
echo "📦 Installing Python monitoring dependencies..."
pip install prometheus-client psutil 2>/dev/null || echo "⚠️  Could not install dependencies via pip"

# Create necessary directories
echo "📁 Creating monitoring directories..."
mkdir -p monitoring/prometheus/data
mkdir -p monitoring/grafana/data
mkdir -p monitoring/alertmanager/data

# Set proper permissions
echo "🔐 Setting permissions..."
sudo chown -R 472:472 monitoring/grafana/ 2>/dev/null || echo "⚠️  Could not set Grafana permissions"
sudo chown -R 65534:65534 monitoring/prometheus/ 2>/dev/null || echo "⚠️  Could not set Prometheus permissions"

# Start the monitoring stack
echo "🚀 Starting monitoring services..."
cd docker
docker-compose -f docker-compose.monitoring.yml up -d

echo ""
echo "✅ Monitoring stack is starting up!"
echo ""
echo "🔗 Service URLs:"
echo "   📊 Grafana Dashboard: http://localhost:3000 (admin/admin123)"
echo "   📈 Prometheus: http://localhost:9090"
echo "   🚨 AlertManager: http://localhost:9093"
echo "   🎬 Movie API: http://localhost:8001"
echo "   📊 API Metrics: http://localhost:8002/metrics"
echo ""
echo "📋 Getting service status..."
sleep 5
docker-compose -f docker-compose.monitoring.yml ps

echo ""
echo "🎉 Setup complete! Wait a few minutes for all services to fully start."
echo ""
echo "📚 Quick Start Guide:"
echo "   1. Open Grafana at http://localhost:3000"
echo "   2. Login with admin/admin123"
echo "   3. Navigate to the 'Movie Recommendation System' dashboard"
echo "   4. Generate some API traffic to see metrics"
echo ""
echo "🧪 Test the API:"
echo "   curl -X POST http://localhost:8001/recommend \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"movies\": [1, 2, 3], \"model\": \"sae\", \"num_recommendations\": 5}'"
