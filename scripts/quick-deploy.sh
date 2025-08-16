#!/bin/bash
# Quick deployment script with fallback options
set -e

echo "🎬 Movie Recommendation System - Quick Deploy"
echo "============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# Check prerequisites
print_info "Checking Docker..."
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Clean up any previous failed builds
print_info "Cleaning up previous builds..."
docker system prune -f --volumes || true

# Option to use existing monitoring setup
print_info "Starting with existing quick setup..."
docker-compose -f docker-compose.quick.yml down 2>/dev/null || true

# Start just monitoring services first
print_info "Starting Prometheus and Grafana..."
docker-compose -f docker-compose.quick.yml up -d

# Wait for services
print_info "Waiting for services to start..."
sleep 15

# Check if services are running
if curl -f http://localhost:9090 >/dev/null 2>&1; then
    print_status "Prometheus is running!"
else
    print_warning "Prometheus might still be starting..."
fi

if curl -f http://localhost:3000 >/dev/null 2>&1; then
    print_status "Grafana is running!"
else
    print_warning "Grafana might still be starting..."
fi

# Now try to build and run our app separately
print_info "Building Movie Recommendation App (this may take a while)..."

# Build with optimized Dockerfile
if docker build -f docker/Dockerfile.app -t movie-rec-app:latest .; then
    print_status "Docker build successful!"
    
    # Run the app
    print_info "Starting Movie Recommendation App..."
    docker run -d \
        --name movie-rec-app \
        --network docker_default \
        -p 8001:8001 \
        -v "$(pwd)/models:/app/models:ro" \
        -v "$(pwd)/data:/app/data:ro" \
        -v "$(pwd)/logs:/app/logs" \
        -e PYTHONPATH=/app \
        movie-rec-app:latest
    
    # Wait and test
    sleep 20
    if curl -f http://localhost:8001/health >/dev/null 2>&1; then
        print_status "App is healthy!"
    else
        print_warning "App might still be starting... checking logs:"
        docker logs movie-rec-app --tail 20
    fi
else
    print_error "Docker build failed. Falling back to local development mode..."
    print_info "You can still use the monitoring services and run the app locally:"
    print_info "python src/api/enhanced_fastapi_app.py"
fi

echo ""
print_status "🎉 Setup completed!"
echo ""
echo "🔗 Available Services:"
echo "   📊 Grafana Dashboard: http://localhost:3000 (admin/admin123)"
echo "   📈 Prometheus: http://localhost:9090"
if docker ps | grep -q movie-rec-app; then
    echo "   🎬 Movie Recommendation API: http://localhost:8001"
    echo "   📚 API Documentation: http://localhost:8001/docs"
fi
echo ""
echo "📋 Useful Commands:"
echo "   • View app logs: docker logs movie-rec-app"
echo "   • Stop services: docker-compose -f docker-compose.quick.yml down"
echo "   • Remove app: docker rm -f movie-rec-app"
