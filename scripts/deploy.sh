#!/bin/bash
# Movie Recommendation System - Complete Deployment Script
set -e

echo "🎬 Movie Recommendation System - Docker Deployment"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check prerequisites
echo ""
print_info "Checking prerequisites..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi
print_status "Docker is running"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose is not available. Please install docker-compose."
    exit 1
fi
print_status "docker-compose is available"

# Check if models exist
if [ ! -d "models" ] || [ -z "$(ls -A models)" ]; then
    print_warning "No models found in models/ directory. Training models first..."
    if [ -f "start_training.py" ]; then
        print_info "Running model training..."
        python start_training.py
    else
        print_error "No training script found. Please train models first."
        exit 1
    fi
else
    print_status "Trained models found"
fi

# Create necessary directories
print_info "Creating necessary directories..."
mkdir -p monitoring/prometheus/data
mkdir -p monitoring/grafana/data
mkdir -p logs/{api,training,evaluation}
mkdir -p data/cache

# Set proper permissions for monitoring
print_info "Setting permissions for monitoring..."
# For Grafana (user ID 472)
sudo chown -R 472:472 monitoring/grafana/ 2>/dev/null || print_warning "Could not set Grafana permissions"
# For Prometheus (user ID 65534)
sudo chown -R 65534:65534 monitoring/prometheus/ 2>/dev/null || print_warning "Could not set Prometheus permissions"

# Make entrypoint scripts executable
chmod +x docker/entrypoint-*.sh 2>/dev/null || print_warning "Could not make entrypoint scripts executable"

# Environment setup
print_info "Setting up environment..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
# Movie Recommendation System Environment Variables
MLFLOW_TRACKING_URI=http://localhost:5000
REDIS_URL=redis://redis:6379
LOG_LEVEL=INFO
METRICS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=admin123
PROMETHEUS_RETENTION_TIME=200h
EOF
    print_status "Created .env file"
else
    print_status ".env file already exists"
fi

# Choose deployment mode
echo ""
echo "Choose deployment mode:"
echo "1) Full stack (App + Monitoring + Nginx)"
echo "2) Application only"
echo "3) Monitoring stack only"
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        COMPOSE_FILE="docker/docker-compose.full.yml"
        print_info "Deploying full stack..."
        ;;
    2)
        COMPOSE_FILE="docker/docker-compose.yml"
        print_info "Deploying application only..."
        ;;
    3)
        COMPOSE_FILE="docker/docker-compose.monitoring.yml"
        print_info "Deploying monitoring stack only..."
        ;;
    *)
        print_error "Invalid choice. Defaulting to full stack."
        COMPOSE_FILE="docker/docker-compose.full.yml"
        ;;
esac

# Deploy the services
echo ""
print_info "Starting services with $COMPOSE_FILE..."
docker-compose -f $COMPOSE_FILE down 2>/dev/null || true
docker-compose -f $COMPOSE_FILE up -d

# Wait for services to start
print_info "Waiting for services to start..."
sleep 30

# Check service status
echo ""
print_info "Checking service status..."
docker-compose -f $COMPOSE_FILE ps

# Test API health
echo ""
print_info "Testing API health..."
max_attempts=30
attempt=1
while [ $attempt -le $max_attempts ]; do
    if curl -f http://localhost:8001/health >/dev/null 2>&1; then
        print_status "API is healthy!"
        break
    else
        print_warning "Attempt $attempt/$max_attempts - API not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    print_error "API failed to start properly"
    echo ""
    print_info "Checking logs..."
    docker-compose -f $COMPOSE_FILE logs movie-rec-app
    exit 1
fi

# Display service URLs
echo ""
print_status "🎉 Deployment completed successfully!"
echo ""
echo "🔗 Service URLs:"
echo "   🎬 Movie Recommendation API: http://localhost:8001"
echo "   📊 API Documentation (Swagger): http://localhost:8001/docs"
echo "   🌐 Web Interface (Nginx): http://localhost:80"
echo "   📈 Prometheus: http://localhost:9090"
echo "   📊 Grafana Dashboard: http://localhost:3000 (admin/admin123)"
echo "   🔧 Node Exporter: http://localhost:9100"
echo "   💾 Redis: localhost:6379"
echo ""
echo "📚 Quick Start Guide:"
echo "   1. Open Grafana at http://localhost:3000"
echo "   2. Login with admin/admin123"
echo "   3. Import pre-configured dashboards"
echo "   4. Test the API at http://localhost:8001/docs"
echo ""
echo "📋 Useful Commands:"
echo "   • View logs: docker-compose -f $COMPOSE_FILE logs [service_name]"
echo "   • Stop services: docker-compose -f $COMPOSE_FILE down"
echo "   • Restart services: docker-compose -f $COMPOSE_FILE restart"
echo "   • Scale app: docker-compose -f $COMPOSE_FILE up -d --scale movie-rec-app=3"
echo ""
echo "🔍 Monitoring:"
echo "   • Check metrics: curl http://localhost:8001/metrics"
echo "   • View API health: curl http://localhost:8001/health"
echo "   • System metrics: http://localhost:9100/metrics"
echo ""
print_status "Setup complete! Your Movie Recommendation System is running."
