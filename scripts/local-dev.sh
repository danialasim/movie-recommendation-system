#!/bin/bash
# Local development with monitoring
set -e

echo "🎬 Movie Recommendation System - Local Development Mode"
echo "======================================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# Start monitoring services only
print_info "Starting monitoring services (Prometheus + Grafana)..."
docker-compose -f docker-compose.quick.yml up -d

print_info "Waiting for services to start..."
sleep 10

# Check services
if curl -f http://localhost:9090 >/dev/null 2>&1; then
    print_status "Prometheus is running at http://localhost:9090"
else
    print_warning "Prometheus is still starting..."
fi

if curl -f http://localhost:3000 >/dev/null 2>&1; then
    print_status "Grafana is running at http://localhost:3000 (admin/admin123)"
else
    print_warning "Grafana is still starting..."
fi

print_info "Setting up Python environment..."

# Check if virtual environment exists
if [ ! -d "movies-env" ]; then
    print_info "Creating Python virtual environment..."
    python -m venv movies-env
fi

# Activate virtual environment
source movies-env/bin/activate

# Install requirements
print_info "Installing Python requirements..."
pip install -r requirements.txt

# Set environment variables
export PYTHONPATH=$(pwd):$(pwd)/src
export LOG_LEVEL=INFO

print_status "🚀 Ready to start!"
echo ""
echo "To start the Movie Recommendation API locally:"
echo "1. Activate environment: source movies-env/bin/activate"
echo "2. Set path: export PYTHONPATH=\$(pwd):\$(pwd)/src"
echo "3. Start API: python src/api/enhanced_fastapi_app.py"
echo ""
echo "Or run this command:"
echo "source movies-env/bin/activate && export PYTHONPATH=\$(pwd):\$(pwd)/src && python src/api/enhanced_fastapi_app.py"
echo ""
echo "🔗 Services:"
echo "   📊 Grafana: http://localhost:3000 (admin/admin123)"
echo "   📈 Prometheus: http://localhost:9090"
echo "   🎬 API (when started): http://localhost:8001"
