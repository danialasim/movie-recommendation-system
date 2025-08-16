#!/bin/bash
# Movie Recommendation System - Docker Management Script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

show_usage() {
    echo "🎬 Movie Recommendation System - Docker Manager"
    echo "=============================================="
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     - Start monitoring services (Prometheus + Grafana)"
    echo "  stop      - Stop all services"
    echo "  restart   - Restart all services"
    echo "  status    - Check status of all services"
    echo "  logs      - Show logs from services"
    echo "  cleanup   - Stop and remove all containers/volumes"
    echo "  full      - Start full Docker stack (including API)"
    echo ""
    echo "Examples:"
    echo "  $0 start    # Start monitoring"
    echo "  $0 stop     # Stop everything"
    echo "  $0 status   # Check what's running"
}

check_status() {
    print_info "Checking service status..."
    echo ""
    
    # Check Docker containers
    if docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(prometheus|grafana|movie-rec)" > /dev/null 2>&1; then
        print_status "Running containers:"
        docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(prometheus|grafana|movie-rec|redis)"
    else
        print_warning "No Movie Recommendation containers running"
    fi
    
    echo ""
    
    # Check service endpoints
    if curl -s http://localhost:9090 > /dev/null 2>&1; then
        print_status "Prometheus: http://localhost:9090"
    else
        print_error "Prometheus: Not running"
    fi
    
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        print_status "Grafana: http://localhost:3000 (admin/admin123)"
    else
        print_error "Grafana: Not running"
    fi
    
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        print_status "Movie API: http://localhost:8001"
    else
        print_warning "Movie API: Not running (may be local process)"
    fi
}

start_services() {
    print_info "Starting Movie Recommendation monitoring services..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    # Start monitoring services
    docker-compose -f docker-compose.quick.yml up -d
    
    print_info "Waiting for services to start..."
    sleep 10
    
    check_status
    
    echo ""
    print_status "🎉 Services started!"
    echo ""
    print_info "Next steps:"
    echo "  1. Open Grafana: http://localhost:3000 (admin/admin123)"
    echo "  2. Start API locally: conda activate ./movies-env && python src/api/enhanced_fastapi_app.py"
    echo "  3. Or use full Docker: $0 full"
}

start_full() {
    print_info "Starting full Docker stack..."
    
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    docker-compose -f docker/docker-compose.full.yml up -d
    
    print_info "Waiting for services to start..."
    sleep 15
    
    check_status
}

stop_services() {
    print_info "Stopping Movie Recommendation services..."
    
    # Stop docker-compose services
    docker-compose -f docker-compose.quick.yml down 2>/dev/null || true
    docker-compose -f docker/docker-compose.full.yml down 2>/dev/null || true
    
    # Stop any individual containers
    docker stop prometheus-standalone grafana-standalone movie-rec-app redis 2>/dev/null || true
    
    print_status "All services stopped"
}

restart_services() {
    print_info "Restarting services..."
    stop_services
    sleep 3
    start_services
}

show_logs() {
    print_info "Recent logs from services:"
    echo ""
    
    if docker ps --format "{{.Names}}" | grep prometheus-standalone > /dev/null; then
        echo "📈 Prometheus logs:"
        docker logs prometheus-standalone --tail 5
        echo ""
    fi
    
    if docker ps --format "{{.Names}}" | grep grafana-standalone > /dev/null; then
        echo "📊 Grafana logs:"
        docker logs grafana-standalone --tail 5
        echo ""
    fi
    
    if docker ps --format "{{.Names}}" | grep movie-rec-app > /dev/null; then
        echo "🎬 Movie API logs:"
        docker logs movie-rec-app --tail 10
        echo ""
    fi
}

cleanup() {
    print_warning "This will stop and remove all containers, networks, and volumes!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cleaning up..."
        
        # Stop and remove everything
        docker-compose -f docker-compose.quick.yml down --volumes --remove-orphans 2>/dev/null || true
        docker-compose -f docker/docker-compose.full.yml down --volumes --remove-orphans 2>/dev/null || true
        
        # Remove individual containers
        docker rm -f prometheus-standalone grafana-standalone movie-rec-app redis 2>/dev/null || true
        
        # Remove custom images (optional)
        docker rmi movie-rec-app:latest 2>/dev/null || true
        
        print_status "Cleanup completed"
    else
        print_info "Cleanup cancelled"
    fi
}

# Main script logic
case "${1:-}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        check_status
        ;;
    logs)
        show_logs
        ;;
    cleanup)
        cleanup
        ;;
    full)
        start_full
        ;;
    *)
        show_usage
        ;;
esac
