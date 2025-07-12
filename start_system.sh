#!/bin/bash

# Genomic Pleiotropy Cryptanalysis - System Startup Script

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

echo "=========================================="
echo "Starting Genomic Pleiotropy Analysis System"
echo "=========================================="

# Check dependencies
echo "Checking dependencies..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check Rust
if ! command -v cargo &> /dev/null; then
    echo "WARNING: Rust/Cargo not found. Rust components will run in Docker only."
else
    echo "✓ Rust/Cargo found"
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "WARNING: Python3 not found. Python components will run in Docker only."
else
    echo "✓ Python3 found"
fi

# Function to check if Redis is running
check_redis() {
    docker ps | grep -q redis || return 1
}

# Function to check if services are healthy
check_health() {
    echo "Checking service health..."
    
    # Check Redis
    if docker exec pleiotropy-redis redis-cli ping > /dev/null 2>&1; then
        echo "✓ Redis is healthy"
    else
        echo "✗ Redis is not responding"
        return 1
    fi
    
    # Check coordinator
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "✓ Coordinator is healthy"
    else
        echo "✗ Coordinator is not responding"
        return 1
    fi
    
    return 0
}

# Parse command line arguments
MODE="docker"  # Default mode
DETACHED=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            MODE="local"
            shift
            ;;
        --docker)
            MODE="docker"
            shift
            ;;
        -d|--detached)
            DETACHED="-d"
            shift
            ;;
        --stop)
            echo "Stopping all services..."
            docker-compose -f "$PROJECT_ROOT/docker-compose.yml" down
            echo "Services stopped."
            exit 0
            ;;
        --status)
            echo "Checking system status..."
            docker-compose -f "$PROJECT_ROOT/docker-compose.yml" ps
            check_health || echo "Some services are unhealthy"
            exit 0
            ;;
        --logs)
            docker-compose -f "$PROJECT_ROOT/docker-compose.yml" logs -f
            exit 0
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --local       Run services locally (requires dependencies)"
            echo "  --docker      Run services in Docker containers (default)"
            echo "  -d, --detached Run in background"
            echo "  --stop        Stop all services"
            echo "  --status      Check service status"
            echo "  --logs        Show service logs"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ "$MODE" = "docker" ]; then
    echo "Starting services in Docker mode..."
    
    # Build images if needed
    echo "Building Docker images..."
    docker-compose -f "$PROJECT_ROOT/docker-compose.yml" build
    
    # Start services
    echo "Starting services..."
    docker-compose -f "$PROJECT_ROOT/docker-compose.yml" up $DETACHED
    
    if [ -z "$DETACHED" ]; then
        # Running in foreground, exit handled by docker-compose
        exit 0
    fi
    
else
    echo "Starting services in local mode..."
    
    # Check if Redis is running locally
    if ! pgrep -x "redis-server" > /dev/null; then
        echo "Starting Redis..."
        redis-server --daemonize yes
    fi
    
    # Build Rust components
    echo "Building Rust components..."
    cd "$PROJECT_ROOT/rust_impl"
    cargo build --release
    cd "$PROJECT_ROOT"
    
    # Install Python dependencies
    echo "Installing Python dependencies..."
    cd "$PROJECT_ROOT/python_analysis"
    pip3 install -r requirements.txt
    cd "$PROJECT_ROOT/trial_database/swarm"
    pip3 install redis aioredis asyncio
    cd "$PROJECT_ROOT"
    
    # Start coordinator
    echo "Starting Swarm Coordinator..."
    python3 "$PROJECT_ROOT/trial_database/swarm/coordinator.py" &
    COORDINATOR_PID=$!
    
    # Wait for coordinator to start
    sleep 3
    
    # Start agents
    echo "Starting Rust Analyzer Agent..."
    python3 "$PROJECT_ROOT/trial_database/swarm/rust_analyzer_agent.py" &
    RUST_AGENT_PID=$!
    
    echo "Starting Python Visualizer Agent..."
    python3 "$PROJECT_ROOT/trial_database/swarm/python_visualizer_agent.py" &
    PYTHON_AGENT_PID=$!
    
    # Save PIDs
    echo $COORDINATOR_PID > "$PROJECT_ROOT/.coordinator.pid"
    echo $RUST_AGENT_PID > "$PROJECT_ROOT/.rust_agent.pid"
    echo $PYTHON_AGENT_PID > "$PROJECT_ROOT/.python_agent.pid"
    
    echo "Services started in local mode."
    echo "PIDs saved to .*.pid files"
    
    if [ -z "$DETACHED" ]; then
        # Wait for services
        echo "Press Ctrl+C to stop..."
        wait
    fi
fi

# Health check
if [ -n "$DETACHED" ]; then
    echo ""
    echo "Waiting for services to be ready..."
    sleep 5
    
    if check_health; then
        echo ""
        echo "✓ All services are running!"
        echo ""
        echo "Access points:"
        echo "  - Coordinator API: http://localhost:8080"
        echo "  - Redis: localhost:6379"
        echo ""
        echo "To view logs: $0 --logs"
        echo "To stop: $0 --stop"
    else
        echo ""
        echo "✗ Some services failed to start properly"
        echo "Check logs with: $0 --logs"
        exit 1
    fi
fi