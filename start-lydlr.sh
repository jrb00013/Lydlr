#!/bin/bash

# Lydlr Full-Stack Startup Script
# This script starts the complete Lydlr system with Docker

set -e

echo "=========================================="
echo "Lydlr Full-Stack System Launcher"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[ERROR] Docker is not installed${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed (try both new and old syntax)
if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}[ERROR] Docker Compose is not installed${NC}"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Use docker-compose if available, otherwise docker compose
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
else
    DOCKER_COMPOSE_CMD="docker compose"
fi

echo -e "${GREEN}[OK] Docker and Docker Compose found${NC}"
echo ""

# Check for port conflicts
echo -e "${YELLOW}[INFO] Checking for port conflicts...${NC}"
if command -v lsof &> /dev/null; then
    if lsof -i :6380 &> /dev/null; then
        echo -e "${YELLOW}[WARNING] Port 6380 (Redis) is already in use${NC}"
        echo "  Run: docker ps | grep redis"
        echo "  Or: sudo lsof -i :6380"
        echo "  To stop: docker stop \$(docker ps -q --filter 'publish=6380')"
    fi
    if lsof -i :27017 &> /dev/null; then
        echo -e "${YELLOW}[WARNING] Port 27017 (MongoDB) is already in use${NC}"
    fi
fi
echo ""

# Check if .env exists, if not copy from example
if [ ! -f .env ]; then
    echo -e "${YELLOW}[WARNING] .env file not found, creating from .env.example${NC}"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}[OK] .env file created${NC}"
    else
        echo -e "${RED}[ERROR] .env.example not found${NC}"
        echo "Creating a basic .env file with defaults..."
        cat > .env << 'EOF'
# MongoDB Configuration
MONGODB_URL=mongodb://lydlr:lydlr_password@mongodb:27017/lydlr_db?authSource=admin
MONGO_INITDB_ROOT_USERNAME=lydlr
MONGO_INITDB_ROOT_PASSWORD=lydlr_password
MONGO_INITDB_DATABASE=lydlr_db

# Redis Configuration
REDIS_URL=redis://redis:6379

# API Configuration
API_URL=http://localhost:8000
WS_URL=ws://localhost:8000

# Model Service
MODEL_SERVICE_URL=http://localhost:8001
MODEL_DIR=/app/models

# ROS2 Configuration
ROS_DOMAIN_ID=0

# Frontend Configuration
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
EOF
        echo -e "${GREEN}[OK] Basic .env file created${NC}"
    fi
fi

# Parse command line arguments
BUILD_FLAG=""
DETACHED_FLAG=""
SHOW_LOGS=false
LAUNCH_ROS2=false  # ROS2 is handled by Docker service (ros2-runtime)
STOP_ROS2=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD_FLAG="--build"
            shift
            ;;
        -d|--detached)
            DETACHED_FLAG="-d"
            shift
            ;;
        --logs)
            SHOW_LOGS=true
            shift
            ;;
        --ros2|--launch-ros2)
            LAUNCH_ROS2=true
            shift
            ;;
        --no-ros2|--skip-ros2)
            LAUNCH_ROS2=false
            shift
            ;;
        --stop-ros2)
            STOP_ROS2=true
            shift
            ;;
        --stop)
            echo -e "${YELLOW}[INFO] Stopping Lydlr system...${NC}"
            $DOCKER_COMPOSE_CMD down
            
            # Also stop ROS2 if running
            if pgrep -f "launch_lydlr_system.sh" > /dev/null; then
                echo -e "${YELLOW}Stopping ROS2 runtime...${NC}"
                pkill -f "launch_lydlr_system.sh" || true
                pkill -f "ros2 run lydlr_ai" || true
            fi
            
            echo -e "${GREEN}[OK] Lydlr system stopped${NC}"
            exit 0
            ;;
        --restart)
            echo -e "${YELLOW}[INFO] Restarting Lydlr system...${NC}"
            $DOCKER_COMPOSE_CMD restart
            echo -e "${GREEN}[OK] Lydlr system restarted${NC}"
            exit 0
            ;;
        --status)
            echo -e "${BLUE}[STATUS] Lydlr system status:${NC}"
            $DOCKER_COMPOSE_CMD ps
            echo ""
            echo -e "${BLUE}ROS2 Runtime Status:${NC}"
            if pgrep -f "launch_lydlr_system.sh" > /dev/null; then
                echo -e "${GREEN}[OK] ROS2 runtime is running${NC}"
                ps aux | grep -E "(launch_lydlr_system|ros2 run lydlr_ai)" | grep -v grep | head -5
            else
                echo -e "${YELLOW}[WARNING] ROS2 runtime is not running${NC}"
            fi
            exit 0
            ;;
        --logs-all)
            $DOCKER_COMPOSE_CMD logs -f
            exit 0
            ;;
        --clean)
            echo -e "${YELLOW}[INFO] Cleaning up Lydlr system...${NC}"
            $DOCKER_COMPOSE_CMD down -v
            
            # Also stop ROS2 if running
            if pgrep -f "launch_lydlr_system.sh" > /dev/null; then
                echo -e "${YELLOW}Stopping ROS2 runtime...${NC}"
                pkill -f "launch_lydlr_system.sh" || true
                pkill -f "ros2 run lydlr_ai" || true
            fi
            
            echo -e "${GREEN}[OK] Lydlr system cleaned (volumes removed)${NC}"
            exit 0
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --build         Build images before starting"
            echo "  -d, --detached  Run in detached mode"
            echo "  --logs          Show logs after starting"
            echo "  --ros2          Launch ROS2 runtime natively (outside Docker) - Docker service handles ROS2 by default"
            echo "  --stop-ros2     Stop ROS2 runtime only"
            echo "  --stop          Stop the system (Docker + ROS2)"
            echo "  --restart       Restart all services"
            echo "  --status        Show status of all services"
            echo "  --logs-all      Show logs for all services"
            echo "  --clean         Stop and remove all containers and volumes"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Handle stop ROS2 only
if [ "$STOP_ROS2" = true ]; then
    echo -e "${YELLOW}[INFO] Stopping ROS2 runtime...${NC}"
    if pgrep -f "launch_lydlr_system.sh" > /dev/null; then
        pkill -f "launch_lydlr_system.sh" || true
        pkill -f "ros2 run lydlr_ai" || true
        echo -e "${GREEN}[OK] ROS2 runtime stopped${NC}"
    else
        echo -e "${YELLOW}[WARNING] ROS2 runtime is not running${NC}"
    fi
    exit 0
fi

echo -e "${BLUE}[INFO] Starting Lydlr Docker services...${NC}"
echo ""

# Start Docker services
if [ -z "$DETACHED_FLAG" ]; then
    # Attached mode - start in background but we'll wait
    $DOCKER_COMPOSE_CMD up $BUILD_FLAG -d
    sleep 5
else
    # Detached mode
    $DOCKER_COMPOSE_CMD up $BUILD_FLAG -d
    echo -e "${YELLOW}[INFO] Waiting for services to start...${NC}"
    sleep 10
fi

# Wait for services to be healthy
echo -e "${YELLOW}[INFO] Waiting for services to be ready...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}[OK] Backend is ready${NC}"
        break
    fi
    sleep 1
done

echo ""
echo -e "${GREEN}[OK] Docker services started successfully!${NC}"
echo ""
echo "=========================================="
echo -e "${BLUE}Access Points:${NC}"
echo "=========================================="
echo ""
echo -e "  Frontend:        ${GREEN}http://localhost${NC}"
echo -e "  Backend API:      ${GREEN}http://localhost:8000/docs${NC}"
echo -e "  Model Service:    ${GREEN}http://localhost:8001${NC}"
echo -e "  MongoDB:         ${GREEN}mongodb://localhost:27017${NC}"
echo -e "  Redis:           ${GREEN}localhost:6380${NC}"
echo ""
echo "=========================================="
echo -e "${BLUE}Docker Service Status:${NC}"
echo "=========================================="
echo ""
$DOCKER_COMPOSE_CMD ps
echo ""

# Launch ROS2 runtime if requested
if [ "$LAUNCH_ROS2" = true ]; then
    echo ""
    echo "=========================================="
    echo -e "${BLUE}Launching ROS2 Runtime...${NC}"
    echo "=========================================="
    echo ""
    
    # Check if launch_lydlr_system.sh exists
    if [ ! -f "launch_lydlr_system.sh" ]; then
        echo -e "${RED}[ERROR] launch_lydlr_system.sh not found${NC}"
        echo "ROS2 runtime will not be started"
    else
        # Check if ROS2 is already running
        if pgrep -f "launch_lydlr_system.sh" > /dev/null; then
            echo -e "${YELLOW}[WARNING] ROS2 runtime is already running${NC}"
            echo "Use --stop-ros2 to stop it first"
        else
            # Make sure script is executable
            chmod +x launch_lydlr_system.sh
            
            # Launch ROS2 system in background
            echo -e "${YELLOW}[INFO] Starting ROS2 nodes...${NC}"
            ./launch_lydlr_system.sh > /tmp/lydlr_ros2_launch.log 2>&1 &
            ROS2_PID=$!
            
            sleep 3
            
            if kill -0 $ROS2_PID 2>/dev/null; then
                echo -e "${GREEN}[OK] ROS2 runtime started (PID: $ROS2_PID)${NC}"
                echo -e "${YELLOW}[INFO] ROS2 logs: /tmp/lydlr_ros2_launch.log${NC}"
            else
                echo -e "${RED}[ERROR] Failed to start ROS2 runtime${NC}"
                echo "Check /tmp/lydlr_ros2_launch.log for details"
            fi
        fi
    fi
    echo ""
fi

# Show final status
if [ "$SHOW_LOGS" = true ]; then
    echo -e "${BLUE}[INFO] Showing Docker logs (Ctrl+C to stop):${NC}"
    $DOCKER_COMPOSE_CMD logs -f
else
    echo -e "${YELLOW}Tips:${NC}"
    echo "  - View Docker logs: $DOCKER_COMPOSE_CMD logs -f"
    echo "  - View ROS2 logs: tail -f /tmp/lydlr_ros2_launch.log"
    echo "  - Stop Docker: $DOCKER_COMPOSE_CMD down"
    echo "  - Stop ROS2: ./start-lydlr.sh --stop-ros2"
    echo "  - Stop all: ./start-lydlr.sh --stop"
    echo "  - Check status: ./start-lydlr.sh --status"
    echo ""
    
    echo -e "${GREEN}[OK] Lydlr is ready!${NC}"
    echo "  - Web Interface: http://localhost"
    echo "  - ROS2 Runtime: Running in Docker (ros2-runtime service)"
    if [ "$LAUNCH_ROS2" = true ]; then
        echo "  - Native ROS2: Also running natively"
    fi
fi

