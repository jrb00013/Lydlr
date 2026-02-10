#!/bin/bash
# Script to fix Redis port conflict by stopping existing Redis or changing port

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Redis Port Conflict Fixer${NC}"
echo ""

# Check if port 6380 (Redis host port) is in use
if lsof -i :6380 2>/dev/null || netstat -tuln 2>/dev/null | grep -q ":6380"; then
    echo -e "${YELLOW}Port 6380 (Redis host port) is in use. Checking what's using it...${NC}"
    
    # Check if it's a Docker container
    CONTAINER=$(docker ps --format "{{.Names}}" | grep -i redis)
    if [ ! -z "$CONTAINER" ]; then
        echo -e "${GREEN}Found Docker Redis container: $CONTAINER${NC}"
        echo "Stopping it..."
        docker stop "$CONTAINER" 2>/dev/null || true
        docker rm "$CONTAINER" 2>/dev/null || true
        echo -e "${GREEN}Container stopped${NC}"
    else
        # Check if it's a system Redis service
        if systemctl is-active --quiet redis 2>/dev/null || systemctl is-active --quiet redis-server 2>/dev/null; then
            echo -e "${YELLOW}Found system Redis service${NC}"
            echo "You may need to stop it manually:"
            echo "  sudo systemctl stop redis"
            echo "  OR"
            echo "  sudo systemctl stop redis-server"
        else
            echo -e "${RED}Port 6380 is in use by an unknown process${NC}"
            echo "Run this to see what's using it:"
            echo "  sudo lsof -i :6380"
            echo "  OR"
            echo "  sudo netstat -tuln | grep 6380"
        fi
    fi
else
    echo -e "${GREEN}Port 6380 is available${NC}"
fi

