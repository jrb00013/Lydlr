#!/bin/bash
# Helper script to check for port conflicts

echo "Checking for port conflicts..."
echo ""

# Check Redis port 6380 (host port)
if lsof -i :6380 2>/dev/null || netstat -tuln 2>/dev/null | grep -q ":6380"; then
    echo "⚠️  Port 6380 (Redis host port) is in use:"
    lsof -i :6380 2>/dev/null || netstat -tuln 2>/dev/null | grep ":6380"
    echo ""
    echo "Options:"
    echo "1. Stop the existing Redis instance"
    echo "2. Change Redis port in docker-compose.yml"
    echo ""
fi

# Check MongoDB port 27017
if lsof -i :27017 2>/dev/null || netstat -tuln 2>/dev/null | grep -q ":27017"; then
    echo "⚠️  Port 27017 (MongoDB) is in use:"
    lsof -i :27017 2>/dev/null || netstat -tuln 2>/dev/null | grep ":27017"
    echo ""
fi

# Check if there are other Docker containers using these ports
echo "Checking Docker containers..."
docker ps --format "table {{.Names}}\t{{.Ports}}" | grep -E "6380|6379|27017" || echo "No Docker containers found using these ports"

