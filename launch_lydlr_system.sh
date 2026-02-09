#!/bin/bash
# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# Master Launch Script for Lydlr Revolutionary Compression System
# - Sets up environment
# - Builds ROS2 package
# - Deploys models
# - Launches all nodes
# - Monitors performance

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$SCRIPT_DIR"
SRC_DIR="$WORKSPACE_DIR/src/lydlr_ai"
MODEL_DIR="$SRC_DIR/models"
NUM_NODES=${NUM_NODES:-2}
MODEL_VERSION=${MODEL_VERSION:-"vv1.0"}

# Check if we're in WSL
if grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null ; then
    IS_WSL=true
else
    IS_WSL=false
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 Lydlr Revolutionary System Launcher${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Step 1: Check ROS2
print_info "Step 1: Checking ROS2 installation..."
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    print_status "ROS2 Humble found"
else
    print_error "ROS2 Humble not found. Please install ROS2 Humble first."
    exit 1
fi

# Step 2: Setup Python environment
print_info "Step 2: Setting up Python environment..."
cd "$WORKSPACE_DIR"

if [ ! -d ".venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
print_status "Virtual environment activated"

# Step 3: Install dependencies if needed
print_info "Step 3: Checking dependencies..."
if ! python -c "import torch" 2>/dev/null; then
    print_info "Installing PyTorch and dependencies..."
    pip install torch torchvision torchaudio numpy tqdm lpips scikit-image psutil open3d --quiet
fi
print_status "Dependencies ready"

# Step 4: Setup PYTHONPATH
export PYTHONPATH="$SRC_DIR:${PYTHONPATH}"
print_status "PYTHONPATH configured"

# Step 5: Build ROS2 package
print_info "Step 4: Building ROS2 package..."
cd "$WORKSPACE_DIR"

# Check if we're in a ROS2 workspace
if [ ! -d "src" ]; then
    print_error "Not in a ROS2 workspace. Please run from workspace root."
    exit 1
fi

# Build package
if colcon build --symlink-install --packages-select lydlr_ai 2>&1 | tee /tmp/lydlr_build.log; then
    source install/setup.bash
    print_status "Package built successfully"
else
    print_error "Build failed. Check /tmp/lydlr_build.log"
    exit 1
fi

# Step 6: Check models
print_info "Step 5: Checking trained models..."
if [ ! -f "$MODEL_DIR/compressor_${MODEL_VERSION}.pth" ]; then
    print_error "Model not found: compressor_${MODEL_VERSION}.pth"
    print_info "Available models:"
    ls -lh "$MODEL_DIR"/*.pth 2>/dev/null || echo "  No models found"
    print_info "To train models, run:"
    echo "  cd $SRC_DIR && python lydlr_ai/model/train_synthetic_models.py --epochs 20 --samples 1000"
    exit 1
fi
print_status "Model found: compressor_${MODEL_VERSION}.pth"

# Step 7: Create necessary directories
print_info "Step 6: Creating directories..."
mkdir -p "$MODEL_DIR/node_0"
mkdir -p "$MODEL_DIR/node_1"
mkdir -p "$SCRIPT_DIR/scripts/node_0"
mkdir -p "$SCRIPT_DIR/scripts/node_1"
print_status "Directories created"

# Step 8: Copy models to node directories
print_info "Step 7: Deploying models to nodes..."
for i in $(seq 0 $((NUM_NODES - 1))); do
    NODE_ID="node_$i"
    NODE_MODEL_DIR="$MODEL_DIR/$NODE_ID"
    mkdir -p "$NODE_MODEL_DIR"
    
    # Copy model if not exists
    if [ ! -f "$NODE_MODEL_DIR/compressor_${MODEL_VERSION}.pth" ]; then
        cp "$MODEL_DIR/compressor_${MODEL_VERSION}.pth" "$NODE_MODEL_DIR/"
        cp "$MODEL_DIR/metadata_${MODEL_VERSION}.json" "$NODE_MODEL_DIR/" 2>/dev/null || true
        print_status "Model deployed to $NODE_ID"
    else
        print_status "Model already deployed to $NODE_ID"
    fi
done

# Step 9: Setup display (for WSL/headless)
if [ "$IS_WSL" = true ]; then
    print_info "Step 8: Setting up display for WSL..."
    export XDG_RUNTIME_DIR=/tmp/runtime-root
    mkdir -p $XDG_RUNTIME_DIR && chmod 700 $XDG_RUNTIME_DIR
    if ! pgrep -x "Xvfb" > /dev/null; then
        Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
        sleep 1
    fi
    export DISPLAY=:99
    print_status "Display configured"
fi

# Step 10: Launch nodes
print_info "Step 9: Launching Lydlr system..."
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 Starting Lydlr Nodes${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create a function to cleanup on exit
cleanup() {
    echo ""
    print_info "Shutting down Lydlr system..."
    kill $(jobs -p) 2>/dev/null || true
    wait
    print_status "All nodes stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Launch synthetic data publisher
print_info "Launching synthetic data publisher..."
ros2 run lydlr_ai synthetic_multimodal_publisher > /tmp/lydlr_synthetic.log 2>&1 &
SYNTHETIC_PID=$!
sleep 2
print_status "Synthetic publisher started (PID: $SYNTHETIC_PID)"

# Launch edge nodes
EDGE_PIDS=()
for i in $(seq 0 $((NUM_NODES - 1))); do
    NODE_ID="node_$i"
    print_info "Launching edge node: $NODE_ID..."
    NODE_ID=$NODE_ID ros2 run lydlr_ai edge_compressor_node > /tmp/lydlr_${NODE_ID}.log 2>&1 &
    EDGE_PIDS+=($!)
    sleep 1
    print_status "$NODE_ID started (PID: ${EDGE_PIDS[$i]})"
done

# Launch deployment manager
print_info "Launching model deployment manager..."
ros2 run lydlr_ai model_deployment_manager > /tmp/lydlr_deployment.log 2>&1 &
DEPLOYMENT_PID=$!
sleep 2
print_status "Deployment manager started (PID: $DEPLOYMENT_PID)"

# Launch coordinator
print_info "Launching distributed coordinator..."
ros2 run lydlr_ai distributed_coordinator > /tmp/lydlr_coordinator.log 2>&1 &
COORDINATOR_PID=$!
sleep 2
print_status "Coordinator started (PID: $COORDINATOR_PID)"

# Step 11: Deploy models
print_info "Step 10: Deploying models to nodes..."
sleep 3  # Wait for nodes to be ready

for i in $(seq 0 $((NUM_NODES - 1))); do
    NODE_ID="node_$i"
    print_info "Deploying model to $NODE_ID..."
    ros2 topic pub --once /${NODE_ID}/model/deploy std_msgs/String "data: '${MODEL_VERSION}'" > /dev/null 2>&1
    sleep 1
    print_status "Model deployed to $NODE_ID"
done

# Step 12: Monitoring
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}📊 Performance Monitoring${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
print_info "System is running! Monitoring performance..."
print_info "Press Ctrl+C to stop all nodes"
echo ""

# Monitoring loop
MONITOR_INTERVAL=5
while true; do
    clear
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}📊 Lydlr System Status${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    # Check if nodes are running
    echo -e "${YELLOW}Node Status:${NC}"
    for i in $(seq 0 $((NUM_NODES - 1))); do
        NODE_ID="node_$i"
        if kill -0 ${EDGE_PIDS[$i]} 2>/dev/null; then
            echo -e "  ${GREEN}✅ $NODE_ID${NC} (PID: ${EDGE_PIDS[$i]})"
        else
            echo -e "  ${RED}❌ $NODE_ID${NC} (stopped)"
        fi
    done
    
    echo ""
    echo -e "${YELLOW}System Components:${NC}"
    if kill -0 $SYNTHETIC_PID 2>/dev/null; then
        echo -e "  ${GREEN}✅ Synthetic Publisher${NC} (PID: $SYNTHETIC_PID)"
    else
        echo -e "  ${RED}❌ Synthetic Publisher${NC} (stopped)"
    fi
    
    if kill -0 $DEPLOYMENT_PID 2>/dev/null; then
        echo -e "  ${GREEN}✅ Deployment Manager${NC} (PID: $DEPLOYMENT_PID)"
    else
        echo -e "  ${RED}❌ Deployment Manager${NC} (stopped)"
    fi
    
    if kill -0 $COORDINATOR_PID 2>/dev/null; then
        echo -e "  ${GREEN}✅ Coordinator${NC} (PID: $COORDINATOR_PID)"
    else
        echo -e "  ${RED}❌ Coordinator${NC} (stopped)"
    fi
    
    echo ""
    echo -e "${YELLOW}Performance Metrics (last 5 seconds):${NC}"
    
    # Get metrics from nodes
    for i in $(seq 0 $((NUM_NODES - 1))); do
        NODE_ID="node_$i"
        METRICS=$(timeout 1 ros2 topic echo /${NODE_ID}/metrics --once 2>/dev/null | grep -A 5 "data:" | tail -5 | tr -d '[],' | xargs)
        
        if [ ! -z "$METRICS" ]; then
            echo -e "  ${BLUE}$NODE_ID:${NC}"
            echo "    Compression Ratio: $(echo $METRICS | awk '{print $1}')x"
            echo "    Latency: $(echo $METRICS | awk '{print $2}') ms"
            echo "    Quality: $(echo $METRICS | awk '{print $4}')"
            echo "    Bandwidth: $(echo $METRICS | awk '{print $5}')"
        else
            echo -e "  ${YELLOW}$NODE_ID:${NC} No metrics yet..."
        fi
    done
    
    echo ""
    echo -e "${YELLOW}Log Files:${NC}"
    echo "  Synthetic: /tmp/lydlr_synthetic.log"
    for i in $(seq 0 $((NUM_NODES - 1))); do
        echo "  node_$i: /tmp/lydlr_node_$i.log"
    done
    echo "  Deployment: /tmp/lydlr_deployment.log"
    echo "  Coordinator: /tmp/lydlr_coordinator.log"
    
    echo ""
    echo -e "${YELLOW}Commands:${NC}"
    echo "  View metrics: ros2 topic echo /node_0/metrics"
    echo "  View compressed: ros2 topic echo /node_0/compressed"
    echo "  List topics: ros2 topic list"
    echo "  List nodes: ros2 node list"
    
    echo ""
    echo -e "${GREEN}System running... (Refreshing every ${MONITOR_INTERVAL}s)${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    
    sleep $MONITOR_INTERVAL
done

