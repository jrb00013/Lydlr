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
# Support both old and new structure for compatibility
if [ -d "$WORKSPACE_DIR/ros2/src/lydlr_ai" ]; then
    SRC_DIR="$WORKSPACE_DIR/ros2/src/lydlr_ai"
elif [ -d "$WORKSPACE_DIR/src/lydlr_ai" ]; then
    SRC_DIR="$WORKSPACE_DIR/src/lydlr_ai"
else
    # In container, use /app/src/lydlr_ai
    SRC_DIR="/app/src/lydlr_ai"
fi
MODEL_DIR="$SRC_DIR/models"
MODEL_VERSION=${MODEL_VERSION:-"vv1.0"}

# Node configuration - will be loaded from MongoDB
NODE_IDS=()  # Array of node IDs to start
TARGET_NODE_COUNT=0  # Target number of nodes from config

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
# Try multiple possible ROS2 locations
ROS2_SETUP=""
if [ -f "/opt/ros/humble/setup.bash" ]; then
    ROS2_SETUP="/opt/ros/humble/setup.bash"
elif [ -f "/opt/ros/humble/setup.sh" ]; then
    ROS2_SETUP="/opt/ros/humble/setup.sh"
elif [ -d "/opt/ros/humble" ]; then
    # Try to find setup file
    ROS2_SETUP=$(find /opt/ros/humble -name "setup.bash" -o -name "setup.sh" | head -1)
fi

if [ -n "$ROS2_SETUP" ] && [ -f "$ROS2_SETUP" ]; then
    source "$ROS2_SETUP"
    print_status "ROS2 Humble found at $ROS2_SETUP"
    # Verify ROS2 is working
    if command -v ros2 &> /dev/null; then
        print_status "ROS2 command available: $(ros2 --version 2>/dev/null || echo 'version check failed')"
    else
        print_error "ROS2 setup file found but ros2 command not available"
        exit 1
    fi
else
    print_error "ROS2 Humble not found. Please ensure ROS2 is installed."
    print_info "Checked locations:"
    print_info "  /opt/ros/humble/setup.bash"
    print_info "  /opt/ros/humble/setup.sh"
    print_info "  /opt/ros/humble/"
    exit 1
fi

# Step 2: Setup Python environment
print_info "Step 2: Setting up Python environment..."
cd "$WORKSPACE_DIR"

# Check if venv exists and is valid
if [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
    print_info "Using existing virtual environment..."
    source .venv/bin/activate
    if [ $? -eq 0 ]; then
        print_status "Virtual environment activated"
    else
        print_error "Failed to activate existing virtual environment"
        print_info "Recreating virtual environment..."
        rm -rf .venv
        python3 -m venv .venv
        if [ $? -ne 0 ] || [ ! -f ".venv/bin/activate" ]; then
            print_error "Failed to create virtual environment"
            exit 1
        fi
        source .venv/bin/activate
    fi
else
    print_info "Creating virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ] || [ ! -f ".venv/bin/activate" ]; then
        print_error "Failed to create virtual environment"
        exit 1
    fi
    source .venv/bin/activate
    print_status "Virtual environment created and activated"
fi

# Verify activation worked
if ! command -v python &> /dev/null || [ "$(which python)" != "$WORKSPACE_DIR/.venv/bin/python" ]; then
    print_error "Virtual environment activation failed - python not in venv"
    exit 1
fi

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

# Check if we're in a ROS2 workspace (support both /app and workspace root)
if [ ! -d "src" ] && [ -d "/app/src" ]; then
    WORKSPACE_DIR="/app"
    cd "$WORKSPACE_DIR"
    print_info "Using container workspace at /app"
fi

if [ ! -d "src" ]; then
    print_error "Not in a ROS2 workspace. Please run from workspace root."
    exit 1
fi

# Ensure install directory exists
mkdir -p install log

# Build package with better error handling
print_info "Building lydlr_ai package..."
# Build without symlink-install to avoid --editable issues
if colcon build --packages-select lydlr_ai 2>&1 | tee /tmp/lydlr_build.log; then
    if [ -f install/setup.bash ]; then
        source install/setup.bash
        print_status "Package built successfully"
    else
        print_error "Build completed but setup.bash not found"
        exit 1
    fi
else
    print_error "Build failed. Check /tmp/lydlr_build.log"
    print_info "Attempting to continue with existing build..."
    if [ -f install/setup.bash ]; then
        source install/setup.bash
        print_status "Using existing build"
    else
        print_error "No existing build found. Cannot continue."
        exit 1
    fi
fi

# Step 6: Check models
print_info "Step 5: Checking trained models..."
# Try new naming convention first, then fall back to old
MODEL_FILE=""
if [ -f "$MODEL_DIR/lydlr_compressor_${MODEL_VERSION}.pth" ]; then
    MODEL_FILE="lydlr_compressor_${MODEL_VERSION}.pth"
elif [ -f "$MODEL_DIR/compressor_${MODEL_VERSION}.pth" ]; then
    MODEL_FILE="compressor_${MODEL_VERSION}.pth"
else
    print_error "Model not found: lydlr_compressor_${MODEL_VERSION}.pth or compressor_${MODEL_VERSION}.pth"
    print_info "Available models:"
    ls -lh "$MODEL_DIR"/*.pth 2>/dev/null || echo "  No models found"
    print_info "To train models, run:"
    echo "  cd $SRC_DIR && python lydlr_ai/model/train_synthetic_models.py --epochs 20 --samples 1000"
    exit 1
fi
print_status "Model found: $MODEL_FILE"

# Step 7: Get node configuration from MongoDB
print_info "Step 6: Loading node configuration from database..."
if command -v python3 &> /dev/null && [ -f "$SCRIPT_DIR/scripts/get_nodes_from_db.py" ]; then
    # Try to get nodes from MongoDB
    NODE_CONFIG=$(python3 "$SCRIPT_DIR/scripts/get_nodes_from_db.py" 2>/dev/null)
    if [ $? -eq 0 ] && [ ! -z "$NODE_CONFIG" ]; then
        # Parse JSON response
        NODE_IDS_JSON=$(echo "$NODE_CONFIG" | python3 -c "import sys, json; data=json.load(sys.stdin); print(' '.join(data.get('nodes', [])))" 2>/dev/null)
        if [ ! -z "$NODE_IDS_JSON" ]; then
            NODE_IDS=($NODE_IDS_JSON)
            TARGET_COUNT=$(echo "$NODE_CONFIG" | python3 -c "import sys, json; data=json.load(sys.stdin); config=data.get('config', {}); print(config.get('target_node_count', 0))" 2>/dev/null)
            TARGET_NODE_COUNT=${TARGET_COUNT:-0}
            print_status "Found ${#NODE_IDS[@]} nodes in database: ${NODE_IDS[*]}"
        else
            print_info "No nodes found in database, will start with default configuration"
            # Default: start with 2 nodes if nothing configured
            NODE_IDS=("node_0" "node_1")
            TARGET_NODE_COUNT=2
        fi
    else
        print_info "Could not connect to database, using default nodes"
        NODE_IDS=("node_0" "node_1")
        TARGET_NODE_COUNT=2
    fi
else
    print_info "Python helper not available, using default nodes"
    NODE_IDS=("node_0" "node_1")
    TARGET_NODE_COUNT=2
fi

# Step 8: Create necessary directories
print_info "Step 7: Creating directories for nodes..."
for NODE_ID in "${NODE_IDS[@]}"; do
    mkdir -p "$MODEL_DIR/$NODE_ID"
    # Try to create scripts directory, but don't fail if it's read-only
    if mkdir -p "$SCRIPT_DIR/scripts/$NODE_ID" 2>/dev/null; then
        print_status "Created scripts directory for $NODE_ID"
    else
        # Use a writable location instead (like /app/log or /tmp)
        mkdir -p "/app/log/scripts/$NODE_ID"
        print_info "Using /app/log/scripts/$NODE_ID (scripts directory is read-only)"
    fi
done
print_status "Directories created for ${#NODE_IDS[@]} nodes"

# Step 9: Copy models to node directories
print_info "Step 8: Deploying models to nodes..."
for NODE_ID in "${NODE_IDS[@]}"; do
    NODE_MODEL_DIR="$MODEL_DIR/$NODE_ID"
    mkdir -p "$NODE_MODEL_DIR"
    
    # Copy model if not exists
    # Copy model file (support both naming conventions)
    if [ -f "$MODEL_DIR/lydlr_compressor_${MODEL_VERSION}.pth" ]; then
        if [ ! -f "$NODE_MODEL_DIR/lydlr_compressor_${MODEL_VERSION}.pth" ]; then
            cp "$MODEL_DIR/lydlr_compressor_${MODEL_VERSION}.pth" "$NODE_MODEL_DIR/"
        fi
    elif [ -f "$MODEL_DIR/compressor_${MODEL_VERSION}.pth" ]; then
        # Copy model if not exists (support both naming conventions)
        if [ -f "$MODEL_DIR/lydlr_compressor_${MODEL_VERSION}.pth" ]; then
            if [ ! -f "$NODE_MODEL_DIR/lydlr_compressor_${MODEL_VERSION}.pth" ]; then
                cp "$MODEL_DIR/lydlr_compressor_${MODEL_VERSION}.pth" "$NODE_MODEL_DIR/"
            fi
            cp "$MODEL_DIR/metadata_lydlr_compressor_${MODEL_VERSION}.json" "$NODE_MODEL_DIR/" 2>/dev/null || true
        elif [ -f "$MODEL_DIR/compressor_${MODEL_VERSION}.pth" ]; then
            if [ ! -f "$NODE_MODEL_DIR/compressor_${MODEL_VERSION}.pth" ]; then
                cp "$MODEL_DIR/compressor_${MODEL_VERSION}.pth" "$NODE_MODEL_DIR/"
            fi
            cp "$MODEL_DIR/metadata_${MODEL_VERSION}.json" "$NODE_MODEL_DIR/" 2>/dev/null || true
        fi
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
NODE_INDEX=0
for NODE_ID in "${NODE_IDS[@]}"; do
    print_info "Launching edge node: $NODE_ID..."
    NODE_ID=$NODE_ID ros2 run lydlr_ai edge_compressor_node > /tmp/lydlr_${NODE_ID}.log 2>&1 &
    EDGE_PIDS+=($!)
    sleep 1
    print_status "$NODE_ID started (PID: ${EDGE_PIDS[$NODE_INDEX]})"
    NODE_INDEX=$((NODE_INDEX + 1))
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

# Step 12: Deploy models
print_info "Step 11: Deploying models to nodes..."
sleep 3  # Wait for nodes to be ready

for NODE_ID in "${NODE_IDS[@]}"; do
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
    NODE_INDEX=0
    for NODE_ID in "${NODE_IDS[@]}"; do
        if [ $NODE_INDEX -lt ${#EDGE_PIDS[@]} ] && kill -0 ${EDGE_PIDS[$NODE_INDEX]} 2>/dev/null; then
            echo -e "  ${GREEN}✅ $NODE_ID${NC} (PID: ${EDGE_PIDS[$NODE_INDEX]})"
        else
            echo -e "  ${RED}❌ $NODE_ID${NC} (stopped)"
        fi
        NODE_INDEX=$((NODE_INDEX + 1))
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
    for NODE_ID in "${NODE_IDS[@]}"; do
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
    for NODE_ID in "${NODE_IDS[@]}"; do
        echo "  $NODE_ID: /tmp/lydlr_${NODE_ID}.log"
    done
    echo "  Deployment: /tmp/lydlr_deployment.log"
    echo "  Coordinator: /tmp/lydlr_coordinator.log"
    
    echo ""
    echo -e "${YELLOW}Commands:${NC}"
    if [ ${#NODE_IDS[@]} -gt 0 ]; then
        FIRST_NODE="${NODE_IDS[0]}"
        echo "  View metrics: ros2 topic echo /${FIRST_NODE}/metrics"
        echo "  View compressed: ros2 topic echo /${FIRST_NODE}/compressed"
    fi
    echo "  List topics: ros2 topic list"
    echo "  List nodes: ros2 node list"
    echo "  Total nodes: ${#NODE_IDS[@]}"
    
    echo ""
    echo -e "${GREEN}System running... (Refreshing every ${MONITOR_INTERVAL}s)${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    
    sleep $MONITOR_INTERVAL
done

