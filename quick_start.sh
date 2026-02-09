#!/bin/bash
# Quick Start Script - Simplified version for testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source ROS2
source /opt/ros/humble/setup.bash 2>/dev/null || {
    echo "❌ ROS2 Humble not found. Please install ROS2 first."
    exit 1
}

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install torch torchvision torchaudio numpy tqdm lpips scikit-image psutil open3d --quiet
fi

# Build
echo "Building ROS2 package..."
colcon build --symlink-install --packages-select lydlr_ai
source install/setup.bash

# Export PYTHONPATH
export PYTHONPATH="$(pwd)/src/lydlr_ai:${PYTHONPATH}"

# Setup display for WSL
if grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null ; then
    export XDG_RUNTIME_DIR=/tmp/runtime-root
    mkdir -p $XDG_RUNTIME_DIR && chmod 700 $XDG_RUNTIME_DIR
    if ! pgrep -x "Xvfb" > /dev/null; then
        Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
    fi
    export DISPLAY=:99
fi

# Launch
echo "🚀 Launching Lydlr system..."
echo "Press Ctrl+C to stop"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $(jobs -p) 2>/dev/null || true
    wait
    exit 0
}

trap cleanup SIGINT SIGTERM

# Launch all components
ros2 run lydlr_ai synthetic_multimodal_publisher &
sleep 2

NODE_ID=node_0 ros2 run lydlr_ai edge_compressor_node &
sleep 1

NODE_ID=node_1 ros2 run lydlr_ai edge_compressor_node &
sleep 1

ros2 run lydlr_ai model_deployment_manager &
sleep 1

ros2 run lydlr_ai distributed_coordinator &
sleep 2

# Deploy models
ros2 topic pub --once /node_0/model/deploy std_msgs/String "data: 'vv1.0'" > /dev/null 2>&1
ros2 topic pub --once /node_1/model/deploy std_msgs/String "data: 'vv1.0'" > /dev/null 2>&1

echo "✅ System launched!"
echo ""
echo "Monitor with:"
echo "  ros2 topic echo /node_0/metrics"
echo "  ros2 topic list"
echo ""

# Wait
wait

