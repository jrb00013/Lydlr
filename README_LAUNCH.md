# 🚀 Lydlr System Launch Guide

## Quick Start

### Option 1: Full Featured Launch Script (Recommended)

```bash
cd /mnt/c/Users/josep/Documents/Lydlr/Lydlr
./launch_lydlr_system.sh
```

This script will:
- ✅ Check and setup ROS2 environment
- ✅ Activate Python virtual environment
- ✅ Install dependencies if needed
- ✅ Build ROS2 package
- ✅ Deploy models to all nodes
- ✅ Launch all components (publisher, nodes, coordinator, manager)
- ✅ Monitor performance in real-time
- ✅ Display system status dashboard

### Option 2: Quick Start Script (Simplified)

```bash
cd /mnt/c/Users/josep/Documents/Lydlr/Lydlr
./quick_start.sh
```

This is a simplified version for quick testing.

## Manual Launch

If you prefer to launch components manually:

### 1. Setup Environment

```bash
# Source ROS2
source /opt/ros/humble/setup.bash

# Activate venv
cd /mnt/c/Users/josep/Documents/Lydlr/Lydlr
source .venv/bin/activate

# Build package
colcon build --symlink-install --packages-select lydlr_ai
source install/setup.bash

# Setup PYTHONPATH
export PYTHONPATH="$(pwd)/src/lydlr_ai:${PYTHONPATH}"

# Setup display (WSL)
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p $XDG_RUNTIME_DIR && chmod 700 $XDG_RUNTIME_DIR
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

### 2. Launch Components

**Terminal 1: Synthetic Data Publisher**
```bash
ros2 run lydlr_ai synthetic_multimodal_publisher
```

**Terminal 2: Edge Node 0**
```bash
NODE_ID=node_0 ros2 run lydlr_ai edge_compressor_node
```

**Terminal 3: Edge Node 1**
```bash
NODE_ID=node_1 ros2 run lydlr_ai edge_compressor_node
```

**Terminal 4: Deployment Manager**
```bash
ros2 run lydlr_ai model_deployment_manager
```

**Terminal 5: Coordinator**
```bash
ros2 run lydlr_ai distributed_coordinator
```

### 3. Deploy Models

```bash
# Deploy to node_0
ros2 topic pub /node_0/model/deploy std_msgs/String "data: 'vv1.0'"

# Deploy to node_1
ros2 topic pub /node_1/model/deploy std_msgs/String "data: 'vv1.0'"
```

### 4. Monitor Performance

```bash
# View metrics
ros2 topic echo /node_0/metrics
ros2 topic echo /node_1/metrics

# View compressed data
ros2 topic echo /node_0/compressed

# List all topics
ros2 topic list

# List all nodes
ros2 node list
```

## Configuration

### Environment Variables

```bash
# Number of nodes (default: 2)
export NUM_NODES=2

# Model version (default: vv1.0)
export MODEL_VERSION=vv1.0
```

### Custom Model Version

```bash
MODEL_VERSION=v2.0 ./launch_lydlr_system.sh
```

## Troubleshooting

### Models Not Found

If you see "Model not found", train models first:

```bash
cd src/lydlr_ai
source ../../.venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python lydlr_ai/model/train_synthetic_models.py --epochs 20 --samples 1000
```

### ROS2 Not Found

Install ROS2 Humble:
```bash
# Follow ROS2 installation guide
# https://docs.ros.org/en/humble/Installation.html
```

### Build Errors

Clean and rebuild:
```bash
rm -rf build/ install/ log/
colcon build --symlink-install --packages-select lydlr_ai
```

### Display Issues (WSL)

Make sure Xvfb is running:
```bash
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

## Log Files

All logs are saved to `/tmp/`:
- `/tmp/lydlr_synthetic.log` - Synthetic publisher
- `/tmp/lydlr_node_0.log` - Edge node 0
- `/tmp/lydlr_node_1.log` - Edge node 1
- `/tmp/lydlr_deployment.log` - Deployment manager
- `/tmp/lydlr_coordinator.log` - Coordinator

## Performance Monitoring

The launch script provides a real-time dashboard showing:
- Node status (running/stopped)
- Compression ratios
- Latency metrics
- Quality scores
- Bandwidth usage

## Stopping the System

Press `Ctrl+C` in the launch script terminal. All nodes will be stopped automatically.

## Next Steps

1. **Customize Scripts**: Load custom Python scripts on nodes
2. **Deploy New Models**: Train and deploy updated models
3. **Scale Nodes**: Add more nodes by setting `NUM_NODES`
4. **Monitor Performance**: Use the built-in dashboard or ROS2 tools

---

**Ready to revolutionize compression!** 🚀

