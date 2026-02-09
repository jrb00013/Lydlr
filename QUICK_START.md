# 🚀 Lydlr Revolutionary Compression System - Quick Start Guide

## Prerequisites

```bash
# ROS 2 Humble
# Python 3.10+
# PyTorch
# CUDA (optional, for GPU)
```

## Installation

```bash
cd ~/lydlr_ws
colcon build --symlink-install --packages-select lydlr_ai
source install/setup.bash
```

## 5-Minute Quick Start

### Step 1: Train Models (2 minutes)

```bash
python train_synthetic_models.py --epochs 10 --samples 500
```

This creates:
- `models/compressor_v{version}.pth`
- `models/sensor_motor_v{version}.pth`

### Step 2: Launch System (1 minute)

```bash
# Terminal 1: Launch all components
ros2 launch lydlr_ai revolutionary_system.launch.py

# Or launch individually:
# Terminal 1: Synthetic data publisher
ros2 run lydlr_ai synthetic_multimodal_publisher

# Terminal 2: Edge node 0
NODE_ID=node_0 ros2 run lydlr_ai edge_compressor_node

# Terminal 3: Edge node 1
NODE_ID=node_1 ros2 run lydlr_ai edge_compressor_node

# Terminal 4: Coordinator
ros2 run lydlr_ai distributed_coordinator

# Terminal 5: Deployment manager
ros2 run lydlr_ai model_deployment_manager
```

### Step 3: Deploy Models (30 seconds)

```bash
# Deploy to specific node
ros2 topic pub /node_0/model/deploy std_msgs/String "data: 'v1.0'"

# Or let deployment manager auto-deploy latest
# (happens automatically)
```

### Step 4: Load Scripts (30 seconds)

```bash
# Load custom processing script
ros2 topic pub /node_0/script/load std_msgs/String "data: 'custom_processor'"
```

### Step 5: Monitor (1 minute)

```bash
# Monitor metrics
ros2 topic echo /node_0/metrics

# Monitor compressed data
ros2 topic echo /node_0/compressed

# Check system status
ros2 topic list | grep node
```

## Common Commands

### Training
```bash
# Basic training
python train_synthetic_models.py

# Custom training
python train_synthetic_models.py \
    --epochs 20 \
    --samples 1000 \
    --batch-size 4 \
    --version "v2.0"
```

### Deployment
```bash
# Deploy to all nodes
ros2 topic pub /model/deploy_all std_msgs/String "data: 'v1.0'"

# Deploy to specific node
ros2 topic pub /node_0/model/deploy std_msgs/String "data: 'v1.0'"

# List available models
# (Check models/ directory)
ls models/compressor_v*.pth
```

### Scripts
```bash
# Load script
ros2 topic pub /node_0/script/load std_msgs/String "data: 'script_name'"

# Scripts location
ls scripts/node_0/
```

### Monitoring
```bash
# Node metrics
ros2 topic echo /node_0/metrics

# System performance
ros2 topic echo /coordinator/performance

# Deployment status
ros2 topic echo /deployment_manager/status
```

## Troubleshooting

### Models not loading?
```bash
# Check model directory
ls models/node_0/

# Check model version format
# Should be: compressor_v{version}.pth
```

### Scripts not executing?
```bash
# Check script location
ls scripts/node_0/

# Check script name matches
# Should be: {script_name}.py
```

### Nodes not communicating?
```bash
# Check ROS2 discovery
ros2 node list

# Check topics
ros2 topic list

# Check node status
ros2 node info /edge_compressor_node_0
```

## Performance Tuning

### Compression Level
- High compression (0.9): Low bandwidth, lower quality
- Medium compression (0.7): Balanced
- Low compression (0.5): High quality, more bandwidth

### Bandwidth Allocation
- Edit `distributed_coordinator.py`:
  ```python
  self.total_bandwidth = 100.0  # Mbps
  ```

### Compression Frequency
- Edit `edge_compressor_node.py`:
  ```python
  self.compression_timer = self.create_timer(0.1, ...)  # 10 Hz
  ```

## Next Steps

1. **Read Architecture**: See `ARCHITECTURE.md`
2. **Design Details**: See `DESIGN_WALKTHROUGH.md`
3. **Create Custom Scripts**: See `scripts/example_scripts/`
4. **Train on Your Data**: Modify `train_synthetic_models.py`

## Support

- Check documentation in `ARCHITECTURE.md`
- Review code comments
- Check ROS2 logs: `ros2 topic echo /rosout`

---

**Ready to compress?** 🚀

