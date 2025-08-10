# Lydlr AI — Multimodal Sensor Data Compression with Adaptive Real-Time Optimization

## Installation and Setup

## Overview
_Target: macOS + Docker + ROS 2 Humble + Python venv_

Lydlr is an AI-powered compression system designed to optimize storage and transmission of multimodal sensor data in real time. It processes data streams from cameras, LiDAR, IMU, and audio sensors by encoding and fusing them into a compact latent representation using convolutional and recurrent neural networks. The system leverages temporal context through LSTM layers to improve compression efficiency by learning patterns over time. A reinforcement learning-based controller dynamically adjusts compression levels based on system conditions such as CPU load, battery status, and network bandwidth, ensuring an optimal balance between data quality and resource usage. Additionally, a real-time quality assessment module uses perceptual metrics (LPIPS) to monitor reconstruction fidelity, enabling adaptive tuning on the fly. Synthetic sensor data streams simulate diverse environments for thorough testing and development. The entire pipeline is designed for deployment on edge devices like Raspberry Pi or NVIDIA Jetson, with model quantization and export capabilities for efficient execution on constrained hardware.

### ROS 2 + Docker Workspace Setup Guide  
_Target: macOS + Docker + ROS 2 Humble + Python venv_

### REQUIREMENTS:
- Docker installed  
- Homebrew + Python 3 installed  
- Basic understanding of terminal and ROS 2

### SECTION 0 — Pull Docker Image

```bash
docker pull osrf/ros:humble-desktop
```

### SECTION 1 — Build & Run ROS 2 Container

Build with Xvfb (GUI headless display):

```bash
docker build -t ros2_xvfb .
```

Run container with volume and full setup:

```bash
docker run -it \
    --name ros2_ai \
    -v ~/Documents/lydlr:/root/lydlr \
    ros2_xvfb \
    bash -c "export PYTHONPATH=\$PYTHONPATH:/root/lydlr/lydlr_ws/src:/root/lydlr/lydlr_ws/.venv/lib/python3.10/site-packages && \
             export XDG_RUNTIME_DIR=/tmp/runtime-root && \
             mkdir -p \$XDG_RUNTIME_DIR && chmod 700 \$XDG_RUNTIME_DIR && \
             Xvfb :99 -screen 0 1024x768x24 & export DISPLAY=:99 && \
             source /opt/ros/humble/setup.bash && \
             cd /root/lydlr/lydlr_ws && exec bash"
```

Alternative (no Xvfb, uses host display):

```bash
docker run -it \
    --name ros2_ai \
    -e DISPLAY=host.docker.internal:0 \
    -v ~/Documents/lydlr:/root/lydlr \
    osrf/ros:humble-desktop
```

### SECTION 2 — Reconnecting to Container

Re-enter running container:

```bash
docker exec -it ros2_ai bash -c "cd /root/lydlr/lydlr_ws && bash"
```

Restart and attach:

```bash
docker start -ai ros2_ai
```

Stop and remove container:

```bash
docker stop ros2_ai
docker rm ros2_ai
```

### SECTION 3 — Python Virtual Environment

Create venv (run once):

```bash
cd ~/lydlr/lydlr_ws
python3 -m venv .venv
```

Activate venv:

```bash
source ~/lydlr/lydlr_ws/.venv/bin/activate
```

### SECTION 4 — ROS 2 & Workspace Setup

Source ROS 2 environment:

```bash
source /opt/ros/humble/setup.bash
```

Create workspace:

```bash
mkdir -p /root/lydlr/lydlr_ws/src
cd /root/lydlr/lydlr_ws
```

Create package:

```bash
cd src
ros2 pkg create --build-type ament_python \
    --dependencies rclpy std_msgs sensor_msgs \
    -- lydlr_ai
```

### SECTION 5 — Building & Sourcing

Build (after creating or editing any package/node):

```bash
cd /root/lydlr/lydlr_ws
colcon build --symlink-install --packages-select lydlr_ai
```

Source after build:

```bash
source install/setup.bash
```

Reactivate venv (if needed):

```bash
source .venv/bin/activate
```

### SECTION 6 — Running a Node

Add entry to `setup.py`:

```python
entry_points={
    'console_scripts': [
        'your_node_name = lydlr_ai.your_node_file_name:main',
    ],
}
```

Rebuild and source:

```bash
colcon build --symlink-install --packages-select lydlr_ai
source install/setup.bash
```

Environment setup before running:

```bash
export PYTHONPATH=$PYTHONPATH:/root/lydlr/lydlr_ws/src:/root/lydlr/lydlr_ws/.venv/lib/python3.10/site-packages
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p $XDG_RUNTIME_DIR && chmod 700 $XDG_RUNTIME_DIR
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

Run node:

```bash
ros2 run lydlr_ai your_node_name
```

### SECTION 7 — Xvfb (Headless GUI)

Start Xvfb display:

```bash
xvfb-run -s "-screen 0 1024x768x24" bash
```

### SECTION 8 — Clean Workspace

Clean cache for package:

```bash
colcon build --packages-select lydlr_ai --cmake-clean-cache
```

Full clean:

```bash
cd ~/lydlr/lydlr_ws
rm -rf build/ install/ log/
```

### SECTION 9 — Test Publishing

```bash
ros2 topic pub /camera/image_raw sensor_msgs/msg/Image "{
  header: {frame_id: 'fake_camera'},
  height: 3,
  width: 2,
  encoding: 'mono8',
  is_bigendian: 0,
  step: 2,
  data: [0, 50, 100, 150, 200, 255]
}"
```

### SECTION 10 — Debugging optimizer_node.py

**VS Code: Reopen container**  
- Ctrl+Shift+P → "Docker: Reopen in Container"  
- Terminal shows: "Connected to dev container: lydlr_ws"

**Terminal 1 — build & env**

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select lydlr_ai
source install/setup.bash
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/root/lydlr/lydlr_ws/src:/root/lydlr/lydlr_ws/.venv/lib/python3.10/site-packages
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p $XDG_RUNTIME_DIR && chmod 700 $XDG_RUNTIME_DIR
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

Run synthetic data publisher:

```bash
ros2 run lydlr_ai synthetic_multimodal_publisher.py
```

**Terminal 2 — debugging**

```bash
xvfb-run -s "-screen 0 1024x768x24" bash
```

VS Code:  
- Ctrl+Shift+D → "Run & Debug"  
- Select: "Debug ROS2 optimizer_node (launch)"  
- Press F5 or green ▶️ button  

Add breakpoints in `optimizer_node.py`.

Stop both terminals with Ctrl+C when done.

### SECTION 11 — File Structure Reference

- `optimizer_node.py`: `src/lydlr_ai/lydlr_ai/optimizer_node.py`  
- `synthetic_publisher.py`: `src/lydlr_ai/lydlr_ai/synthetic_multimodal_publisher.py`

### SECTION 12 — Optional Debug Tips

- Use **Debug Console** to inspect variables  
- Step over: F10  
- Step in: F11  
- Enable `"justMyCode": false` to debug libraries

