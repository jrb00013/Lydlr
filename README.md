# Lydlr AI — Multimodal Sensor Data Compression with Adaptive Real-Time Optimization

## Installation and Setup

## Architecture (deep dive)

- [MongoDB data layer](/docs/architecture/DATA_LAYER.md) — fleet collections, model registry, metrics TTL and rollups, API table endpoints  
- [ROS 2 communication and transport](/docs/architecture/ROS2_COMMUNICATION.md) — `/lydlr/**` topic graph, LYDT wire encoding, QoS, ground relay  
- [Benchmark protocol](/docs/architecture/BENCHMARK_PROTOCOL.md) — matched-rate RD, countable bits only, temporal-residual beat-JPEG axis  

**Bring up:** `./start-lydlr.sh --build -d --ros2` • **Fleet launch (after `colcon build`):** `ros2 launch lydlr_ai drone_iot_transport.launch.py`

## Beat JPEG (measurable claim)

JPEG treats each frame independently and has no temporal memory. Correlated sensor video on robots and drones carries redundancy across time that intra-frame coders cannot exploit. Lydlr’s falsifiable goal is **temporal residual coding**—spending countable bits on what changed versus a learned prediction—beating **intra JPEG at matched clip bpp** on real recorded clips, not headline percentages.

**Run the harness (numbers belong in script output, not this README):**

```bash
PYTHONPATH=ros2/src/lydlr_ai python scripts/bench_codecs.py --clip path/to/clip.npz --codecs jpeg,webp,h264,lydlr
PYTHONPATH=ros2/src/lydlr_ai python scripts/prove_beat_jpeg.py
```

Until you run those on your hardware and checkpoint, treat any RD table here as a placeholder. See [BENCHMARK_PROTOCOL.md](/docs/architecture/BENCHMARK_PROTOCOL.md) for rate-matching rules (countable payload only; entropy proxy never counted as wire bits).

## Overview

Lydlr is an AI-powered compression system designed to optimize storage and transmission of multimodal sensor data in real time. It processes data streams from cameras, LiDAR, IMU, and audio sensors by encoding and fusing them into a compact latent representation using convolutional and recurrent neural networks. The system leverages temporal context through LSTM layers to improve compression efficiency by learning patterns over time. A reinforcement learning-based controller dynamically adjusts compression levels based on system conditions such as CPU load, battery status, and network bandwidth, ensuring an optimal balance between data quality and resource usage. Additionally, a real-time quality assessment module uses perceptual metrics (LPIPS) to monitor reconstruction fidelity, enabling adaptive tuning on the fly. Synthetic sensor data streams simulate diverse environments for thorough testing and development. The entire pipeline is designed for deployment on edge devices like Raspberry Pi or NVIDIA Jetson, with model quantization and export capabilities for efficient execution on constrained hardware.

## Real-World Applications

Lydlr addresses critical challenges in modern sensor data processing across multiple industries:

### Autonomous Vehicles
Compress sensor data from cameras, LiDAR, and IMU sensors before transmission to cloud infrastructure. Matched-rate benchmarks against JPEG/WebP on the same clips quantify bandwidth savings; run `scripts/bench_codecs.py` and `scripts/prove_beat_jpeg.py` rather than assuming a fixed reduction percentage. Enables efficient data offloading from vehicles to central processing systems without overwhelming network infrastructure.

### Drones
Reduce bandwidth consumption for real-time video and LiDAR streaming during flight operations. Critical for long-range missions where maintaining communication links is essential. Allows operators to receive high-quality sensor feeds even over limited bandwidth connections, enabling extended operational range and improved mission success rates.

### Robotics
Optimize storage for long-duration data collection in research and industrial applications. Robots can operate for extended periods without storage limitations, capturing comprehensive sensor data for analysis, training, and system improvement. Essential for autonomous systems that need to learn from extended operational periods.

### Edge AI
Enable AI processing on devices with limited bandwidth and computational resources. By compressing multimodal sensor data at the edge, systems can reduce transmission costs, improve response times, and enable real-time decision-making without constant cloud connectivity. Critical for applications requiring low latency and privacy-preserving local processing.

### IoT Systems
Compress sensor data from distributed IoT networks for efficient transmission to central monitoring systems. Reduces network congestion, extends battery life of edge devices, and enables cost-effective scaling of sensor networks. Essential for smart cities, industrial monitoring, and environmental sensing applications where thousands of devices transmit data continuously.

## History
Developed to meet the growing need for lightweight AI at the edge, Lydlr represents part of a broader trend in robotics and embedded AI. It enables high-performance multimodal perception and reasoning within strict energy and bandwidth constraints, advancing real-time autonomy and distributed intelligence in the ROS2 ecosystem.


## Impact

Lydlr targets lower **countable** wire rate at matched distortion on multimodal sensor clips, with an RL controller tuning quality under bandwidth and compute constraints. **Do not cite fixed bandwidth, FPS, CPU, or LPIPS figures from this README**—they are not verified here.

| What we measure | How |
|-----------------|-----|
| Rate vs JPEG/WebP/H.264-intra at matched bpp | `scripts/bench_codecs.py` → `scripts/results/rd_curve.json` |
| Beat-JPEG regression / temporal-residual claim | `scripts/prove_beat_jpeg.py` (when present) |
| Protocol (countable bits, matched-rate) | [docs/architecture/BENCHMARK_PROTOCOL.md](/docs/architecture/BENCHMARK_PROTOCOL.md) |

Example (after you have a recorded clip NPZ and optional checkpoint):

```bash
PYTHONPATH=ros2/src/lydlr_ai python scripts/bench_codecs.py \
  --clip path/to/real_clip.npz --codecs jpeg,lydlr --target-bpps 0.05,0.1,0.2,0.5
```

The printed markdown table and JSON are the source of truth for PSNR/SSIM at matched rate. LPIPS is optional (`--lpips`) and requires PyTorch.

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
- Press F5 or green  button  

Add breakpoints in `optimizer_node.py`.

Stop both terminals with Ctrl+C when done.

### SECTION 11 — File Structure Reference

- `optimizer_node.py`: `ros2/src/lydlr_ai/lydlr_ai/optimizer_node.py`  
- `synthetic_publisher.py`: `ros2/src/lydlr_ai/lydlr_ai/synthetic_multimodal_publisher.py`

### SECTION 12 — Optional Debug Tips

- Use **Debug Console** to inspect variables  
- Step over: F10  
- Step in: F11  
- Enable `"justMyCode": false` to debug libraries
![Image](https://images.openai.com/static-rsc-4/vq7a8SGRsFQAAGV9yusbgcMyrVwKDGnfPpDBPV8KSRFxUdSM4AzE0Ii421kAw8OcXYHIl3QUcrEAMCRF9f9XEaRgMxo-byKJyGDeybzKfwsqMmd5FkKGtpDuXlPi9CrZcLHFqP6Y6E4Dkxh6io3d2z_UdszrYrBVlcvH1sRy-cR9YYV1hg9_uJXxVds6Zd-i?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/q1c6uKM-bUFUOzoznSh0-oX60mnoVZGUzkqQ0TIrIlvqSxAmcA_dyXOfGHKcrBKaIraz7UN143vC-fPko0w9Qg3YO5T8czjiYCOUXXGxhZS0LAeRorJrP9_pI1R-zNgmvqd0zL2X0GFYnBsfQfTeMOBKgH82X3miIoVIYE70EFVK-kDwvPSYgOv1nmo0CbTe?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/hFjrPaFdOpq4X5HuwYWqJZBZEqp7uJ5oIfoJ6ETGjUD2HMXyEBaZCJDYtplJkKAdWHw4qjX0AnE8Jxuh3yL42Nd3-Ynqprh_HNbob_IMtiVsYXpRPg9w9TTGnOQkCkfmSc1HrkB0ovsecAJTfbVKihdwf8EVE3PoJlp5AcfnQXzq6o3poda2f6YnuLJeC9g8?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/JYvJkvTYsxKruuQ1gt985ADfa4C7BPZp8q-5uVnr5rSFtcx-1XGyrRJ1ei2NzF7z_YPgRole5OzTefEpGOQUxezL5oCxk4xbTC62ObRRVFvlk7VaM8RRNf7GbjfMeAe--YuXXkLytj6ymVXvr8BOmwUV5v5gib6exetBZ4vLzoqaGktVHat9D0gNZcpnT2EF?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/Q2CaJfwlcbqNpL7S4ljZc8egr-bRnzJIK5osrs1diA-Tv3hLQqwOrgdX65iVeL-0defo6IQjeODXgTdGBBNF5FNMn5h0nElVf_RXZ1jo1x8Oge8F231mD12Qj7TKdybftbisYhE3Kr3Yjn_jPPrBeHgdy0gDedVZuHwQNiOkIUXOlmeVOgxR6a9DW5rhkJZM?purpose=fullsize)

## Lydlr

Lydlr is a software framework designed for multimodal artificial intelligence (AI) compression and deployment at the edge, integrated with the Robot Operating System 2 ecosystem. It aims to optimize the performance of machine-learning models across heterogeneous edge devices where bandwidth, memory, and compute resources are limited.

### Key facts

* **Primary domain:** Multimodal edge AI and robotics
* **Integration:** Built around ROS2 nodes and message-passing
* **Function:** Compresses and optimizes multimodal data streams
* **Deployment target:** Resource-constrained edge devices
* **Focus:** Efficient inference for robotics and IoT applications

### Architecture and Design

Lydlr operates as a modular compression and inference layer within ROS2. It enables efficient encoding and transmission of multimodal data—such as vision, audio, and sensor inputs—between distributed ROS2 nodes. By employing adaptive quantization and neural compression, Lydlr reduces model size and latency while preserving task performance. Its containerized components can be orchestrated alongside standard ROS2 packages.

### Use Cases

The system supports robotics and IoT applications where real-time decision-making is required without relying heavily on cloud computation. Example use cases include autonomous mobile robots, industrial inspection systems, and smart surveillance networks. Lydlr allows these devices to process sensor fusion tasks locally while keeping communication overhead low.

### Technical Features

Lydlr leverages model pruning, quantization-aware training, and knowledge distillation techniques to achieve efficient model deployment. It also supports runtime optimization for hardware accelerators such as GPUs, NPUs, and FPGAs commonly used in edge computing. Compatibility with standard ROS2 message types ensures seamless integration into existing robotics pipelines.

### Development and Impact

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3).

You are free to use, copy, modify, and distribute this software under the terms of the GPLv3 license.

A copy of the GNU General Public License v3.0 is included in this repository or can be found at:

https://www.gnu.org/licenses/gpl-3.0.en.html

---

**Disclaimer:**

This program is distributed in the hope that it will be useful, but **WITHOUT ANY WARRANTY**; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

