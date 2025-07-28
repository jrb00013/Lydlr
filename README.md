# Lydlr AI — Multimodal Sensor Data Compression with Adaptive Real-Time Optimization

## Overview

Lydlr is an AI-powered compression system designed to optimize storage and transmission of multimodal sensor data in real time. It processes data streams from cameras, LiDAR, IMU, and audio sensors by encoding and fusing them into a compact latent representation using convolutional and recurrent neural networks. The system leverages temporal context through LSTM layers to improve compression efficiency by learning patterns over time. A reinforcement learning-based controller dynamically adjusts compression levels based on system conditions such as CPU load, battery status, and network bandwidth, ensuring an optimal balance between data quality and resource usage. Additionally, a real-time quality assessment module uses perceptual metrics (LPIPS) to monitor reconstruction fidelity, enabling adaptive tuning on the fly. Synthetic sensor data streams simulate diverse environments for thorough testing and development. The entire pipeline is designed for deployment on edge devices like Raspberry Pi or NVIDIA Jetson, with model quantization and export capabilities for efficient execution on constrained hardware.

## Installation

### Prerequisites

- ROS 2 (Humble or Foxy recommended)
- Python 3.8+
- PyTorch
- lpips (`pip install lpips`)
- psutil (`pip install psutil`)

### Setup

Clone the repository into your ROS 2 workspace:

```bash
cd ~/ros2_ws/src
git clone https://github.com/yourusername/lydlr_ai.git
cd ~/ros2_ws
colcon build
source install/setup.bash
