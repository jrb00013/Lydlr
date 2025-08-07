# Lydlr AI — Multimodal Sensor Data Compression with Adaptive Real-Time Optimization

## Overview

Lydlr is an AI-powered compression system designed to optimize storage and transmission of multimodal sensor data in real time. It processes data streams from cameras, LiDAR, IMU, and audio sensors by encoding and fusing them into a compact latent representation using convolutional and recurrent neural networks.

The goal is to ensure an optimal balance between data quality and resource usage. 

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
