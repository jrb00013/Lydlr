# Lydlr AI — Multimodal Sensor Data Compression with Adaptive Real-Time Optimization

## Overview

Lydlr AI is an advanced ROS 2 system designed for real-time intelligent compression of multimodal sensor data, including camera images, LiDAR, IMU, and audio streams, specifically tailored for edge devices such as Raspberry Pi and NVIDIA Jetson.

This system leverages state-of-the-art neural network architectures combining convolutional encoders, temporal LSTM modeling, and reinforcement learning to dynamically adapt compression quality based on system resources and perceptual feedback. It integrates real-time quality assessment via the LPIPS metric to maintain high fidelity while optimizing bandwidth and storage.

---

## Features

- **Multimodal Input Support:** Handles RGB/monochrome images, LiDAR vectors, IMU sensor data, and audio spectrograms.
- **Neural Compression Architecture:** Convolutional encoders for spatial data, LSTM for temporal sequence modeling enabling differential compression.
- **Adaptive Compression Policy:** Reinforcement learning-based controller adjusts compression level dynamically based on CPU load, battery state, and network conditions.
- **Real-Time Quality Assessment:** Integrates LPIPS perceptual similarity metric for evaluating compression fidelity on-the-fly.
- **Synthetic Data Simulation:** ROS 2 nodes generate realistic noisy multimodal sensor data for testing and benchmarking.
- **Edge Deployment Ready:** Includes utilities for model quantization and TorchScript export targeting low-resource platforms.
- **Extensible:** Framework designed to incorporate advanced compression methods such as VAE, VQ-VAE, normalizing flows, and transformer-based models.

---

## Installation

### Prerequisites

- ROS 2 (Foxy/Galactic/Humble recommended)
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
