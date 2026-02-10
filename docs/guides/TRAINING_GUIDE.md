# Lydlr Training Guide

This guide explains how to train the Lydlr AI compression model using both synthetic and real sensor data.

## Overview

Lydlr is an AI-powered compression system that processes multimodal sensor data (camera, LiDAR, IMU, audio) and compresses it using neural networks. The training process involves:

1. **Data Collection**: Recording sensor data from your robot
2. **Data Preparation**: Processing and formatting the data for training
3. **Model Training**: Training the compression model
4. **Model Deployment**: Loading the trained model in your ROS2 nodes

## Quick Start

### 1. Test with Synthetic Data (Immediate)

```bash
cd ~/lydlr_ws
source install/setup.bash

# Test the data loader with synthetic data
python3 scripts/data_loader.py

# Train with synthetic data (this will work immediately)
python3 ros2/src/lydlr_ai/train.py
```

### 2. Collect Real Training Data

```bash
# Launch the data collection node
ros2 launch lydlr_ai collect_training_data.launch.py

# Or run directly
ros2 run lydlr_ai collect_training_data.py
```

### 3. Train with Real Data

Edit `ros2/src/lydlr_ai/train.py` and change:
```python
USE_SYNTHETIC = False  # Use real data instead of synthetic
```

Then run:
```bash
python3 ros2/src/lydlr_ai/train.py
```

## Data Requirements

### Expected Data Shapes

- **Images**: `[B, T, 3, 480, 640]` - RGB images (batch, time, channels, height, width)
- **LiDAR**: `[B, T, 1024, 3]` - Point clouds (batch, time, 1024 points, 3D coordinates)
- **IMU**: `[B, T, 6]` - 6-axis IMU (batch, time, [ax, ay, az, gx, gy, gz])
- **Audio**: `[B, T, 16384]` - Audio spectrograms (batch, time, 128x128 flattened)

### Data Sources

#### Option 1: Your Robot's Sensors
- **Camera**: `/camera/image_raw` (sensor_msgs/Image)
- **LiDAR**: `/lidar/points` (sensor_msgs/PointCloud2)
- **IMU**: `/imu/data` (sensor_msgs/Imu)
- **Audio**: Audio topic (you'll need to add this)

#### Option 2: Public Datasets
- **KITTI**: Autonomous driving dataset
- **NuScenes**: Multimodal autonomous driving
- **TUM RGB-D**: SLAM datasets with IMU

#### Option 3: Synthetic Data (Current)
- Random data for initial development and testing
- Gradually replace with real data

## Data Collection Process

### 1. Configure Your Robot

Ensure your robot publishes sensor data on these topics:
```bash
# Check available topics
ros2 topic list

# Check topic types
ros2 topic info /camera/image_raw
ros2 topic info /lidar/points
ros2 topic info /imu/data
```

### 2. Adjust Topic Names

Edit `scripts/collect_training_data.py` and update the topic names:
```python
self.image_sub = self.create_subscription(
    Image, '/your_camera_topic', self.image_callback, 10)
self.lidar_sub = self.create_subscription(
    PointCloud2, '/your_lidar_topic', self.lidar_callback, 10)
self.imu_sub = self.create_subscription(
    Imu, '/your_imu_topic', self.imu_callback, 10)
```

### 3. Collect Data

```bash
# Start data collection
ros2 launch lydlr_ai collect_training_data.launch.py

# Drive your robot around to collect diverse data
# The system will automatically save sequences to ~/lydlr_ws/data/training_data/
```

### 4. Verify Data Collection

```bash
# Check collected data
ls ~/lydlr_ws/data/training_data/
# You should see: sequence_YYYYMMDD_HHMMSS/

# Check data contents
ls ~/lydlr_ws/data/training_data/sequence_*/
# Should contain: images.npy, lidar.npy, imu.npy, audio.npy, metadata.json
```

## Training Process

### 1. Configure Training Parameters

Edit `ros2/src/lydlr_ai/train.py`:
```python
# Training hyperparameters
BATCH_SIZE = 2          # Adjust based on your GPU memory
SEQ_LEN = 4             # Sequence length for temporal modeling
EPOCHS = 10             # Number of training epochs
LR = 1e-4               # Learning rate

# Data configuration
DATA_DIR = "~/lydlr_ws/data/training_data"
USE_SYNTHETIC = False   # Set to False for real data
```

### 2. Start Training

```bash
cd ~/lydlr_ws
source install/setup.bash

# Train the model
python3 ros2/src/lydlr_ai/train.py
```

### 3. Monitor Training

The training script will show:
- Loss values (reconstruction, perceptual, rate)
- Progress per epoch
- Model saving confirmation

## Model Deployment

### 1. Load Trained Model

After training, the model is saved as `multimodal_compressor.pth`. Load it in your ROS2 nodes:

```python
# In your optimizer_node.py
import torch
from lydlr_ai.model.compressor import MultimodalCompressor

class StorageOptimizerNode(Node):
    def __init__(self):
        super().__init__('storage_optimizer')
        
        # Load trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MultimodalCompressor().to(self.device)
        
        # Load trained weights
        model_path = 'path/to/multimodal_compressor.pth'
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
```

### 2. Use for Inference

```python
def sensor_callback(self, img, lidar, imu, audio):
    # Preprocess sensor data
    img_tensor = self.preprocess_image(img)
    lidar_tensor = self.preprocess_lidar(lidar)
    imu_tensor = self.preprocess_imu(imu)
    audio_tensor = self.preprocess_audio(audio)
    
    # Run compression
    with torch.no_grad():
        encoded, decoded, _, recon_img = self.model(
            img_tensor, lidar_tensor, imu_tensor, audio_tensor,
            hidden_state=None, compression_level=0.8
        )
    
    # Use compressed data for storage/transmission
    self.publish_compressed_data(encoded)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure your Python path includes the scripts directory
2. **Shape Mismatches**: Check that your data matches expected shapes
3. **Memory Issues**: Reduce batch size or sequence length
4. **Topic Not Found**: Verify your sensor topics are publishing

### Debug Commands

```bash
# Check ROS2 topics
ros2 topic list
ros2 topic echo /camera/image_raw --once
ros2 topic echo /lidar/points --once
ros2 topic echo /imu/data --once

# Check data collection
ros2 node list
ros2 node info /training_data_collector

# Test data loader
python3 scripts/data_loader.py
```

## Next Steps

1. **Validate IMU and LiDAR inputs** - Ensure data quality
2. **Increase efficiency** - Optimize compression ratios
3. **Try wild future ideas** - Experiment with new architectures

## File Structure

```
lydlr_ws/
├── ros2/src/lydlr_ai/
│   ├── train.py                    # Main training script
│   └── lydlr_ai/model/
│       └── compressor.py           # Model architecture
├── scripts/
│   ├── collect_training_data.py    # Data collection
│   └── data_loader.py             # Data loading utilities
├── launch/
│   └── collect_training_data.launch.py
├── data/
│   └── training_data/             # Collected sensor data
└── TRAINING_GUIDE.md              # This file
```

## Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify your ROS2 environment is properly set up
3. Ensure all dependencies are installed
4. Check the data shapes match expected formats
