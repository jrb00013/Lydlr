# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Revolutionary Edge Compression Node
- Real-time Python script execution
- Dynamic model deployment
- Sensor/Motor data compression
- Adaptive bandwidth reduction
"""

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, PointCloud2, Imu
    from std_msgs.msg import Float32MultiArray, String, UInt8MultiArray
    from geometry_msgs.msg import Twist
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    # Dummy classes for when ROS2 is not available (e.g., during training)
    class Node:
        pass
    class Image:
        pass
    class PointCloud2:
        pass
    class Imu:
        pass
    class Float32MultiArray:
        pass
    class String:
        pass
    class UInt8MultiArray:
        pass
    class Twist:
        pass

import torch
import torch.nn as nn
import numpy as np
import importlib.util
import sys
import os
import json
import time
import threading
import queue
from pathlib import Path
import pickle
import zlib
from typing import Dict, Any, Optional, Callable

try:
    import psutil
except ImportError:
    psutil = None

from lydlr_ai.model.compressor import EnhancedMultimodalCompressor
try:
    from lydlr_ai.model.quality_predictor import QualityPredictor
except ImportError:
    QualityPredictor = None


class ModelRegistry:
    """Manages model versions and hot-swapping"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.current_model = None
        self.model_version = None
        self.model_metadata = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lock = threading.Lock()
    
    def load_model(self, version: str) -> bool:
        """Load a specific model version"""
        try:
            model_path = self.model_dir / f"compressor_v{version}.pth"
            metadata_path = self.model_dir / f"metadata_v{version}.json"
            
            if not model_path.exists():
                return False
            
            with self.lock:
                # Load metadata
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        self.model_metadata = json.load(f)
                
                # Load model
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Initialize model architecture
                model = EnhancedMultimodalCompressor().to(self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                self.current_model = model
                self.model_version = version
                
            return True
        except Exception as e:
            print(f"Error loading model v{version}: {e}")
            return False
    
    def get_model(self):
        """Get current model (thread-safe)"""
        with self.lock:
            return self.current_model
    
    def list_versions(self) -> list:
        """List all available model versions"""
        versions = []
        for f in self.model_dir.glob("compressor_v*.pth"):
            version = f.stem.split("_v")[1]
            versions.append(version)
        return sorted(versions, reverse=True)


class ScriptExecutor:
    """Executes Python scripts dynamically in real-time"""
    
    def __init__(self, script_dir: str = "scripts"):
        self.script_dir = Path(script_dir)
        self.script_dir.mkdir(exist_ok=True)
        self.loaded_scripts: Dict[str, Any] = {}
        self.script_context = {
            'torch': torch,
            'np': np,
            'rclpy': rclpy,
        }
    
    def load_script(self, script_name: str) -> bool:
        """Load a Python script dynamically"""
        script_path = self.script_dir / f"{script_name}.py"
        
        if not script_path.exists():
            return False
        
        try:
            spec = importlib.util.spec_from_file_location(script_name, script_path)
            module = importlib.util.module_from_spec(spec)
            
            # Inject context
            module.__dict__.update(self.script_context)
            
            spec.loader.exec_module(module)
            self.loaded_scripts[script_name] = module
            
            return True
        except Exception as e:
            print(f"Error loading script {script_name}: {e}")
            return False
    
    def execute_function(self, script_name: str, function_name: str, *args, **kwargs):
        """Execute a function from a loaded script"""
        if script_name not in self.loaded_scripts:
            if not self.load_script(script_name):
                return None
        
        module = self.loaded_scripts[script_name]
        if hasattr(module, function_name):
            func = getattr(module, function_name)
            return func(*args, **kwargs)
        return None


class SensorMotorCompressor(nn.Module):
    """Advanced compressor for sensor and motor data"""
    
    def __init__(self, sensor_dim=256, motor_dim=6, latent_dim=64):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.motor_dim = motor_dim
        self.latent_dim = latent_dim
        
        # Sensor encoder
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, latent_dim)
        )
        
        # Motor encoder (for motor commands)
        self.motor_encoder = nn.Sequential(
            nn.Linear(motor_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim // 2)
        )
        
        # Temporal compression
        self.temporal_compressor = nn.LSTM(
            latent_dim + latent_dim // 2, 
            latent_dim, 
            batch_first=True,
            num_layers=2
        )
        
        # Adaptive compression controller
        self.compression_controller = nn.Sequential(
            nn.Linear(latent_dim + 1, 64),  # +1 for bandwidth signal
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Decoders
        self.sensor_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, sensor_dim)
        )
        
        self.motor_decoder = nn.Sequential(
            nn.Linear(latent_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, motor_dim)
        )
    
    def forward(self, sensor_data, motor_data=None, bandwidth_signal=1.0, hidden_state=None):
        # Encode
        sensor_encoded = self.sensor_encoder(sensor_data)
        
        if motor_data is not None:
            motor_encoded = self.motor_encoder(motor_data)
            combined = torch.cat([sensor_encoded, motor_encoded], dim=-1)
        else:
            combined = sensor_encoded
        
        # Temporal compression
        combined_seq = combined.unsqueeze(1)  # Add time dimension
        temporal_out, hidden_state = self.temporal_compressor(combined_seq, hidden_state)
        temporal_out = temporal_out.squeeze(1)
        
        # Adaptive compression based on bandwidth
        bandwidth_tensor = torch.full((temporal_out.size(0), 1), bandwidth_signal, 
                                     device=temporal_out.device)
        compression_level = self.compression_controller(
            torch.cat([temporal_out, bandwidth_tensor], dim=-1)
        )
        
        # Apply compression
        compressed = temporal_out * compression_level
        
        # Decode
        sensor_decoded = self.sensor_decoder(compressed[:, :self.latent_dim])
        
        if motor_data is not None:
            motor_decoded = self.motor_decoder(compressed[:, self.latent_dim:])
        else:
            motor_decoded = None
        
        return compressed, sensor_decoded, motor_decoded, hidden_state, compression_level


class EdgeCompressorNode(Node):
    """Revolutionary edge compression node with real-time capabilities"""
    
    def __init__(self, node_name: str = "edge_compressor", node_id: str = "node_0"):
        super().__init__(node_name)
        self.node_id = node_id
        
        # Model registry
        self.model_registry = ModelRegistry(model_dir=f"models/{node_id}")
        
        # Script executor
        self.script_executor = ScriptExecutor(script_dir=f"scripts/{node_id}")
        
        # Compression models
        self.multimodal_compressor = None
        self.sensor_motor_compressor = SensorMotorCompressor().to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Quality predictor
        self.quality_predictor = QualityPredictor()
        
        # State
        self.hidden_state = None
        self.bandwidth_estimate = 1.0  # Normalized bandwidth (0-1)
        self.compression_stats = {
            'total_in': 0,
            'total_out': 0,
            'compression_ratio': 0.0,
            'latency_ms': 0.0
        }
        
        # Data buffers
        self.sensor_buffer = queue.Queue(maxsize=10)
        self.motor_buffer = queue.Queue(maxsize=10)
        
        # Subscribers - Sensor inputs
        self.create_subscription(Image, '/camera/image_raw', 
                                self.image_callback, 10)
        self.create_subscription(Float32MultiArray, '/lidar/data',
                                self.lidar_callback, 10)
        self.create_subscription(Float32MultiArray, '/imu/data',
                                self.imu_callback, 10)
        self.create_subscription(Float32MultiArray, '/audio/data',
                                self.audio_callback, 10)
        
        # Subscribers - Motor/Actuator inputs
        self.create_subscription(Twist, '/cmd_vel',
                                self.motor_callback, 10)
        
        # Subscribers - Model deployment
        self.create_subscription(String, '/model/deploy',
                                self.model_deploy_callback, 10)
        self.create_subscription(String, '/script/load',
                                self.script_load_callback, 10)
        
        # Publishers - Compressed data
        self.compressed_pub = self.create_publisher(
            UInt8MultiArray, f'/{node_id}/compressed', 10
        )
        self.metrics_pub = self.create_publisher(
            Float32MultiArray, f'/{node_id}/metrics', 10
        )
        
        # Publishers - Decompressed (for other nodes)
        self.decompressed_pub = self.create_publisher(
            Float32MultiArray, f'/{node_id}/decompressed', 10
        )
        
        # Timer for real-time compression
        self.compression_timer = self.create_timer(0.1, self.compress_loop)  # 10 Hz
        
        # Timer for bandwidth monitoring
        self.bandwidth_timer = self.create_timer(1.0, self.update_bandwidth)
        
        # Load latest model
        versions = self.model_registry.list_versions()
        if versions:
            self.model_registry.load_model(versions[0])
            self.multimodal_compressor = self.model_registry.get_model()
        
        self.get_logger().info(f"🚀 Edge Compressor Node {node_id} initialized")
        self.get_logger().info(f"   Available models: {versions}")
    
    def image_callback(self, msg):
        """Process camera image"""
        try:
            # Convert ROS Image to tensor
            img_np = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == 'rgb8':
                img_np = img_np.reshape(msg.height, msg.width, 3)
                img_np = img_np.astype(np.float32) / 255.0
                img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0)
            else:
                return
            
            self.sensor_buffer.put({
                'type': 'image',
                'data': img_tensor,
                'timestamp': time.time()
            })
        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")
    
    def lidar_callback(self, msg):
        """Process LiDAR data"""
        try:
            lidar_data = np.array(msg.data, dtype=np.float32)
            lidar_tensor = torch.tensor(lidar_data).unsqueeze(0)
            
            self.sensor_buffer.put({
                'type': 'lidar',
                'data': lidar_tensor,
                'timestamp': time.time()
            })
        except Exception as e:
            self.get_logger().error(f"LiDAR processing error: {e}")
    
    def imu_callback(self, msg):
        """Process IMU data"""
        try:
            imu_data = np.array(msg.data, dtype=np.float32)
            imu_tensor = torch.tensor(imu_data).unsqueeze(0)
            
            self.sensor_buffer.put({
                'type': 'imu',
                'data': imu_tensor,
                'timestamp': time.time()
            })
        except Exception as e:
            self.get_logger().error(f"IMU processing error: {e}")
    
    def audio_callback(self, msg):
        """Process audio data"""
        try:
            audio_data = np.array(msg.data, dtype=np.float32)
            audio_tensor = torch.tensor(audio_data).unsqueeze(0)
            
            self.sensor_buffer.put({
                'type': 'audio',
                'data': audio_tensor,
                'timestamp': time.time()
            })
        except Exception as e:
            self.get_logger().error(f"Audio processing error: {e}")
    
    def motor_callback(self, msg):
        """Process motor/actuator commands"""
        try:
            motor_data = np.array([
                msg.linear.x, msg.linear.y, msg.linear.z,
                msg.angular.x, msg.angular.y, msg.angular.z
            ], dtype=np.float32)
            motor_tensor = torch.tensor(motor_data).unsqueeze(0)
            
            self.motor_buffer.put({
                'data': motor_tensor,
                'timestamp': time.time()
            })
        except Exception as e:
            self.get_logger().error(f"Motor processing error: {e}")
    
    def model_deploy_callback(self, msg):
        """Deploy a new model version"""
        version = msg.data
        self.get_logger().info(f"🔄 Deploying model version {version}...")
        
        if self.model_registry.load_model(version):
            self.multimodal_compressor = self.model_registry.get_model()
            self.get_logger().info(f"✅ Model v{version} deployed successfully")
        else:
            self.get_logger().error(f"❌ Failed to deploy model v{version}")
    
    def script_load_callback(self, msg):
        """Load a Python script dynamically"""
        script_name = msg.data
        self.get_logger().info(f"📜 Loading script: {script_name}")
        
        if self.script_executor.load_script(script_name):
            self.get_logger().info(f"✅ Script {script_name} loaded")
        else:
            self.get_logger().error(f"❌ Failed to load script {script_name}")
    
    def update_bandwidth(self):
        """Monitor and update bandwidth estimate"""
        # Monitor network bandwidth (simplified)
        net_io = psutil.net_io_counters()
        cpu_load = psutil.cpu_percent() / 100.0
        
        # Adaptive bandwidth estimate based on system load
        self.bandwidth_estimate = max(0.1, 1.0 - cpu_load * 0.5)
    
    def compress_loop(self):
        """Main compression loop - runs in real-time"""
        if self.sensor_buffer.empty():
            return
        
        start_time = time.time()
        
        try:
            # Collect sensor data
            sensor_data = []
            while not self.sensor_buffer.empty() and len(sensor_data) < 4:
                sensor_data.append(self.sensor_buffer.get())
            
            if not sensor_data:
                return
            
            # Get motor data if available
            motor_data = None
            if not self.motor_buffer.empty():
                motor_item = self.motor_buffer.get()
                motor_data = motor_item['data']
            
            # Execute custom script if loaded
            if 'custom_processor' in self.script_executor.loaded_scripts:
                result = self.script_executor.execute_function(
                    'custom_processor', 'process_sensor_data', sensor_data
                )
                if result:
                    sensor_data = result
            
            # Multimodal compression
            if self.multimodal_compressor is not None:
                # Extract modalities
                image = None
                lidar = None
                imu = None
                audio = None
                
                for item in sensor_data:
                    if item['type'] == 'image':
                        image = item['data']
                    elif item['type'] == 'lidar':
                        lidar = item['data']
                    elif item['type'] == 'imu':
                        imu = item['data']
                    elif item['type'] == 'audio':
                        audio = item['data']
                
                # Use defaults if missing
                if image is None:
                    image = torch.zeros(1, 3, 224, 224)
                if lidar is None:
                    lidar = torch.zeros(1, 1024 * 3)
                if imu is None:
                    imu = torch.zeros(1, 6)
                if audio is None:
                    audio = torch.zeros(1, 128 * 128)
                
                # Compress
                with torch.no_grad():
                    (compressed, temporal_out, predicted, recon_img, mu, logvar,
                     adjusted_compression, predicted_quality) = self.multimodal_compressor(
                        image, lidar, imu, audio, self.hidden_state,
                        compression_level=self.bandwidth_estimate,
                        target_quality=0.8
                    )
                    self.hidden_state = temporal_out
                
                # Sensor-motor compression
                sensor_feat = compressed.mean(dim=-1, keepdim=True).expand(-1, 256)
                compressed_sm, sensor_decoded, motor_decoded, _, comp_level = \
                    self.sensor_motor_compressor(
                        sensor_feat, motor_data, self.bandwidth_estimate, None
                    )
                
                # Serialize and compress
                compressed_bytes = pickle.dumps(compressed_sm.cpu().numpy())
                compressed_bytes = zlib.compress(compressed_bytes)
                
                # Publish compressed data
                msg = UInt8MultiArray()
                msg.data = list(compressed_bytes)
                self.compressed_pub.publish(msg)
                
                # Calculate metrics
                input_size = sum(item['data'].numel() * 4 for item in sensor_data)
                if motor_data is not None:
                    input_size += motor_data.numel() * 4
                output_size = len(compressed_bytes)
                
                compression_ratio = input_size / max(output_size, 1)
                latency_ms = (time.time() - start_time) * 1000
                
                # Update stats
                self.compression_stats['total_in'] += input_size
                self.compression_stats['total_out'] += output_size
                self.compression_stats['compression_ratio'] = compression_ratio
                self.compression_stats['latency_ms'] = latency_ms
                
                # Publish metrics
                metrics_msg = Float32MultiArray()
                metrics_msg.data = [
                    compression_ratio,
                    latency_ms,
                    float(comp_level.item()),
                    float(predicted_quality.item()),
                    self.bandwidth_estimate
                ]
                self.metrics_pub.publish(metrics_msg)
                
                self.get_logger().info(
                    f"📊 Compression: {compression_ratio:.2f}x | "
                    f"Latency: {latency_ms:.2f}ms | "
                    f"Quality: {predicted_quality.item():.3f}"
                )
        
        except Exception as e:
            self.get_logger().error(f"Compression error: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    # Get node ID from environment or use default
    node_id = os.getenv('NODE_ID', 'node_0')
    node = EdgeCompressorNode(node_id=node_id)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

