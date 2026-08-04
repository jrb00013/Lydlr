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
from collections import deque

try:
    import psutil
except ImportError:
    psutil = None

from lydlr_ai.model.compressor import EnhancedMultimodalCompressor, unpack_compressor_output
from lydlr_ai.model.true_rate import rate_report
from lydlr_ai.utils.metrics_reporter import report_metrics
from lydlr_ai.utils.preview_reporter import report_preview
from lydlr_ai.communication.edge_transport import EdgeTransportLayer, sensor_qos
from lydlr_ai.communication.topics import LydlrTopics
from lydlr_ai.communication import wire

try:
    import cv2
except ImportError:
    cv2 = None
from lydlr_ai.communication.link_policy import (
    NodeLinkPolicy,
    vision_frame_skip,
    prioritize_modalities,
    should_transmit_modality,
    estimate_output_kbps,
)
from lydlr_ai.communication.modality_codec import (
    encode_imu_delta,
    frame_multimodal_payload,
    downsample_lidar,
)
try:
    from lydlr_ai.model.rl_policy import RLCompressionController
except ImportError:
    RLCompressionController = None
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
            # Try new naming convention first, then fall back to old
            model_path = self.model_dir / f"lydlr_compressor_v{version}.pth"
            metadata_path = self.model_dir / f"metadata_lydlr_compressor_v{version}.json"
            if not model_path.exists():
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
                missing, unexpected = model.load_state_dict(
                    checkpoint['model_state_dict'], strict=False
                )
                if missing or unexpected:
                    print(
                        f"Loaded v{version} with strict=False "
                        f"(missing={len(missing)}, unexpected={len(unexpected)})"
                    )
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
        for pattern in ("lydlr_compressor_v*.pth", "compressor_v*.pth"):
            for f in self.model_dir.glob(pattern):
                if "_v" in f.stem:
                    versions.append(f.stem.split("_v", 1)[1])
        return sorted(set(versions), reverse=True)


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

        # Quality guard state — LPIPS-driven level adjustment
        self._quality_history = deque(maxlen=20)
        self._consecutive_low_quality = 0
        self._consecutive_good_quality = 0
        self._quality_guard_suppression = 0.0
        self._min_quality_threshold = 0.7

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
        self._transport_seq = 0
        self._allocated_mbps = float(os.getenv("UPLINK_MBPS", "0"))
        self._uplink_budget_kbps = float(os.getenv("UPLINK_BUDGET_KBPS", "0"))
        self._image_tick = 0
        vertical = os.getenv("NODE_VERTICAL", os.getenv("LYDLR_VERTICAL", "drone"))
        self._link_policy = NodeLinkPolicy.from_dict(
            node_id,
            {
                "vertical": vertical,
                "uplink_budget_kbps": self._uplink_budget_kbps or None,
            },
        )
        ingest_hz = float(os.getenv("LYDLR_INGEST_HZ", "10" if vertical == "drone" else "2"))
        self._vision_skip = vision_frame_skip(self._link_policy, ingest_hz)
        self._modality_weights = prioritize_modalities(self._link_policy)
        self._rl_mode = os.getenv("RL_MODE", "heuristic")
        self._rl_controller: Optional[RLCompressionController] = None
        if self._rl_mode != "heuristic" and RLCompressionController is not None:
            model_path = os.getenv("RL_MODEL_PATH", "")
            self._rl_controller = RLCompressionController(
                mode=self._rl_mode,
                model_path=Path(model_path) if model_path else None,
            )
        self._last_imu: Optional[np.ndarray] = None
        self._interval_bytes_out = 0
        self._interval_start = time.time()
        self._active_modalities = {"camera", "lidar", "imu", "audio"}
        self._min_quality_threshold = self._link_policy.min_quality

        # Lydlr transport layer (LYDT wire + legacy topics)
        self.transport = EdgeTransportLayer(self, node_id)
        self.transport.subscribe_deploy(self.model_deploy_callback)
        self.transport.subscribe_script(self.script_load_callback)
        self.transport.subscribe_coordination(self._coordination_callback)

        sqos = sensor_qos()
        self.create_subscription(Image, LydlrTopics.CAMERA, self.image_callback, sqos)
        self.create_subscription(Float32MultiArray, LydlrTopics.LIDAR, self.lidar_callback, sqos)
        self.create_subscription(Float32MultiArray, LydlrTopics.IMU, self.imu_callback, sqos)
        self.create_subscription(Float32MultiArray, LydlrTopics.AUDIO, self.audio_callback, sqos)
        self.create_subscription(Twist, LydlrTopics.CMD_VEL, self.motor_callback, sqos)

        self.decompressed_pub = self.create_publisher(
            Float32MultiArray, f'/{node_id}/decompressed', 10
        )
        self.preview_raw_pub = self.create_publisher(
            Image, LydlrTopics.preview_raw(node_id), 2
        )
        self.preview_recon_pub = self.create_publisher(
            Image, LydlrTopics.preview_reconstructed(node_id), 2
        )
        self.preview_heatmap_pub = self.create_publisher(
            Image, LydlrTopics.preview_heatmap(node_id), 2
        )
        self._preview_tick = 0
        self._preview_every_n = max(1, int(os.getenv("LYDLR_PREVIEW_EVERY_N", "5")))
        self._preview_max_dim = int(os.getenv("LYDLR_PREVIEW_MAX_DIM", "320"))

        self.compression_timer = self.create_timer(0.1, self.compress_loop)
        self.bandwidth_timer = self.create_timer(1.0, self.update_bandwidth)
        self.heartbeat_timer = self.create_timer(2.0, self._publish_heartbeat)
        
        # Load latest model
        versions = self.model_registry.list_versions()
        if versions:
            self.model_registry.load_model(versions[0])
            self.multimodal_compressor = self.model_registry.get_model()
        
        self.get_logger().info(f"🚀 Edge Compressor {node_id} [{self.transport.vertical}]")
        self.get_logger().info(f"   Transport: {LydlrTopics.compressed_transport(node_id)}")
        self.get_logger().info(f"   Models: {versions}")
    
    def _coordination_callback(self, payload: wire.CoordinationPayload):
        """Apply fleet coordinator bandwidth / compression targets."""
        self.bandwidth_estimate = max(0.1, min(0.98, payload.target_compression))
        if payload.allocated_mbps > 0:
            self._allocated_mbps = payload.allocated_mbps
            self._uplink_budget_kbps = payload.allocated_mbps * 1000.0
            self._link_policy.allocated_mbps = payload.allocated_mbps
            self._link_policy.uplink_budget_kbps = self._uplink_budget_kbps

    def _budget_ratio(self) -> float:
        elapsed = max(time.time() - self._interval_start, 0.05)
        est_kbps = estimate_output_kbps(self._interval_bytes_out, elapsed)
        budget = max(self._link_policy.uplink_budget_kbps, 8.0)
        return est_kbps / budget

    def _refresh_modality_gates(self):
        ratio = self._budget_ratio()
        self._active_modalities = {
            mod
            for mod in ("camera", "lidar", "imu", "audio")
            if should_transmit_modality(mod, self._modality_weights, budget_ratio=ratio)
        }

    def _publish_heartbeat(self):
        version = self.model_registry.model_version or ""
        self.transport.publish_heartbeat(version)

    def _tensor_to_rgb_u8(self, tensor) -> Optional[np.ndarray]:
        """Convert BCHW / CHW float tensor (0-1 or -1..1) to HxWx3 uint8 RGB."""
        if tensor is None or cv2 is None:
            return None
        try:
            arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            if arr.dtype != np.uint8:
                if arr.min() < -0.01:
                    arr = (arr + 1.0) * 0.5
                arr = np.clip(arr, 0.0, 1.0)
                arr = (arr * 255.0).astype(np.uint8)
            h, w = arr.shape[:2]
            max_dim = self._preview_max_dim
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                arr = cv2.resize(
                    arr,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            return arr
        except Exception as exc:
            self.get_logger().debug(f"preview tensor convert failed: {exc}")
            return None

    def _rgb_to_image_msg(self, rgb: np.ndarray) -> Image:
        msg = Image()
        msg.height = int(rgb.shape[0])
        msg.width = int(rgb.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = rgb.reshape(-1).tobytes()
        return msg

    def _rgb_to_jpeg(self, rgb: np.ndarray) -> Optional[bytes]:
        if cv2 is None or rgb is None:
            return None
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            return None
        return buf.tobytes()

    def _publish_previews(self, raw_tensor, recon_tensor):
        """Downscaled Image topics + HTTP JPEG ingest for the control plane."""
        self._preview_tick += 1
        if (self._preview_tick % self._preview_every_n) != 0:
            return
        raw_rgb = self._tensor_to_rgb_u8(raw_tensor)
        recon_rgb = self._tensor_to_rgb_u8(recon_tensor)
        if raw_rgb is None and recon_rgb is None:
            return
        if raw_rgb is not None:
            self.preview_raw_pub.publish(self._rgb_to_image_msg(raw_rgb))
            jpeg = self._rgb_to_jpeg(raw_rgb)
            if jpeg:
                report_preview(self.node_id, "raw", jpeg)
        if recon_rgb is not None:
            self.preview_recon_pub.publish(self._rgb_to_image_msg(recon_rgb))
            jpeg = self._rgb_to_jpeg(recon_rgb)
            if jpeg:
                report_preview(self.node_id, "reconstructed", jpeg)
        if raw_rgb is not None and recon_rgb is not None and cv2 is not None:
            if raw_rgb.shape != recon_rgb.shape:
                recon_rgb = cv2.resize(recon_rgb, (raw_rgb.shape[1], raw_rgb.shape[0]))
            diff = cv2.absdiff(raw_rgb, recon_rgb)
            gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            heat = cv2.applyColorMap(np.clip(gray.astype(np.uint16) * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(raw_rgb, 0.55, heat_rgb, 0.45, 0)
            self.preview_heatmap_pub.publish(self._rgb_to_image_msg(overlay))
            jpeg = self._rgb_to_jpeg(overlay)
            if jpeg:
                report_preview(self.node_id, "heatmap", jpeg)

    def image_callback(self, msg):
        """Process camera image — may skip frames on IoT/LPWAN budgets."""
        self._image_tick += 1
        if self._vision_skip > 1 and (self._image_tick % self._vision_skip) != 0:
            return
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
        if "lidar" not in self._active_modalities:
            return
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
        if "imu" not in self._active_modalities:
            return
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
        if "audio" not in self._active_modalities:
            return
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
        """Monitor and update bandwidth estimate + modality gates."""
        if psutil is None:
            return
        cpu_load = psutil.cpu_percent() / 100.0
        self.bandwidth_estimate = max(0.1, 1.0 - cpu_load * 0.5)
        self._refresh_modality_gates()
        self._interval_bytes_out = 0
        self._interval_start = time.time()
    
    def compress_loop(self):
        """Main compression loop - runs in real-time"""
        if self.sensor_buffer.empty():
            return
        
        start_time = time.time()
        
        try:
            self._refresh_modality_gates()

            # Collect sensor data
            sensor_data = []
            while not self.sensor_buffer.empty() and len(sensor_data) < 4:
                item = self.sensor_buffer.get()
                mod_type = item.get("type")
                mod_key = "camera" if mod_type == "image" else mod_type
                if mod_key in self._active_modalities:
                    sensor_data.append(item)
            
            if not sensor_data:
                return
            
            modality_bytes_in = {}
            modality_bytes_out = {}
            modality_quality = {}
            framed_chunks = {}
            comp_level = float(self.bandwidth_estimate)

            if self._rl_controller is not None:
                ratio = self._budget_ratio()
                quality_trend = (
                    np.mean(list(self._quality_history)[-5:])
                    if len(self._quality_history) >= 5
                    else self._quality_history[-1] if self._quality_history else 0.85
                )
                adj = self._rl_controller.predict(
                    budget_ratio=ratio,
                    quality_score=self._quality_history[-1] if self._quality_history else 0.85,
                    latency_ms=self.compression_stats["latency_ms"],
                    quality_trend=quality_trend,
                    cpu_load=getattr(self, "_cpu_load", 0.5),
                )
                comp_level = max(0.1, min(0.98, comp_level + adj))
            
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
                    mod = item["type"]
                    raw_bytes = int(item["data"].numel() * 4)
                    mod_key = "camera" if mod == "image" else mod
                    modality_bytes_in[mod_key] = modality_bytes_in.get(mod_key, 0) + raw_bytes

                    if item['type'] == 'image':
                        image = item['data']
                    elif item['type'] == 'lidar':
                        lidar_np = downsample_lidar(
                            item['data'].cpu().numpy().reshape(-1),
                            comp_level,
                        )
                        lidar = torch.tensor(lidar_np, dtype=torch.float32).unsqueeze(0)
                        framed_chunks["lidar"] = lidar_np.astype(np.float32).tobytes()
                    elif item['type'] == 'imu':
                        imu_np = item['data'].cpu().numpy().reshape(-1)
                        delta_bytes, self._last_imu = encode_imu_delta(self._last_imu, imu_np)
                        imu = torch.tensor(imu_np, dtype=torch.float32).unsqueeze(0)
                        framed_chunks["imu"] = delta_bytes
                    elif item['type'] == 'audio':
                        audio = item['data']
                        framed_chunks["audio"] = item['data'].cpu().numpy().astype(np.float32).tobytes()
                
                # Use defaults if missing
                if image is None:
                    image = torch.zeros(1, 3, 224, 224)
                if lidar is None:
                    lidar = torch.zeros(1, 1024 * 3)
                if imu is None:
                    imu = torch.zeros(1, 6)
                if audio is None:
                    audio = torch.zeros(1, 128 * 128)
                
                # Compress (supports legacy 8-tuple and RD 11-tuple outputs)
                with torch.no_grad():
                    cpu_load = 0.0
                    if psutil is not None:
                        try:
                            cpu_load = float(psutil.cpu_percent(interval=None)) / 100.0
                        except Exception:
                            cpu_load = 0.0
                    edge_fast = cpu_load > 0.75 or float(self.bandwidth_estimate) < 0.35

                    packed = unpack_compressor_output(
                        self.multimodal_compressor(
                            image,
                            lidar,
                            imu,
                            audio,
                            self.hidden_state,
                            compression_level=self.bandwidth_estimate,
                            target_quality=0.8,
                            edge_fast=edge_fast,
                        )
                    )
                    compressed = packed["compressed"]
                    temporal_out = packed["temporal_out"]
                    recon_img = packed["recon_img"]
                    predicted_quality = packed["predicted_quality"]
                    rate_bits = packed["rate_bits"]
                    quant_indices = packed.get("quant_indices")
                    tr_stats, packed_idx = rate_report(rate_bits, quant_indices, num_levels=256)
                    self.hidden_state = temporal_out

                    quality_val = float(predicted_quality.item())
                    self._quality_history.append(quality_val)
                    if quality_val < self._min_quality_threshold:
                        self._consecutive_low_quality += 1
                        self._consecutive_good_quality = 0
                    else:
                        self._consecutive_good_quality += 1
                        self._consecutive_low_quality = 0

                    if self._consecutive_low_quality >= 3:
                        self._quality_guard_suppression = min(
                            0.6, self._quality_guard_suppression + 0.15
                        )
                    elif self._consecutive_good_quality >= 5 and self._quality_guard_suppression > 0:
                        self._quality_guard_suppression = max(
                            0.0, self._quality_guard_suppression - 0.1
                        )

                    effective_level = max(
                        0.1,
                        self.bandwidth_estimate - self._quality_guard_suppression,
                    )
                    if effective_level != self.bandwidth_estimate:
                        self.bandwidth_estimate = effective_level

                # Sensor-motor compression
                sensor_feat = compressed.mean(dim=-1, keepdim=True).expand(-1, 256)
                compressed_sm, sensor_decoded, motor_decoded, _, comp_level = \
                    self.sensor_motor_compressor(
                        sensor_feat, motor_data, self.bandwidth_estimate, None
                    )
                
                raw_blob = pickle.dumps(compressed_sm.cpu().numpy())
                if framed_chunks:
                    if image is not None:
                        framed_chunks["camera"] = zlib.compress(
                            image.cpu().numpy().astype(np.float32).tobytes(), level=6
                        )
                    framed_chunks["compressed"] = raw_blob
                    if packed_idx:
                        framed_chunks["quant_indices"] = packed_idx
                    raw_blob = frame_multimodal_payload(framed_chunks)

                input_size = sum(modality_bytes_in.values())
                if motor_data is not None:
                    input_size += motor_data.numel() * 4
                payload_bytes = raw_blob if isinstance(raw_blob, (bytes, bytearray)) else pickle.dumps(raw_blob)
                output_size = len(zlib.compress(payload_bytes, level=6))

                q_score = float(predicted_quality.item())
                for mod_key, b_in in modality_bytes_in.items():
                    share = b_in / max(input_size, 1)
                    modality_bytes_out[mod_key] = int(output_size * share)
                    modality_quality[mod_key] = q_score

                self._interval_bytes_out += output_size

                compression_ratio = input_size / max(output_size, 1)
                latency_ms = (time.time() - start_time) * 1000
                model_ver = self.model_registry.model_version or ""

                self.compression_stats['total_in'] += input_size
                self.compression_stats['total_out'] += output_size
                self.compression_stats['compression_ratio'] = compression_ratio
                self.compression_stats['latency_ms'] = latency_ms

                self.transport.publish_compressed(
                    payload_bytes,
                    model_ver,
                    input_size,
                    compression_ratio,
                )

                try:
                    self._publish_previews(image, recon_img)
                except Exception as preview_exc:
                    self.get_logger().debug(f"preview publish skipped: {preview_exc}")

                rl_mode = self._rl_mode
                rl_action = self._rl_controller.action if self._rl_controller else 0.0
                rl_reward = self._rl_controller.reward if self._rl_controller else 0.0

                metrics = wire.MetricsPayload(
                    node_id=self.node_id,
                    vertical=self.transport.vertical,
                    model_version=model_ver,
                    compression_ratio=compression_ratio,
                    latency_ms=latency_ms,
                    compression_level=float(comp_level.item()),
                    quality_score=q_score,
                    bandwidth_estimate=self._uplink_budget_kbps or self.bandwidth_estimate * 512,
                    bytes_in=input_size,
                    bytes_out=output_size,
                    modality_bytes_in=modality_bytes_in,
                    modality_bytes_out=modality_bytes_out,
                    modality_quality=modality_quality,
                    controller_mode=rl_mode,
                    rl_action=rl_action,
                    rl_reward=rl_reward,
                )
                self.transport.publish_metrics(metrics)

                report_metrics(
                    node_id=self.node_id,
                    compression_ratio=compression_ratio,
                    latency_ms=latency_ms,
                    quality_score=q_score,
                    bandwidth_estimate=self.bandwidth_estimate,
                    compression_level=float(comp_level.item()),
                    vertical=self.transport.vertical,
                    bytes_in=input_size,
                    bytes_out=output_size,
                    modality_bytes_in=modality_bytes_in,
                    modality_bytes_out=modality_bytes_out,
                    modality_quality=modality_quality,
                    controller_mode=rl_mode,
                    rl_action=rl_action,
                    rl_reward=rl_reward,
                )
                
                self.get_logger().info(
                    f"📊 Compression: {compression_ratio:.2f}x | "
                    f"Latency: {latency_ms:.2f}ms | "
                    f"Quality: {predicted_quality.item():.3f} | "
                    f"Rproxy={tr_stats['proxy_rate_bits']:.2f} "
                    f"Rtrue={tr_stats['true_rate_bits']:.1f} bits"
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

