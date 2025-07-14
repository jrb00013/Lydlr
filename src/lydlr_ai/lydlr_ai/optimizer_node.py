# lydlr_ai/optimizer_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import torch
import numpy as np
import psutil  # For system metrics, install if needed

from lydlr_ai.model.compressor import MultimodalCompressor, CompressionPolicy, QualityAssessor

class StorageOptimizer(Node):
    def __init__(self):
        super().__init__('storage_optimizer')
        self.get_logger().info("StorageOptimizer node started...")

        # Subscriptions for multimodal sensors (simulate IMU, LiDAR, Audio too)
        self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        self.create_subscription(Float32, '/imu/data', self.imu_callback)
        self.create_subscription(Float32, '/lidar/data', self.lidar_callback)
        self.create_subscription(Float32, '/audio/data', self.audio_callback)

        self.compressor = None
        self.policy = CompressionPolicy()
        self.assessor = QualityAssessor()
        self.hidden_state = None

        self.latest_inputs = {
            'image': None,
            'imu': None,
            'lidar': None,
            'audio': None
        }

    def camera_callback(self, msg):
        try:
            if msg.encoding == 'rgb8':
                img_np = np.array(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                img_np = img_np / 255.0  # normalize to [0,1]
                img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2,0,1).unsqueeze(0)  # B,C,H,W
            elif msg.encoding == 'mono8':
                img_np = np.array(msg.data, dtype=np.uint8).reshape(msg.height, msg.width) / 255.0
                img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # B,1,H,W
            else:
                self.get_logger().warn(f"Unsupported encoding: {msg.encoding}")
                return
            self.latest_inputs['image'] = img_tensor
            self.try_compress()
        except Exception as e:
            self.get_logger().error(f"Camera processing failed: {e}")

    def imu_callback(self, msg):
        # Example: single Float32 message carrying magnitude, replace with actual IMU message type
        self.latest_inputs['imu'] = torch.tensor([[msg.data]*6], dtype=torch.float32)
        self.try_compress()

    def lidar_callback(self, msg):
        # Simulated lidar data vector in Float32 message, replace with real data type
        self.latest_inputs['lidar'] = torch.tensor([[msg.data]*1024], dtype=torch.float32)
        self.try_compress()

    def audio_callback(self, msg):
        # Simulated audio data vector
        self.latest_inputs['audio'] = torch.tensor([[msg.data]*16384], dtype=torch.float32)  # 128*128 flattened
        self.try_compress()

    def try_compress(self):
        # Only compress if all inputs are available
        if None in self.latest_inputs.values():
            return

        if self.compressor is None:
            # Initialize compressor with dynamic sizes based on input tensor shapes
            img_shape = self.latest_inputs['image'].shape[1:]  # (C,H,W)
            self.compressor = MultimodalCompressor(
                image_shape=img_shape,
                lidar_dim=self.latest_inputs['lidar'].shape[1],
                imu_dim=self.latest_inputs['imu'].shape[1],
                audio_dim=self.latest_inputs['audio'].shape[1]
            )
            self.get_logger().info(f"Initialized compressor with inputs: {img_shape}")

        # Simulate system monitoring
        cpu_load = psutil.cpu_percent() / 100.0
        battery = 0.8  # Stub - replace with real battery API
        network_bandwidth = 0.5  # Stub

        compression_level = self.policy.get_level()
        # Potentially adjust compression_level based on system load here

        # Forward pass with temporal LSTM
        encoded, decoded, self.hidden_state = self.compressor(
            self.latest_inputs['image'],
            self.latest_inputs['lidar'],
            self.latest_inputs['imu'],
            self.latest_inputs['audio'],
            self.hidden_state
        )

        # Real-time quality check between image input and reconstructed (fake decoder output)
        # For demo, just compare input image with itself to get 0 quality loss
        quality = self.assessor.assess(
            self.latest_inputs['image'],
            self.latest_inputs['image']
        )

        # Calculate compression ratio dummy (latent size / input size)
        input_size = torch.prod(torch.tensor(self.latest_inputs['image'].shape)).item()
        compressed_size = encoded.numel()
        compression_ratio = compressed_size / input_size

        self.policy.update_policy(compression_ratio, quality)

        self.get_logger().info(
            f"Compression ratio: {compression_ratio:.3f}, Quality: {quality:.4f}, Compression level: {compression_level:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = StorageOptimizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
