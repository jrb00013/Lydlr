import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import torch

# AI Model logic (you'll create this file)
from lydlr_ai.model.compressor import AICompressor

class StorageOptimizer(Node):
    def __init__(self):
        super().__init__('storage_optimizer')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10)
        self.compressor = AICompressor()

    def listener_callback(self, msg):
        # Example tensor creation from flat byte stream
        img_tensor = torch.tensor(list(msg.data), dtype=torch.uint8)
        compressed = self.compressor.compress(img_tensor)
        self.get_logger().info(f"Compressed image to: {len(compressed)} bytes")

def main(args=None):
    rclpy.init(args=args)
    node = StorageOptimizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
