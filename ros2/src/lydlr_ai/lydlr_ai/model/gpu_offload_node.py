# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#gpu_offload_node.py
import rclpy
from rclpy.node import Node
import torch
import time

class GPUOffloadNode(Node):
    def __init__(self):
        super().__init__('gpu_offload_node')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")

        self.timer = self.create_timer(1.0, self.inference_loop)

    def inference_loop(self):
        start = time.time()
        x = torch.randn(32, 3, 224, 224).to(self.device)
        y = torch.relu(x)
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end = time.time()

        latency_ms = (end - start) * 1000
        self.get_logger().info(f"Inference latency on {self.device}: {latency_ms:.2f} ms")

def main(args=None):
    rclpy.init(args=args)
    node = GPUOffloadNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
