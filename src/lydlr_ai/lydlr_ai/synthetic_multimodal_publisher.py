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

# synthetic_multimodal_publisher.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Header
import numpy as np

class SyntheticMultimodalPublisher(Node):
    def __init__(self):
        super().__init__('synthetic_multimodal_publisher')

        self.pub_img = self.create_publisher(Image, '/camera/image_raw', 10)
        self.pub_imu = self.create_publisher(Float32MultiArray, '/imu/data', 10)
        self.pub_lidar = self.create_publisher(Float32MultiArray, '/lidar/data', 10)
        self.pub_audio = self.create_publisher(Float32MultiArray, '/audio/data', 10)

        self.timer = self.create_timer(0.2, self.publish_all_sensors)  # 5 Hz

    def publish_all_sensors(self):
        now = self.get_clock().now().to_msg()

        # 1) Image: Create random 224x224 RGB image and convert to CHW bytes
        img_h, img_w = 224, 224
        img = np.random.randint(0, 256, (img_h, img_w, 3), dtype=np.uint8)  # HWC uint8 RGB

        # Convert HWC -> CHW for PyTorch style input simulation
        img_chw = np.transpose(img, (2, 0, 1))  # [3, 224, 224]

        img_msg = Image()
        img_msg.header = Header(stamp=now)
        img_msg.height = img_h
        img_msg.width = img_w
        img_msg.encoding = 'rgb8'  # 8-bit RGB
        img_msg.is_bigendian = 0
        img_msg.step = img_w * 3
        img_msg.data = img.flatten().tobytes()  # flatten HWC, raw bytes for rgb8
        self.pub_img.publish(img_msg)

        # 2) IMU: Publish 6 floats [ax, ay, az, gx, gy, gz]
        imu_data = np.random.normal(0.0, 1.0, 6).astype(np.float32)
        imu_msg = Float32MultiArray()
        imu_msg.data = imu_data.tolist()
        self.pub_imu.publish(imu_msg)

        # 3) LiDAR: Publish 3D points for 100 points (x,y,z) flattened
        num_points = 100
        # Simulate points in range [-10,10]
        lidar_points = np.random.uniform(-10.0, 10.0, (num_points, 3)).astype(np.float32)
        lidar_msg = Float32MultiArray()
        lidar_msg.data = lidar_points.flatten().tolist()
        self.pub_lidar.publish(lidar_msg)

        # 4) Audio: Publish 16000 float32 samples representing 1 second waveform at 16kHz
        audio_samples = np.random.uniform(-1.0, 1.0, 16000).astype(np.float32)
        audio_msg = Float32MultiArray()
        audio_msg.data = audio_samples.tolist()
        self.pub_audio.publish(audio_msg)

        self.get_logger().info("📤 Published synthetic data: image, imu (6d), lidar (3D pts), audio (waveform)")

def main(args=None):
    rclpy.init(args=args)
    node = SyntheticMultimodalPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()