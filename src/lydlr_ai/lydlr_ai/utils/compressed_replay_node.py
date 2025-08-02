import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, Imu, AudioData
from std_msgs.msg import Header
import time
import threading

from lydlr_ai.utils.lyd_format import load_lyd_frame  # Your .lyd loader util
from lydlr_ai.utils.voxel_utils import lidar_to_pointcloud
from cv_bridge import CvBridge
import numpy as np
import cv2

class CompressedReplayNode(Node):
    def __init__(self):
        super().__init__('compressed_replay_node')

        self.bridge = CvBridge()
        self.declare_parameter('lyd_file', 'data/sample.lyd')
        self.declare_parameter('replay_rate', 10.0)  # Hz

        self.lyd_file = self.get_parameter('lyd_file').get_parameter_value().string_value
        self.rate = self.get_parameter('replay_rate').get_parameter_value().double_value

        self.pub_image = self.create_publisher(Image, '/camera/reconstructed', 10)
        self.pub_lidar = self.create_publisher(PointCloud2, '/lidar/reconstructed', 10)
        self.pub_imu = self.create_publisher(Imu, '/imu/reconstructed', 10)
        self.pub_audio = self.create_publisher(AudioData, '/audio/reconstructed', 10)

        self.timer = self.create_timer(1.0 / self.rate, self.timer_callback)

        self.frames = self.load_all_frames()
        self.frame_idx = 0

        self.get_logger().info(f"Loaded {len(self.frames)} frames from {self.lyd_file}")

    def load_all_frames(self):
        frames = []
        # Assume load_lyd_frame yields dict with keys: image, lidar, imu, audio
        for frame in load_lyd_frame(self.lyd_file):
            frames.append(frame)
        return frames

    def timer_callback(self):
        if self.frame_idx >= len(self.frames):
            self.get_logger().info("Replay finished. Resetting.")
            self.frame_idx = 0
            return

        frame = self.frames[self.frame_idx]
        self.frame_idx += 1

        now = self.get_clock().now().to_msg()

        # Publish reconstructed image
        img = frame['image']  # numpy uint8 HxWx3
        img_msg = self.bridge.cv2_to_imgmsg(img, encoding='rgb8')
        img_msg.header.stamp = now
        img_msg.header.frame_id = 'camera'
        self.pub_image.publish(img_msg)

        # Publish reconstructed lidar (PointCloud2)
        lidar_pc2 = self.lidar_to_pointcloud2(frame['lidar'], now)
        self.pub_lidar.publish(lidar_pc2)

        # Publish imu
        imu_data = frame['imu']  # dict or msg compatible
        imu_msg = Imu()
        imu_msg.header.stamp = now
        imu_msg.header.frame_id = 'imu_link'
        imu_msg.orientation_covariance[0] = -1  # orientation not provided
        imu_msg.linear_acceleration.x = imu_data['accel_x']
        imu_msg.linear_acceleration.y = imu_data['accel_y']
        imu_msg.linear_acceleration.z = imu_data['accel_z']
        imu_msg.angular_velocity.x = imu_data['gyro_x']
        imu_msg.angular_velocity.y = imu_data['gyro_y']
        imu_msg.angular_velocity.z = imu_data['gyro_z']
        self.pub_imu.publish(imu_msg)

        # Publish audio (raw bytes)
        audio_bytes = frame['audio'].tobytes()
        audio_msg = AudioData()
        audio_msg.data = audio_bytes
        self.pub_audio.publish(audio_msg)

    def lidar_to_pointcloud2(self, lidar_tensor, timestamp):
        # Convert lidar tensor to PointCloud2 message
        # Use helper or create minimal PC2 msg here
        import sensor_msgs_py.point_cloud2 as pc2
        from sensor_msgs.msg import PointCloud2, PointField

        xyz = lidar_to_pointcloud(lidar_tensor.unsqueeze(0))[0].numpy()

        header = Header()
        header.stamp = timestamp
        header.frame_id = 'lidar'

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        pc2_msg = pc2.create_cloud(header, fields, xyz)
        return pc2_msg

def main(args=None):
    rclpy.init(args=args)
    node = CompressedReplayNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
