# synthetic_multimodal_publisher.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import numpy as np
from std_msgs.msg import Header

class SyntheticMultimodalPublisher(Node):
    def __init__(self):
        super().__init__('synthetic_multimodal_publisher')

        self.pub_img = self.create_publisher(Image, '/camera/image_raw', 10)
        self.pub_imu = self.create_publisher(Float32, '/imu/data', 10)
        self.pub_lidar = self.create_publisher(Float32, '/lidar/data', 10)
        self.pub_audio = self.create_publisher(Float32, '/audio/data', 10)

        self.timer = self.create_timer(1.0, self.publish_all_sensors)

    def publish_all_sensors(self):
        # Image: RGB random noise
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_msg = Image()
        img_msg.header = Header()
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.height = 480
        img_msg.width = 640
        img_msg.encoding = 'rgb8'
        img_msg.is_bigendian = 0
        img_msg.step = 640 * 3
        img_msg.data = img.flatten().tolist()
        self.pub_img.publish(img_msg)

        # IMU: Random vector (6 DOF simulated)
        imu_msg = Float32()
        imu_msg.data = float(np.random.rand())
        self.pub_imu.publish(imu_msg)

        # LIDAR: Random float (simulate single value for simplicity)
        lidar_msg = Float32()
        lidar_msg.data = float(np.random.rand())
        self.pub_lidar.publish(lidar_msg)

        # Audio: Random float (simulate single value for simplicity)
        audio_msg = Float32()
        audio_msg.data = float(np.random.rand())
        self.pub_audio.publish(audio_msg)

        self.get_logger().info("Published synthetic multimodal sensor data.")

def main(args=None):
    rclpy.init(args=args)
    node = SyntheticMultimodalPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()