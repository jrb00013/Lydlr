# synthetic_multimodal_publisher.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
import numpy as np
from std_msgs.msg import Header
import array

class SyntheticMultimodalPublisher(Node):
    def __init__(self):
        super().__init__('synthetic_multimodal_publisher')

        self.pub_img = self.create_publisher(Image, '/camera/image_raw', 10)
        self.pub_imu = self.create_publisher(Float32, '/imu/data', 10)
        self.pub_lidar = self.create_publisher(Float32, '/lidar/data', 10)
        self.pub_audio = self.create_publisher(Float32MultiArray, '/audio/data', 10)

        # Publish every 0.2 seconds (~5 Hz) to mimic real streaming
        self.timer = self.create_timer(0.2, self.publish_all_sensors)

    def publish_all_sensors(self):
        now = self.get_clock().now().to_msg()

        # 🖼 Image (480x640 RGB)
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_msg = Image()
        img_msg.header = Header(stamp=now)
        img_msg.height = 480
        img_msg.width = 640
        img_msg.encoding = 'rgb8'
        img_msg.is_bigendian = 0
        img_msg.step = 640 * 3
        img_msg.data = img.flatten().tolist()
        self.pub_img.publish(img_msg)

        # 📡 IMU: Simulated single float representing [ax,ay,az,gx,gy,gz]
        imu_value = float(np.random.normal(loc=0.0, scale=1.0))
        imu_msg = Float32()
        imu_msg.data = imu_value
        self.pub_imu.publish(imu_msg)

        # 🛞 LiDAR: Simulated point cloud magnitude
        lidar_value = float(np.random.uniform(0, 10))
        lidar_msg = Float32()
        lidar_msg.data = lidar_value
        self.pub_lidar.publish(lidar_msg)

        # 🎤 Audio: Simulated waveform chunk (send RMS value)
        audio_rms = float(np.random.normal(0.0, 0.2))
        
        # Generate 16000 samples of random audio (1 second at 16kHz)
        audio_msg = Float32MultiArray()
        samples = np.random.rand(16000).astype(np.float32)
        audio_msg.data = samples.tolist()
        self.pub_audio.publish(audio_msg)

        self.get_logger().info("📤 Published synthetic data (img, imu, lidar, audio)")

def main(args=None):
    rclpy.init(args=args)
    node = SyntheticMultimodalPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
