#async_stream_splitter.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray

class AsyncStreamSplitter(Node):
    def __init__(self):
        super().__init__('async_stream_splitter')

        self.subscription = self.create_subscription(
            ByteMultiArray,
            '/compressed_stream',
            self.stream_callback,
            10)

        self.pub_img = self.create_publisher(ByteMultiArray, '/compressed/img', 10)
        self.pub_lidar = self.create_publisher(ByteMultiArray, '/compressed/lidar', 10)
        self.pub_imu = self.create_publisher(ByteMultiArray, '/compressed/imu', 10)
        self.pub_audio = self.create_publisher(ByteMultiArray, '/compressed/audio', 10)

    def stream_callback(self, msg):
        data = msg.data
        # Placeholder for splitting logic, here just forwarding raw data for demo:
        # Split into parts depending on protocol - here assumed fixed sizes (replace with actual parsing)
        img_data = data[0:1000]
        lidar_data = data[1000:2000]
        imu_data = data[2000:2100]
        audio_data = data[2100:]

        self.pub_img.publish(ByteMultiArray(data=img_data))
        self.pub_lidar.publish(ByteMultiArray(data=lidar_data))
        self.pub_imu.publish(ByteMultiArray(data=imu_data))
        self.pub_audio.publish(ByteMultiArray(data=audio_data))

def main(args=None):
    rclpy.init(args=args)
    node = AsyncStreamSplitter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
