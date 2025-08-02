import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from lydlr_ai.utils.lyd_format import save_lyd_frame  # implement your saving util
from cv_bridge import CvBridge
import numpy as np

class EventTriggeredSaver(Node):
    def __init__(self):
        super().__init__('event_triggered_saver')

        self.declare_parameter('motion_threshold', 0.05)
        self.motion_threshold = self.get_parameter('motion_threshold').value

        self.bridge = CvBridge()

        self.prev_image = None
        self.prev_imu = None

        self.subscription_img = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.subscription_imu = self.create_subscription(Imu, '/imu/data_raw', self.imu_callback, 10)

    def image_callback(self, msg):
        curr_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8') / 255.0
        if self.prev_image is not None:
            diff = np.mean(np.abs(curr_img - self.prev_image))
            if diff > self.motion_threshold:
                self.save_frame(curr_img, None)
        self.prev_image = curr_img

    def imu_callback(self, msg):
        linear_acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        accel_magnitude = np.linalg.norm(linear_acc)
        if accel_magnitude > 1.5:  # threshold for spikes/falls
            self.save_frame(None, linear_acc)

    def save_frame(self, image, imu_data):
        # Compose minimal data dict, extend as needed
        data = {}
        if image is not None:
            data['image'] = (image * 255).astype(np.uint8)
        if imu_data is not None:
            data['imu'] = imu_data

        filename = f'data/event_{int(self.get_clock().now().to_msg().sec)}.lyd'
        save_lyd_frame(filename, data)
        self.get_logger().info(f"Saved event-triggered frame to {filename}")

def main(args=None):
    rclpy.init(args=args)
    node = EventTriggeredSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
