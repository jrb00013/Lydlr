import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

class CompressionHeatmapOverlay(Node):
    def __init__(self):
        super().__init__('compression_heatmap_overlay')

        self.bridge = CvBridge()

        self.sub_orig = self.create_subscription(Image, '/camera/image_raw', self.orig_cb, 10)
        self.sub_recon = self.create_subscription(Image, '/camera/reconstructed', self.recon_cb, 10)

        self.orig_img = None
        self.recon_img = None

        self.pub_overlay = self.create_publisher(Image, '/camera/compression_heatmap', 10)

    def orig_cb(self, msg):
        self.orig_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def recon_cb(self, msg):
        self.recon_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        if self.orig_img is not None:
            self.publish_heatmap()

    def publish_heatmap(self):
        diff = cv2.absdiff(self.orig_img, self.recon_img)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        heatmap = cv2.applyColorMap((gray_diff*4).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(self.orig_img, 0.7, heatmap, 0.3, 0)

        img_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='rgb8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_overlay.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CompressionHeatmapOverlay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
