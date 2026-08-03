import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import os

try:
    import cv2
except ImportError:
    cv2 = None


class CompressionHeatmapOverlay(Node):
    """
    Overlay compression error heatmap for a node.
    Subscribes to Lydlr preview topics (preferred) with legacy fallbacks.
    """

    def __init__(self):
        super().__init__('compression_heatmap_overlay')
        node_id = os.getenv('NODE_ID', 'node_0')
        raw_topic = os.getenv(
            'LYDLR_PREVIEW_RAW',
            f'/lydlr/{node_id}/preview/raw',
        )
        recon_topic = os.getenv(
            'LYDLR_PREVIEW_RECON',
            f'/lydlr/{node_id}/preview/reconstructed',
        )
        out_topic = os.getenv(
            'LYDLR_PREVIEW_HEATMAP',
            f'/lydlr/{node_id}/preview/heatmap',
        )

        self.orig_img = None
        self.recon_img = None

        self.create_subscription(Image, raw_topic, self.orig_cb, 10)
        self.create_subscription(Image, recon_topic, self.recon_cb, 10)
        # Legacy camera bus fallback
        self.create_subscription(Image, '/camera/image_raw', self.orig_cb, 10)
        self.create_subscription(Image, '/camera/reconstructed', self.recon_cb, 10)

        self.pub_overlay = self.create_publisher(Image, out_topic, 10)
        self.get_logger().info(
            f'Heatmap overlay: {raw_topic} + {recon_topic} → {out_topic}'
        )

    def _to_rgb(self, msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding in ('rgb8', 'bgr8'):
            img = arr.reshape(msg.height, msg.width, 3)
            if msg.encoding == 'bgr8' and cv2 is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img.copy()
        if msg.encoding == 'mono8':
            gray = arr.reshape(msg.height, msg.width)
            return np.stack([gray, gray, gray], axis=-1)
        return None

    def orig_cb(self, msg):
        img = self._to_rgb(msg)
        if img is not None:
            self.orig_img = img

    def recon_cb(self, msg):
        img = self._to_rgb(msg)
        if img is not None:
            self.recon_img = img
            if self.orig_img is not None:
                self.publish_heatmap()

    def publish_heatmap(self):
        if cv2 is None or self.orig_img is None or self.recon_img is None:
            return
        orig = self.orig_img
        recon = self.recon_img
        if orig.shape != recon.shape:
            recon = cv2.resize(recon, (orig.shape[1], orig.shape[0]))
        diff = cv2.absdiff(orig, recon)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        heatmap = cv2.applyColorMap(
            np.clip(gray_diff.astype(np.uint16) * 4, 0, 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(orig, 0.65, heatmap_rgb, 0.35, 0)

        img_msg = Image()
        img_msg.height = overlay.shape[0]
        img_msg.width = overlay.shape[1]
        img_msg.encoding = 'rgb8'
        img_msg.is_bigendian = 0
        img_msg.step = img_msg.width * 3
        img_msg.data = overlay.reshape(-1).tobytes()
        img_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_overlay.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CompressionHeatmapOverlay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
