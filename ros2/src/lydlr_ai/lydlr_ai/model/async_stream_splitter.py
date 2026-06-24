#async_stream_splitter.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray

from lydlr_ai.communication.modality_codec import MODALITY_ORDER, frame_multimodal_payload, split_multimodal_payload


class AsyncStreamSplitter(Node):
    """Demux LYMS multimodal compressed stream into per-modality topics."""

    TOPIC_MAP = {
        "camera": "/compressed/img",
        "lidar": "/compressed/lidar",
        "imu": "/compressed/imu",
        "audio": "/compressed/audio",
    }

    def __init__(self):
        super().__init__("async_stream_splitter")

        self.subscription = self.create_subscription(
            ByteMultiArray,
            "/compressed_stream",
            self.stream_callback,
            10,
        )

        self.publishers = {
            mod: self.create_publisher(ByteMultiArray, topic, 10)
            for mod, topic in self.TOPIC_MAP.items()
        }
        self.merger_subs = {}
        self._merge_buffer = {}
        for mod, topic in self.TOPIC_MAP.items():
            self.merger_subs[mod] = self.create_subscription(
                ByteMultiArray,
                topic,
                lambda msg, m=mod: self._merge_part(m, msg),
                10,
            )
        self.merge_pub = self.create_publisher(ByteMultiArray, "/compressed_stream/merged", 10)
        self.get_logger().info("AsyncStreamSplitter ready (LYMS split + merge)")

    def stream_callback(self, msg):
        try:
            chunks = split_multimodal_payload(bytes(msg.data))
        except ValueError as exc:
            self.get_logger().warn(f"Invalid LYMS frame: {exc}")
            return

        for mod, data in chunks.items():
            pub = self.publishers.get(mod)
            if pub:
                pub.publish(ByteMultiArray(data=list(data)))

    def _merge_part(self, modality: str, msg):
        self._merge_buffer[modality] = bytes(msg.data)
        if all(m in self._merge_buffer for m in MODALITY_ORDER):
            framed = frame_multimodal_payload(self._merge_buffer)
            self.merge_pub.publish(ByteMultiArray(data=list(framed)))
            self._merge_buffer.clear()


def main(args=None):
    rclpy.init(args=args)
    node = AsyncStreamSplitter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
