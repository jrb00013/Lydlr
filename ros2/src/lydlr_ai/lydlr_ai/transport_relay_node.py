# transport_relay_node.py — simulates constrained downlink (drone GCS / IoT LPWAN)
"""
Aggregates per-node LYDT compressed streams onto ground uplink topics.
Applies bandwidth shaping so total egress respects GROUND_UPLINK_MBPS.
"""
import os
import time
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray

from lydlr_ai.communication.topics import LydlrTopics, fleet_node_ids
from lydlr_ai.communication.qos import qos_compressed_egress, qos_metrics
from lydlr_ai.communication import wire


class TransportRelayNode(Node):
    def __init__(self):
        super().__init__("transport_relay")
        self.node_ids = fleet_node_ids()
        self.uplink_mbps = float(os.getenv("GROUND_UPLINK_MBPS", "2.0"))
        self._bytes_this_window = 0
        self._window_start = time.time()
        self._lock = threading.Lock()
        self._queue = []
        self._max_queue = int(os.getenv("RELAY_QUEUE_SIZE", "64"))

        self.pub_uplink = self.create_publisher(
            UInt8MultiArray,
            LydlrTopics.ground_uplink(),
            qos_compressed_egress(20),
        )
        self.pub_ground_metrics = self.create_publisher(
            UInt8MultiArray,
            LydlrTopics.ground_metrics(),
            qos_metrics(),
        )

        for nid in self.node_ids:
            self.create_subscription(
                UInt8MultiArray,
                LydlrTopics.compressed_transport(nid),
                lambda msg, n=nid: self._ingress(n, msg),
                qos_compressed_egress(15),
            )
            self.get_logger().info(f"Relay listening: {LydlrTopics.compressed_transport(nid)}")

        self.drain_timer = self.create_timer(0.05, self._drain_queue)
        self.get_logger().info(
            f"📡 Transport relay — {len(self.node_ids)} nodes → "
            f"{LydlrTopics.ground_uplink()} @ {self.uplink_mbps} Mbps cap"
        )

    def _ingress(self, node_id: str, msg: UInt8MultiArray):
        with self._lock:
            if len(self._queue) >= self._max_queue:
                self._queue.pop(0)
            self._queue.append((time.time(), node_id, bytes(msg.data)))

    def _drain_queue(self):
        now = time.time()
        if now - self._window_start >= 1.0:
            self._bytes_this_window = 0
            self._window_start = now

        budget_bytes = (self.uplink_mbps * 1e6 / 8.0)  # per second
        with self._lock:
            if not self._queue:
                return
            _, node_id, frame = self._queue.pop(0)

        if self._bytes_this_window + len(frame) > budget_bytes:
            self.get_logger().debug(f"Uplink cap — dropped frame from {node_id}")
            return

        out = UInt8MultiArray()
        out.data = list(frame)
        self.pub_uplink.publish(out)
        self._bytes_this_window += len(frame)

        try:
            c = wire.decode_compressed(frame)
            m = wire.MetricsPayload(
                node_id=node_id,
                vertical=c.vertical,
                model_version=c.model_version,
                compression_ratio=c.compression_ratio,
                bytes_in=c.bytes_in,
                bytes_out=c.bytes_out,
                latency_ms=0,
                quality_score=0,
                bandwidth_estimate=self.uplink_mbps,
            )
            g = UInt8MultiArray()
            g.data = wire.to_uint8_array_bytes(wire.encode_metrics(m))
            self.pub_ground_metrics.publish(g)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = TransportRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
