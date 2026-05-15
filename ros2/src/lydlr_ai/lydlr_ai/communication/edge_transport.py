"""
ROS2 transport bindings for edge compressor nodes.
"""
import os
from typing import Callable, Optional

from std_msgs.msg import Float32MultiArray, String, UInt8MultiArray

from lydlr_ai.communication.topics import LydlrTopics
from lydlr_ai.communication.qos import (
    qos_sensor_ingress,
    qos_compressed_egress,
    qos_command,
    qos_coordination,
    qos_metrics,
)
from lydlr_ai.communication import wire


class EdgeTransportLayer:
    """Publishers/subscribers for Lydlr wire transport + legacy topics."""

    def __init__(self, node, node_id: str):
        self._node = node
        self.node_id = node_id
        self.vertical = os.getenv("NODE_VERTICAL", os.getenv("LYDLR_VERTICAL", "drone"))
        self._seq = 0

        self.pub_transport_compressed = node.create_publisher(
            UInt8MultiArray,
            LydlrTopics.compressed_transport(node_id),
            qos_compressed_egress(),
        )
        self.pub_transport_metrics = node.create_publisher(
            UInt8MultiArray,
            LydlrTopics.metrics_transport(node_id),
            qos_metrics(),
        )
        self.pub_legacy_compressed = node.create_publisher(
            UInt8MultiArray,
            LydlrTopics.legacy_compressed(node_id),
            qos_compressed_egress(),
        )
        self.pub_legacy_metrics = node.create_publisher(
            Float32MultiArray,
            LydlrTopics.legacy_metrics(node_id),
            qos_metrics(),
        )
        self.pub_heartbeat = node.create_publisher(
            UInt8MultiArray,
            LydlrTopics.heartbeat(node_id),
            qos_metrics(),
        )

    def subscribe_deploy(self, callback: Callable[[str], None]) -> None:
        def _on(msg: String):
            callback(msg.data)

        self._node.create_subscription(
            String,
            LydlrTopics.deploy(self.node_id),
            _on,
            qos_command(),
        )
        self._node.create_subscription(
            String,
            LydlrTopics.legacy_deploy(self.node_id),
            _on,
            qos_command(),
        )

        def _fleet(msg: String):
            # fleet format: node_id:version  or  *:version
            text = msg.data.strip()
            if ":" in text:
                target, version = text.split(":", 1)
                if target in ("*", "all") or target == self.node_id:
                    callback(version.strip())
            else:
                callback(text)

        self._node.create_subscription(
            String,
            LydlrTopics.FLEET_DEPLOY,
            _fleet,
            qos_command(),
        )

    def subscribe_script(self, callback: Callable[[str], None]) -> None:
        def _on(msg: String):
            callback(msg.data)

        self._node.create_subscription(
            String,
            LydlrTopics.script_load(self.node_id),
            _on,
            qos_command(),
        )
        self._node.create_subscription(
            String,
            "/script/load",
            _on,
            qos_command(),
        )

    def subscribe_coordination(self, callback: Callable[[wire.CoordinationPayload], None]) -> None:
        def _on(msg: UInt8MultiArray):
            try:
                frame = wire.from_uint8_array(msg.data)
                _, payload = wire.decode_coordination(frame)
                callback(payload)
            except Exception as exc:
                self._node.get_logger().debug(f"coordination decode: {exc}")

        self._node.create_subscription(
            UInt8MultiArray,
            LydlrTopics.coordination(self.node_id),
            _on,
            qos_coordination(),
        )
        # Legacy Float32 coordination fallback
        def _legacy(msg: Float32MultiArray):
            if len(msg.data) >= 5:
                callback(
                    wire.CoordinationPayload(
                        target_compression=float(msg.data[0]),
                        allocated_mbps=float(msg.data[1]),
                        fleet_avg_compression=float(msg.data[2]),
                        fleet_avg_latency_ms=float(msg.data[3]),
                        fleet_avg_quality=float(msg.data[4]),
                    )
                )

        self._node.create_subscription(
            Float32MultiArray,
            f"/{self.node_id}/coordination",
            _legacy,
            qos_coordination(),
        )

    def publish_compressed(
        self,
        raw_tensor: bytes,
        model_version: str,
        bytes_in: int,
        compression_ratio: float,
    ) -> None:
        self._seq += 1
        frame = wire.CompressedPayload(
            node_id=self.node_id,
            model_version=model_version or "",
            vertical=self.vertical,
            seq=self._seq,
            bytes_in=bytes_in,
            bytes_out=0,
            compression_ratio=compression_ratio,
            payload=b"",
        )
        packed = wire.encode_compressed(frame, raw_tensor)
        msg = UInt8MultiArray()
        msg.data = wire.to_uint8_array_bytes(packed)
        self.pub_transport_compressed.publish(msg)
        self.pub_legacy_compressed.publish(msg)

    def publish_metrics(self, m: wire.MetricsPayload) -> None:
        packed = wire.encode_metrics(m, seq=self._seq)
        tmsg = UInt8MultiArray()
        tmsg.data = wire.to_uint8_array_bytes(packed)
        self.pub_transport_metrics.publish(tmsg)

        legacy = Float32MultiArray()
        legacy.data = m.to_legacy_floats()
        self.pub_legacy_metrics.publish(legacy)

    def publish_heartbeat(self, model_version: str) -> None:
        hb = wire.encode_heartbeat(self.node_id, self.vertical, model_version)
        msg = UInt8MultiArray()
        msg.data = wire.to_uint8_array_bytes(hb)
        self.pub_heartbeat.publish(msg)


def sensor_qos():
    return qos_sensor_ingress()
