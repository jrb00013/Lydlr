# communication_hub_node.py — fleet graph, discovery, API-facing status
"""
Publishes /lydlr/fleet/status and /lydlr/fleet/graph for observability.
Subscribes to all transport metrics + heartbeats.
"""
import json
import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8MultiArray

from lydlr_ai.communication.topics import LydlrTopics, fleet_node_ids
from lydlr_ai.communication.qos import qos_metrics, qos_command
from lydlr_ai.communication import wire


class CommunicationHubNode(Node):
    def __init__(self):
        super().__init__("communication_hub")
        self.node_ids = fleet_node_ids()
        self._nodes = {nid: {"last_seen": 0, "vertical": "", "model_version": ""} for nid in self.node_ids}
        self._metrics = {}

        self.pub_status = self.create_publisher(String, LydlrTopics.FLEET_STATUS, qos_command())
        self.pub_graph = self.create_publisher(String, LydlrTopics.FLEET_GRAPH, qos_command())

        for nid in self.node_ids:
            self.create_subscription(
                UInt8MultiArray,
                LydlrTopics.metrics_transport(nid),
                lambda msg, n=nid: self._on_metrics(n, msg),
                qos_metrics(),
            )
            self.create_subscription(
                UInt8MultiArray,
                LydlrTopics.heartbeat(nid),
                lambda msg, n=nid: self._on_heartbeat(n, msg),
                qos_metrics(),
            )

        self.create_subscription(
            String,
            LydlrTopics.FLEET_DEPLOY,
            self._on_fleet_deploy,
            qos_command(),
        )

        self.timer = self.create_timer(1.0, self._publish_fleet_status)
        self.get_logger().info(f"🛰️ Communication hub — tracking {self.node_ids}")

    def _on_metrics(self, node_id: str, msg: UInt8MultiArray):
        try:
            m = wire.decode_metrics(wire.from_uint8_array(msg.data))
            self._metrics[node_id] = m
            self._nodes[node_id]["last_seen"] = time.time()
        except Exception as exc:
            self.get_logger().debug(f"metrics {node_id}: {exc}")

    def _on_heartbeat(self, node_id: str, msg: UInt8MultiArray):
        try:
            _, meta = wire.decode_heartbeat(wire.from_uint8_array(msg.data))
            self._nodes[node_id].update(meta)
            self._nodes[node_id]["last_seen"] = time.time()
        except Exception:
            pass

    def _on_fleet_deploy(self, msg: String):
        self.get_logger().info(f"Fleet deploy command: {msg.data}")

    def _publish_fleet_status(self):
        now = time.time()
        status = {
            "timestamp": now,
            "fleet_profile": os.getenv("LYDLR_VERTICAL", "drone"),
            "nodes": [],
        }
        graph = {"nodes": [], "edges": []}

        for nid, info in self._nodes.items():
            alive = (now - info.get("last_seen", 0)) < 5.0
            m = self._metrics.get(nid)
            node_status = {
                "node_id": nid,
                "alive": alive,
                "vertical": info.get("vertical") or (m.vertical if m else ""),
                "model_version": info.get("model_version") or (m.model_version if m else ""),
                "compression_ratio": m.compression_ratio if m else 0,
                "latency_ms": m.latency_ms if m else 0,
                "quality_score": m.quality_score if m else 0,
            }
            status["nodes"].append(node_status)
            graph["nodes"].append({"id": nid, "vertical": node_status["vertical"]})
            graph["edges"].append({"from": nid, "to": "ground_uplink", "type": "compressed"})

        smsg = String()
        smsg.data = json.dumps(status)
        self.pub_status.publish(smsg)

        gmsg = String()
        gmsg.data = json.dumps(graph)
        self.pub_graph.publish(gmsg)


def main(args=None):
    rclpy.init(args=args)
    node = CommunicationHubNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
