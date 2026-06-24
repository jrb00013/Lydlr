# federated_sync_node.py
import hashlib
import json
import threading
import time
import struct
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from lydlr_ai.communication.link_policy import NodeLinkPolicy


@dataclass
class WeightDelta:
    node_id: str
    round_id: str
    layer_deltas: Dict[str, List[float]]
    delta_size_bytes: int
    checksum_sha256: str
    timestamp: float = 0.0


def _compute_checksum(deltas: Dict[str, List[float]]) -> str:
    raw = json.dumps(deltas, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _estimate_delta_kbps(delta_bytes: int, interval_sec: float = 1.0) -> float:
    if interval_sec <= 0:
        return 0.0
    return (delta_bytes * 8) / interval_sec / 1000.0


def _cap_delta(
    deltas: Dict[str, List[float]],
    max_bytes: int,
) -> Dict[str, List[float]]:
    encoded = json.dumps(deltas, sort_keys=True, separators=(",", ":")).encode()
    if len(encoded) <= max_bytes:
        return deltas
    factor = max_bytes / len(encoded)
    capped = {}
    for layer, values in deltas.items():
        n = max(1, int(len(values) * factor))
        step = max(1, len(values) // n)
        capped[layer] = values[::step]
    return capped


def _fedavg_merge(
    deltas: List[WeightDelta],
    base_weights: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, List[float]]:
    if not deltas:
        return base_weights or {}
    merged: Dict[str, List[float]] = {}
    count = len(deltas)
    for delta in deltas:
        for layer, values in delta.layer_deltas.items():
            if layer not in merged:
                merged[layer] = [0.0] * len(values)
            for i, v in enumerate(values):
                merged[layer][i] += v / count
    if base_weights:
        for layer in merged:
            if layer in base_weights:
                for i in range(min(len(merged[layer]), len(base_weights[layer]))):
                    merged[layer][i] += base_weights[layer][i]
    return merged


def _verify_delta(delta: WeightDelta) -> bool:
    expected = _compute_checksum(delta.layer_deltas)
    return expected == delta.checksum_sha256


class FederatedSyncNode(Node):
    def __init__(self):
        super().__init__("federated_sync_node")

        self.local_model_version = 0
        self.master_model_version = 0
        self.base_weights: Dict[str, List[float]] = {}
        self.pending_deltas: List[WeightDelta] = []
        self.policy = NodeLinkPolicy(node_id="coordinator")

        self.srv = self.create_service(Trigger, "sync_model", self.handle_sync_model)

        self.lock = threading.Lock()
        threading.Thread(target=self.local_training_loop, daemon=True).start()

    def _simulate_weight_delta(self) -> WeightDelta:
        layer_deltas = {
            "encoder.0.weight": [0.001 * (hash(str(i)) % 100) / 100.0 for i in range(64)],
            "encoder.0.bias": [0.0005 * (hash(str(i + 100)) % 100) / 100.0 for i in range(32)],
            "decoder.0.weight": [0.002 * (hash(str(i + 200)) % 100) / 100.0 for i in range(128)],
        }
        encoded = json.dumps(layer_deltas, sort_keys=True, separators=(",", ":")).encode()
        max_bytes = int(self.policy.uplink_budget_kbps * 1000 / 8)
        capped = _cap_delta(layer_deltas, max_bytes)
        checksum = _compute_checksum(capped)
        return WeightDelta(
            node_id=self.get_name(),
            round_id=f"round_{self.local_model_version}",
            layer_deltas=capped,
            delta_size_bytes=len(json.dumps(capped, sort_keys=True, separators=(",", ":")).encode()),
            checksum_sha256=checksum,
            timestamp=time.time(),
        )

    def local_training_loop(self):
        while rclpy.ok():
            time.sleep(5.0)
            with self.lock:
                self.local_model_version += 1
                delta = self._simulate_weight_delta()
                self.pending_deltas.append(delta)
                kbps = _estimate_delta_kbps(delta.delta_size_bytes, 5.0)
                verified = _verify_delta(delta)
                self.get_logger().info(
                    f"Local round {self.local_model_version}: "
                    f"delta={delta.delta_size_bytes}B ({kbps:.1f} kbps), "
                    f"sha256={'OK' if verified else 'FAIL'}"
                )

    def handle_sync_model(self, request, response):
        with self.lock:
            if not self.pending_deltas:
                response.success = False
                response.message = "No pending deltas to sync"
                return response

            merged = _fedavg_merge(self.pending_deltas, self.base_weights)
            self.base_weights = merged
            verified_count = sum(1 for d in self.pending_deltas if _verify_delta(d))
            self.pending_deltas.clear()
            self.master_model_version = self.local_model_version

            response.success = True
            response.message = (
                f"FedAvg merged, master v{self.master_model_version}, "
                f"{verified_count}/{self.local_model_version} deltas verified, "
                f"weights shape: { {k: len(v) for k, v in merged.items()} }"
            )
            self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = FederatedSyncNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
