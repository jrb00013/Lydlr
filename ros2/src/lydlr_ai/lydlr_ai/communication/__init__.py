"""Lydlr ROS2 communication layer — topics, QoS, wire transport.

QoS helpers depend on ``rclpy`` and are imported lazily so wire / link_policy
unit tests can run on hosts without a ROS 2 install (including CI).
"""

from __future__ import annotations

from typing import Any

from lydlr_ai.communication.topics import LydlrTopics, fleet_node_ids
from lydlr_ai.communication import wire

__all__ = [
    "LydlrTopics",
    "fleet_node_ids",
    "qos_sensor_ingress",
    "qos_compressed_egress",
    "qos_command",
    "qos_coordination",
    "qos_metrics",
    "wire",
]


def __getattr__(name: str) -> Any:
    if name in {
        "qos_sensor_ingress",
        "qos_compressed_egress",
        "qos_command",
        "qos_coordination",
        "qos_metrics",
    }:
        from lydlr_ai.communication import qos as _qos

        return getattr(_qos, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
