"""
Link-budget rules for drone vs IoT edge compression.
Used by the distributed coordinator and edge compressor nodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


VERTICAL_DEFAULTS: Dict[str, Dict] = {
    "drone": {
        "uplink_budget_kbps": 512,
        "vision_fps_cap": 15,
        "prioritize": ["lidar", "imu", "camera", "audio"],
        "min_quality": 0.75,
    },
    "iot": {
        "uplink_budget_kbps": 64,
        "vision_fps_cap": 2,
        "prioritize": ["imu", "lidar", "camera", "audio"],
        "min_quality": 0.65,
    },
}


@dataclass
class NodeLinkPolicy:
    node_id: str
    vertical: str = "drone"
    uplink_budget_kbps: float = 512.0
    vision_fps_cap: float = 10.0
    prioritize: List[str] = field(default_factory=lambda: ["lidar", "imu", "camera"])
    min_quality: float = 0.7
    allocated_mbps: float = 0.5

    @classmethod
    def from_dict(cls, node_id: str, data: Optional[Dict] = None) -> "NodeLinkPolicy":
        data = data or {}
        vertical = (data.get("vertical") or "drone").lower()
        defaults = VERTICAL_DEFAULTS.get(vertical, VERTICAL_DEFAULTS["drone"])
        budget = float(data.get("uplink_budget_kbps") or defaults["uplink_budget_kbps"])
        return cls(
            node_id=node_id,
            vertical=vertical,
            uplink_budget_kbps=budget,
            vision_fps_cap=float(data.get("vision_fps_cap") or defaults["vision_fps_cap"]),
            prioritize=list(data.get("prioritize") or defaults["prioritize"]),
            min_quality=float(data.get("min_quality") or defaults["min_quality"]),
            allocated_mbps=budget / 1000.0,
        )


def kbps_to_mbps(kbps: float) -> float:
    return max(kbps, 1.0) / 1000.0


def estimate_output_kbps(bytes_out: int, interval_sec: float = 0.1) -> float:
    if interval_sec <= 0:
        return 0.0
    return (bytes_out * 8) / interval_sec / 1000.0


def target_compression_level(
    policy: NodeLinkPolicy,
    *,
    estimated_output_kbps: float,
    quality_score: float,
    latency_ms: float,
) -> float:
    """
    Return compression level 0.1–0.98 based on link budget vs estimated uplink use.
    Higher = more aggressive compression.
    """
    budget = max(policy.uplink_budget_kbps, 8.0)
    ratio = estimated_output_kbps / budget

    level = 0.75
    if ratio > 1.1:
        level = min(0.98, 0.75 + (ratio - 1.0) * 0.35)
    elif ratio > 0.85:
        level = 0.82
    elif ratio < 0.4:
        level = max(0.45, 0.75 - (0.4 - ratio) * 0.5)

    if quality_score < policy.min_quality:
        level = max(0.35, level - 0.12)

    if latency_ms > 50:
        level = min(0.98, level + 0.08)
    elif latency_ms < 15 and ratio < 0.7:
        level = max(0.4, level - 0.05)

    return round(max(0.1, min(0.98, level)), 3)


def vision_frame_skip(policy: NodeLinkPolicy, publish_hz: float) -> int:
    """Skip N-1 of every N camera frames to respect vision_fps_cap."""
    cap = max(policy.vision_fps_cap, 0.5)
    if publish_hz <= cap:
        return 1
    return max(1, int(round(publish_hz / cap)))


def prioritize_modalities(policy: NodeLinkPolicy) -> Dict[str, float]:
    """Weight per modality when over budget (lower weight = drop first)."""
    order = policy.prioritize
    weights = {}
    for idx, mod in enumerate(reversed(order)):
        weights[mod] = 0.25 + idx * 0.2
    for mod in ("camera", "lidar", "imu", "audio"):
        weights.setdefault(mod, 0.5)
    return weights
