"""Tests for link budget logic (loads ROS module without rclpy)."""
import importlib.util
from pathlib import Path


def _load_link_policy():
    path = (
        Path(__file__).resolve().parents[2]
        / "ros2/src/lydlr_ai/lydlr_ai/communication/link_policy.py"
    )
    spec = importlib.util.spec_from_file_location("link_policy", path)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["link_policy"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_target_compression_over_budget():
    lp = _load_link_policy()
    policy = lp.NodeLinkPolicy.from_dict("node_0", {"vertical": "drone", "uplink_budget_kbps": 512})
    level = lp.target_compression_level(
        policy,
        estimated_output_kbps=700,
        quality_score=0.8,
        latency_ms=20,
    )
    assert level > 0.8


def test_iot_vision_frame_skip():
    lp = _load_link_policy()
    policy = lp.NodeLinkPolicy.from_dict("iot_gateway_01", {"vertical": "iot"})
    assert lp.vision_frame_skip(policy, publish_hz=10) >= 4
