"""Unit tests for drone/IoT link-budget policy helpers."""

from __future__ import annotations

import pytest

from lydlr_ai.communication.link_policy import (
    NodeLinkPolicy,
    estimate_output_kbps,
    kbps_to_mbps,
    target_compression_level,
    vision_frame_skip,
)


def test_from_dict_applies_vertical_defaults():
    drone = NodeLinkPolicy.from_dict("d1", {"vertical": "drone"})
    assert drone.uplink_budget_kbps == 512.0
    assert drone.vision_fps_cap == 15.0
    assert drone.min_quality == 0.75

    iot = NodeLinkPolicy.from_dict("i1", {"vertical": "iot"})
    assert iot.uplink_budget_kbps == 64.0
    assert iot.vision_fps_cap == 2.0
    assert iot.prioritize[0] == "imu"


def test_kbps_helpers():
    assert kbps_to_mbps(1000) == pytest.approx(1.0)
    assert estimate_output_kbps(1250, interval_sec=0.1) == pytest.approx(100.0)
    assert estimate_output_kbps(100, interval_sec=0) == 0.0


def test_target_compression_raises_when_over_budget():
    policy = NodeLinkPolicy.from_dict("d1", {"vertical": "drone"})
    low = target_compression_level(
        policy,
        estimated_output_kbps=100.0,
        quality_score=0.9,
        latency_ms=20.0,
    )
    high = target_compression_level(
        policy,
        estimated_output_kbps=800.0,
        quality_score=0.9,
        latency_ms=20.0,
    )
    assert high > low
    assert 0.1 <= low <= 0.98
    assert high <= 0.98


def test_quality_floor_softens_compression():
    policy = NodeLinkPolicy.from_dict("d1", {"vertical": "drone", "min_quality": 0.8})
    healthy = target_compression_level(
        policy,
        estimated_output_kbps=600.0,
        quality_score=0.95,
        latency_ms=20.0,
    )
    soft = target_compression_level(
        policy,
        estimated_output_kbps=600.0,
        quality_score=0.5,
        latency_ms=20.0,
    )
    assert soft < healthy


def test_vision_frame_skip():
    policy = NodeLinkPolicy.from_dict("d1", {"vertical": "drone", "vision_fps_cap": 10})
    assert vision_frame_skip(policy, publish_hz=10) == 1
    assert vision_frame_skip(policy, publish_hz=30) >= 2
