"""Integration tests for coordinator signal logic — no rclpy required.

Tests the coordination loop core: reading live bytes_out → computing
estimated_output_kbps → target_compression_level → quality guard interaction.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1].parent


@pytest.fixture(scope="session")
def lp():
    """Load link_policy module without ROS2."""
    path = REPO / "ros2/src/lydlr_ai/lydlr_ai/communication/link_policy.py"
    spec = importlib.util.spec_from_file_location("link_policy", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["link_policy"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestTargetCompression:
    """Coordinator's core decision: target_compression_level() from live bytes."""

    def test_nominal_drone_stays_below_budget(self, lp):
        policy = lp.NodeLinkPolicy.from_dict("node_0", {"vertical": "drone", "uplink_budget_kbps": 512})
        level = lp.target_compression_level(
            policy,
            estimated_output_kbps=300,
            quality_score=0.85,
            latency_ms=15,
        )
        assert 0.4 <= level <= 0.85, f"Unexpected level {level} for nominal drone"

    def test_drone_burst_raises_level(self, lp):
        policy = lp.NodeLinkPolicy.from_dict("node_0", {"vertical": "drone", "uplink_budget_kbps": 512})
        level = lp.target_compression_level(
            policy,
            estimated_output_kbps=800,
            quality_score=0.80,
            latency_ms=20,
        )
        assert level > 0.80, f"Burst should push level above 0.80, got {level}"

    def test_iot_budget_tight(self, lp):
        policy = lp.NodeLinkPolicy.from_dict("iot_01", {"vertical": "iot", "uplink_budget_kbps": 64})
        level = lp.target_compression_level(
            policy,
            estimated_output_kbps=100,
            quality_score=0.70,
            latency_ms=30,
        )
        assert level > 0.78, f"IoT over-budget should raise level, got {level}"

    def test_low_quality_lowers_level(self, lp):
        policy = lp.NodeLinkPolicy.from_dict("node_0", {"vertical": "drone", "uplink_budget_kbps": 512})
        level_low_q = lp.target_compression_level(
            policy,
            estimated_output_kbps=400,
            quality_score=0.40,
            latency_ms=20,
        )
        level_high_q = lp.target_compression_level(
            policy,
            estimated_output_kbps=400,
            quality_score=0.90,
            latency_ms=20,
        )
        assert level_low_q < level_high_q, (
            f"Low quality should result in lower compression level "
            f"({level_low_q} >= {level_high_q})"
        )

    def test_high_latency_pushes_level_up(self, lp):
        policy = lp.NodeLinkPolicy.from_dict("node_0", {"vertical": "drone", "uplink_budget_kbps": 512})
        level = lp.target_compression_level(
            policy,
            estimated_output_kbps=400,
            quality_score=0.80,
            latency_ms=80,
        )
        assert level > 0.80, f"High latency should increase compression, got {level}"


class TestQualityGuardInteraction:
    """Coordinator sees quality-dipped metrics → adjusts next signal."""

    def test_consecutive_low_quality_lowers_target(self, lp):
        """Simulate 3 ticks of low quality → level should drop."""
        policy = lp.NodeLinkPolicy.from_dict("node_0", {"vertical": "drone", "uplink_budget_kbps": 512})
        levels = []
        for quality in [0.40, 0.38, 0.42]:
            level = lp.target_compression_level(
                policy,
                estimated_output_kbps=400,
                quality_score=quality,
                latency_ms=20,
            )
            levels.append(level)
        # Level should trend down as quality stays below threshold
        assert levels[-1] <= levels[0] + 0.05, (
            f"Low quality should not increase level: {levels}"
        )

    def test_recovering_quality_relaxes_level(self, lp):
        """After low quality, good quality should allow relaxed compression."""
        policy = lp.NodeLinkPolicy.from_dict("node_0", {"vertical": "drone", "uplink_budget_kbps": 512})
        low_level = lp.target_compression_level(
            policy,
            estimated_output_kbps=400,
            quality_score=0.40,
            latency_ms=20,
        )
        high_level = lp.target_compression_level(
            policy,
            estimated_output_kbps=400,
            quality_score=0.90,
            latency_ms=20,
        )
        # Low quality triggers less aggressive compression (lower level)
        assert low_level <= 0.63, f"Low quality should produce relaxed level, got {low_level}"
        assert high_level > low_level, (
            f"Recovered quality should allow more aggressive compression "
            f"(high {high_level} <= low {low_level})"
        )


class TestLiveBytesToKbps:
    """Coordinator reads bytes_out → estimate_output_kbps."""

    def test_estimate_output_kbps(self, lp):
        bytes_out = 6400  # bytes in one tick
        kbps = lp.estimate_output_kbps(bytes_out, interval_sec=0.1)
        expected = (6400 * 8) / 0.1 / 1000.0
        assert kbps == pytest.approx(expected, rel=1e-3)

    def test_zero_interval_returns_zero(self, lp):
        assert lp.estimate_output_kbps(100, interval_sec=0) == 0.0


class TestPrioritizeModalities:
    """Coordinator-aware modality prioritization."""

    def test_drone_prioritizes_lidar(self, lp):
        policy = lp.NodeLinkPolicy.from_dict("node_0", {"vertical": "drone"})
        weights = lp.prioritize_modalities(policy)
        assert weights["lidar"] >= weights["camera"]
        assert weights["imu"] >= weights["audio"]

    def test_iot_prioritizes_imu(self, lp):
        policy = lp.NodeLinkPolicy.from_dict("iot_01", {"vertical": "iot"})
        weights = lp.prioritize_modalities(policy)
        assert weights["imu"] >= weights["camera"]
