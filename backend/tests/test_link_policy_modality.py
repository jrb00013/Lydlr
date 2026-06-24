"""Link policy modality gating tests."""
import importlib.util
import sys
from pathlib import Path


def _load_link_policy():
    path = Path(__file__).resolve().parents[2] / "ros2/src/lydlr_ai/lydlr_ai/communication/link_policy.py"
    spec = importlib.util.spec_from_file_location("link_policy", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["link_policy"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_should_drop_audio_when_over_budget():
    lp = _load_link_policy()
    policy = lp.NodeLinkPolicy.from_dict("iot_gateway_01", {"vertical": "iot"})
    weights = lp.prioritize_modalities(policy)
    assert lp.should_transmit_modality("imu", weights, budget_ratio=1.0)
    assert not lp.should_transmit_modality("audio", weights, budget_ratio=1.5)


def test_lidar_keep_ratio_decreases_with_compression():
    lp = _load_link_policy()
    low = lp.lidar_keep_ratio(0.2)
    high = lp.lidar_keep_ratio(0.9)
    assert high < low
