"""Tests for the RL compression controller (no SB3 required)."""
import importlib.util
import sys
from pathlib import Path

import pytest

import numpy as np

REPO = Path(__file__).resolve().parents[1].parent
RL_PATH = REPO / "ros2/src/lydlr_ai/lydlr_ai/model/rl_policy.py"


def _load_rl():
    spec = importlib.util.spec_from_file_location("rl_policy", RL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rl_policy"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_env_reset_returns_obs():
    rl = _load_rl()
    env = rl.CompressionControllerEnv(budget_kbps=512, sim_steps=50)
    obs = env.reset()
    assert obs.shape == (rl.OBS_DIM,)
    assert obs.dtype == np.float32


def test_env_step_returns_valid():
    rl = _load_rl()
    env = rl.CompressionControllerEnv(budget_kbps=512, sim_steps=50)
    env.reset()
    obs, reward, done, _ = env.step(np.array([0.1]))
    assert obs.shape == (rl.OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_env_step_simulates():
    rl = _load_rl()
    env = rl.CompressionControllerEnv(budget_kbps=512, sim_steps=10)
    env.reset()
    total = 0.0
    for _ in range(10):
        _, reward, done, _ = env.step(np.array([0.05]))
        total += reward
        if done:
            break
    assert total != 0.0


def test_env_terminates_after_sim_steps():
    rl = _load_rl()
    env = rl.CompressionControllerEnv(budget_kbps=512, sim_steps=5)
    env.reset()
    for _ in range(4):
        _, _, done, _ = env.step(np.array([0.0]))
        assert not done
    _, _, done, _ = env.step(np.array([0.0]))
    assert done


def test_rl_heuristic_mode_predicts():
    rl = _load_rl()
    ctrl = rl.RLCompressionController(mode="heuristic")
    adj = ctrl.predict(budget_ratio=1.0, quality_score=0.8, latency_ms=20)
    assert rl.ACTION_LOW <= adj <= rl.ACTION_HIGH


def test_rl_heuristic_raises_on_burst():
    rl = _load_rl()
    ctrl = rl.RLCompressionController(mode="heuristic")
    adj_high = ctrl.predict(budget_ratio=1.5, quality_score=0.8, latency_ms=20)
    adj_low = ctrl.predict(budget_ratio=0.5, quality_score=0.9, latency_ms=10)
    assert adj_high >= adj_low, "Heuristic should be more aggressive when over budget"


def test_rl_heuristic_lowers_on_low_quality():
    rl = _load_rl()
    ctrl = rl.RLCompressionController(mode="heuristic")
    adj_bad = ctrl.predict(budget_ratio=1.0, quality_score=0.5, latency_ms=20)
    adj_good = ctrl.predict(budget_ratio=1.0, quality_score=0.9, latency_ms=20)
    assert adj_bad <= adj_good, "Heuristic should be less aggressive on low quality"


def test_rl_controller_summary():
    rl = _load_rl()
    ctrl = rl.RLCompressionController(mode="heuristic")
    ctrl.predict(budget_ratio=1.2, quality_score=0.7, latency_ms=30)
    s = ctrl.summary()
    assert s["mode"] == "heuristic"
    assert s["step"] == 1
    assert isinstance(s["action"], float)
    assert isinstance(s["reward"], float)


def test_rl_controller_reward_computation():
    rl = _load_rl()
    ctrl = rl.RLCompressionController(mode="heuristic")
    ctrl.predict(budget_ratio=0.8, quality_score=0.9, latency_ms=15)
    assert ctrl.reward > 0, "Good quality + low budget ratio should give positive reward"
    ctrl.predict(budget_ratio=2.0, quality_score=0.4, latency_ms=100)
    assert ctrl.reward < 0, "Poor quality + over budget + high latency should give negative reward"


def test_rl_env_different_seeds():
    rl = _load_rl()
    env1 = rl.CompressionControllerEnv(budget_kbps=256, hz=5, sim_steps=30)
    env2 = rl.CompressionControllerEnv(budget_kbps=512, hz=10, sim_steps=50)
    assert env1.budget_kbps == 256
    assert env2.budget_kbps == 512
    assert env1.hz == 5
    assert env2.hz == 10


def test_compression_policy_net_forward():
    rl = _load_rl()
    if rl.torch is None:
        pytest.skip("torch not available")
    net = rl.CompressionPolicyNet(obs_dim=rl.OBS_DIM)
    dummy = rl.torch.randn(1, rl.OBS_DIM)
    out = net(dummy)
    assert out.shape == (1, 1)
    assert -0.31 <= out.item() <= 0.31


def test_ppp_mode_raises_without_model():
    rl = _load_rl()
    ctrl = rl.RLCompressionController(mode="heuristic")  # should work
    assert ctrl.mode == "heuristic"
    # PPO mode without model file should raise FileNotFoundError
    try:
        ctrl2 = rl.RLCompressionController(
            mode="ppo",
            model_path=Path("/nonexistent/model.zip"),
        )
        assert False, "Should have raised"
    except (FileNotFoundError, ImportError):
        pass
