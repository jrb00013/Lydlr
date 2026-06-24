"""
RL CompressionController for Lydlr edge compression.

Provides:
- CompressionControllerEnv: a Gym-compatible simulation environment
- RLCompressionController: inference wrapper that loads a trained PPO model
  and outputs compression level adjustments. Falls back to heuristic when
  no trained model is available.
"""
from __future__ import annotations

import math
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

SB3_AVAILABLE = False
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    SB3_AVAILABLE = True
except ImportError:
    PPO = None

# ---------------------------------------------------------------------------
# Observation / action space constants
# ---------------------------------------------------------------------------

OBS_KEYS = ["budget_ratio", "quality_score", "latency_ms", "quality_trend", "cpu_load"]
OBS_DIM = len(OBS_KEYS)
ACTION_LOW = -0.3
ACTION_HIGH = 0.3

# ---------------------------------------------------------------------------
# Standalone policy network (torch) for ONNX export / inference without SB3
# ---------------------------------------------------------------------------

if nn is not None:

    class CompressionPolicyNet(nn.Module):
        """Simple MLP policy: obs → action (tanh-scaled)."""

        def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
                nn.Tanh(),
            )

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return self.net(obs) * ACTION_HIGH

else:

    class CompressionPolicyNet:
        """Stub when torch is not available."""

        def __init__(self, *a, **kw):
            raise ImportError("torch is required for CompressionPolicyNet")

        def forward(self, *a, **kw):
            raise ImportError("torch is required for CompressionPolicyNet")


# ---------------------------------------------------------------------------
# Gym-compatible environment
# ---------------------------------------------------------------------------


class CompressionControllerEnv:
    """
    Simulate compression control loop.

    Observation (5,):
      [budget_ratio, quality_score, latency_ms, quality_trend, cpu_load]

    Action (1,):
      compression_level_adjustment ∈ [-0.3, 0.3]

    Reward per step:
      quality_score * 10 - bandwidth_penalty - latency_penalty
    """

    def __init__(
        self,
        budget_kbps: float = 512.0,
        hz: float = 10.0,
        sim_steps: int = 200,
    ):
        self.budget_kbps = budget_kbps
        self.hz = hz
        self.sim_steps = sim_steps
        self._step = 0
        self._base_level = 0.75
        self._quality = 0.85
        self._latency = 20.0
        self._quality_history: list = []
        self._cpu = 0.5

        # gym compatibility
        self.action_space = type("Space", (), {"shape": (1,), "low": ACTION_LOW, "high": ACTION_HIGH})()
        self.observation_space = type("Space", (), {"shape": (OBS_DIM,)})()

    def reset(self) -> np.ndarray:
        self._step = 0
        self._base_level = 0.75
        self._quality = 0.85
        self._latency = 20.0
        self._quality_history = []
        self._cpu = 0.5
        return self._observe()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        adj = float(np.clip(action[0], ACTION_LOW, ACTION_HIGH))
        self._step += 1

        new_level = max(0.1, min(0.98, self._base_level + adj))
        self._base_level = new_level

        phase = self._step / self.hz
        self._quality = max(0.3, min(1.0, 0.85 + 0.1 * math.sin(phase * 0.5) - 0.15 * new_level + 0.02 * random.gauss(0, 1)))
        self._quality_history.append(self._quality)

        self._latency = max(5, 15 + 10 * new_level + 5 * random.gauss(0, 1))
        self._cpu = max(0.1, min(1.0, 0.5 + 0.3 * new_level + 0.05 * random.gauss(0, 1)))

        bandwidth_penalty = max(0, (self._base_level - 0.75) * 5)
        latency_penalty = max(0, (self._latency - 40) * 0.1)
        reward = self._quality * 10 - bandwidth_penalty - latency_penalty

        done = self._step >= self.sim_steps
        return self._observe(), reward, done, {}

    def _observe(self) -> np.ndarray:
        ratio = self._base_level * 1.2
        trend = np.mean(self._quality_history[-5:]) if len(self._quality_history) >= 5 else self._quality
        return np.array(
            [ratio, self._quality, self._latency, trend, self._cpu],
            dtype=np.float32,
        )

    def render(self, mode: str = "human"):
        pass


# ---------------------------------------------------------------------------
# RL Compression Controller — main integration class
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


class RLCompressionController:
    """
    Reinforcement-learning compression controller.

    Modes:
      "heuristic" — delegate to link_policy.target_compression_level()
      "ppo"       — use trained PPO model (stable-baselines3)
      "onnx"      — use ONNX-exported policy

    Usage:
      ctrl = RLCompressionController(mode="heuristic")
      adjustment = ctrl.predict(budget_ratio=1.2, quality_score=0.7, ...)
    """

    def __init__(
        self,
        mode: str = "heuristic",
        model_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        self.mode = mode
        self.device = device
        self._model = None
        self._policy_net: Optional[CompressionPolicyNet] = None
        self._last_action: float = 0.0
        self._last_reward: float = 0.0
        self._step_count: int = 0
        self._reward_ma: float = 0.0

        if mode == "ppo":
            self._load_ppo(model_path)
        elif mode == "onnx":
            self._load_onnx(model_path)
        elif mode != "heuristic":
            raise ValueError(f"Unknown RL mode: {mode}")

    def _load_ppo(self, model_path: Optional[Path] = None):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required for PPO mode")
        path = model_path or MODEL_DIR / "rl_compression_policy.zip"
        if not path.exists():
            raise FileNotFoundError(f"PPO model not found: {path}")
        self._model = PPO.load(path)

    def _load_onnx(self, model_path: Optional[Path] = None):
        if torch is None:
            raise ImportError("torch required for ONNX mode")
        path = model_path or MODEL_DIR / "rl_compression_policy.onnx"
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        try:
            import onnxruntime as ort
            self._ort_session = ort.InferenceSession(str(path))
        except ImportError:
            raise ImportError("onnxruntime required for ONNX inference")

    def predict(
        self,
        budget_ratio: float,
        quality_score: float,
        latency_ms: float,
        quality_trend: float = 0.0,
        cpu_load: float = 0.5,
    ) -> float:
        """
        Return compression level adjustment in [-0.3, 0.3].

        Apply to base level: new_level = clamp(base + adjustment, 0.1, 0.98)
        """
        self._step_count += 1
        obs = np.array(
            [budget_ratio, quality_score, latency_ms, quality_trend, cpu_load],
            dtype=np.float32,
        )

        if self.mode == "heuristic":
            adj = self._heuristic_action(obs)
        elif self.mode == "ppo" and self._model is not None:
            action, _ = self._model.predict(obs, deterministic=True)
            adj = float(action[0])
        elif self.mode == "onnx" and hasattr(self, "_ort_session"):
            inp = {self._ort_session.get_inputs()[0].name: obs.reshape(1, -1)}
            out = self._ort_session.run(None, inp)
            adj = float(out[0][0, 0])
        else:
            adj = 0.0

        adj = max(ACTION_LOW, min(ACTION_HIGH, adj))
        self._last_action = adj

        self._last_reward = self._compute_reward(
            quality_score, budget_ratio, latency_ms
        )
        self._reward_ma = 0.95 * self._reward_ma + 0.05 * self._last_reward

        return adj

    def _heuristic_action(self, obs: np.ndarray) -> float:
        budget_ratio, quality_score, latency_ms, _, _ = obs
        if budget_ratio > 1.1:
            return 0.15
        if quality_score < 0.65:
            return -0.15
        if latency_ms > 50:
            return 0.10
        return 0.0

    @staticmethod
    def _compute_reward(quality: float, budget_ratio: float, latency: float) -> float:
        bw_penalty = max(0, (budget_ratio - 1.0) * 3)
        lat_penalty = max(0, (latency - 50) * 0.05)
        return quality * 10 - bw_penalty - lat_penalty

    @property
    def action(self) -> float:
        return self._last_action

    @property
    def reward(self) -> float:
        return self._last_reward

    @property
    def reward_ma(self) -> float:
        return self._reward_ma

    def summary(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "step": self._step_count,
            "action": round(self._last_action, 4),
            "reward": round(self._last_reward, 4),
            "reward_ma": round(self._reward_ma, 4),
        }
