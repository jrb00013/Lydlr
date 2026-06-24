#!/usr/bin/env python3
"""Train PPO compression controller using stable-baselines3."""
import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "ros2" / "src" / "lydlr_ai"))

from lydlr_ai.model.rl_policy import CompressionControllerEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    raise SystemExit("stable-baselines3 is required. pip install stable-baselines3")


def main():
    parser = argparse.ArgumentParser(description="Train PPO compression controller")
    parser.add_argument("--budget-kbps", type=float, default=512.0)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--sim-steps", type=int, default=200)
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--out", type=Path, default=_REPO / "ros2/src/lydlr_ai/models")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    def make_env():
        return CompressionControllerEnv(
            budget_kbps=args.budget_kbps,
            hz=args.hz,
            sim_steps=args.sim_steps,
        )

    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=args.total_timesteps)

    dest = args.out / "rl_compression_policy.zip"
    model.save(dest)
    print(f"Model saved to {dest}")


if __name__ == "__main__":
    main()
