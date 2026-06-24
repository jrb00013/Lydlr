#!/usr/bin/env python3
"""Export trained PPO policy network to ONNX for edge deployment."""
import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "ros2" / "src" / "lydlr_ai"))

from lydlr_ai.model.rl_policy import CompressionPolicyNet, OBS_DIM

try:
    import torch
except ImportError:
    raise SystemExit("torch is required for ONNX export")


def main():
    parser = argparse.ArgumentParser(description="Export RL policy to ONNX")
    parser.add_argument(
        "--input",
        type=Path,
        default=_REPO / "ros2/src/lydlr_ai/models/rl_compression_policy.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO / "ros2/src/lydlr_ai/models/rl_compression_policy.onnx",
    )
    parser.add_argument("--obs-dim", type=int, default=OBS_DIM)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        sb_model = PPO.load(args.input)
        policy_net = sb_model.policy.mlp_extractor.policy_net
    except Exception:
        print("SB3 not available or model incompatible; using standalone policy net")
        policy_net = CompressionPolicyNet(obs_dim=args.obs_dim)
        ckpt = args.input.with_suffix(".pt")
        if ckpt.exists():
            policy_net.load_state_dict(torch.load(ckpt, map_location="cpu"))

    policy_net.eval()
    dummy = torch.randn(1, args.obs_dim)
    torch.onnx.export(
        policy_net,
        dummy,
        args.output,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=11,
    )
    print(f"ONNX model exported to {args.output}")


if __name__ == "__main__":
    main()
