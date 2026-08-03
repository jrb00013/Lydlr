#!/usr/bin/env python3
# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Train EnhancedMultimodalCompressor with the rate–distortion objective.

Prepares versioned checkpoints for hot-swap:
  models/lydlr_compressor_v2_<stamp>.pth

Usage:
  PYTHONPATH=ros2/src/lydlr_ai python3 scripts/train_rd_compressor.py --epochs 2 --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2" / "src" / "lydlr_ai"))

from lydlr_ai.model.compressor import (  # noqa: E402
    EnhancedMultimodalCompressor,
    compute_rd_loss,
    unpack_compressor_output,
)


def synthetic_batch(batch_size: int, device: torch.device):
    image = torch.rand(batch_size, 3, 480, 640, device=device)
    lidar = torch.rand(batch_size, 1024, 3, device=device)
    imu = torch.randn(batch_size, 6, device=device)
    audio = torch.rand(batch_size, 128 * 128, device=device)
    return image, lidar, imu, audio


def train(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = EnhancedMultimodalCompressor(
        history_len=args.history_len,
        keyframe_period=args.keyframe_period,
        edge_fast=False,
        latent_dim=args.latent_dim,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    steps_per_epoch = args.steps if not args.smoke else 2
    epochs = args.epochs if not args.smoke else 1

    history = []
    model.train()
    for epoch in range(epochs):
        model.reset_temporal_state()
        epoch_metrics = []
        for step in range(steps_per_epoch):
            image, lidar, imu, audio = synthetic_batch(args.batch_size, device)
            # Mild temporal drift so residual coding sees change
            image = (image + 0.02 * step).clamp(0, 1)

            optimizer.zero_grad(set_to_none=True)
            packed = unpack_compressor_output(
                model(
                    image,
                    lidar,
                    imu,
                    audio,
                    compression_level=args.compression_level,
                    target_quality=args.target_quality,
                )
            )
            loss, metrics = compute_rd_loss(
                recon_img=packed["recon_img"],
                image=image,
                mu=packed["mu"],
                logvar=packed["logvar"],
                compressed=packed["compressed"],
                continuous=packed["continuous"],
                temporal_out=packed["temporal_out"],
                predicted_quality=packed["predicted_quality"],
                rate_bits=packed["rate_bits"],
                target_quality=args.target_quality,
                beta=args.beta,
                lambda_rd=args.lambda_rd,
                temporal_to_latent=model.temporal_to_latent,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_metrics.append(metrics)

        avg = {
            k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics)
            for k in epoch_metrics[0]
        }
        history.append({"epoch": epoch, **avg})
        print(
            f"epoch {epoch+1}/{epochs}  "
            f"D={avg['distortion']:.4f}  R={avg['rate_bits']:.3f}  "
            f"L={avg['total']:.4f}  λ={args.lambda_rd}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    version = f"2_{stamp}"
    ckpt_path = out_dir / f"lydlr_compressor_v{version}.pth"
    meta_path = out_dir / f"metadata_lydlr_compressor_v{version}.json"

    payload = {
        "model_state_dict": model.state_dict(),
        "architecture": "EnhancedMultimodalCompressor",
        "architecture_version": 2,
        "latent_dim": args.latent_dim,
        "history_len": args.history_len,
        "keyframe_period": args.keyframe_period,
        "lambda_rd": args.lambda_rd,
        "trained_at": stamp,
    }
    torch.save(payload, ckpt_path)

    meta = {
        "version": version,
        "architecture": "EnhancedMultimodalCompressor",
        "architecture_version": 2,
        "objective": "D + lambda_rd * R",
        "lambda_rd": args.lambda_rd,
        "history": history,
        "plan": "docs/architecture/NEURAL_COMPRESSION_RD_PLAN.md",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"saved {ckpt_path}")
    print(f"saved {meta_path}")
    return ckpt_path


def main():
    p = argparse.ArgumentParser(description="RD compressor training (v2 prep)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--steps", type=int, default=20, help="steps per epoch")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-rd", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--history-len", type=int, default=4)
    p.add_argument("--keyframe-period", type=int, default=8)
    p.add_argument("--compression-level", type=float, default=0.8)
    p.add_argument("--target-quality", type=float, default=0.8)
    p.add_argument("--out-dir", type=str, default=str(ROOT / "models"))
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke", action="store_true", help="one tiny epoch for CI")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
