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

Full real train:
  PYTHONPATH=ros2/src/lydlr_ai python3 scripts/train_rd_compressor.py --preset full

Resume after interrupt:
  PYTHONPATH=ros2/src/lydlr_ai python3 scripts/train_rd_compressor.py --preset full \\
    --resume models/lydlr_compressor_v2_full_latest.pth

See docs/guides/FULL_RD_TRAIN_HANDOFF.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

sys.path.insert(0, str(ROOT / "scripts"))
from structured_synthetic_data import init_scene, step_scene, relative_residual  # noqa: E402

PRESETS = {
    "smoke": {"epochs": 1, "steps": 2, "seq_len": 1, "lambda_rd": 0.05},
    "short": {"epochs": 25, "steps": 40, "seq_len": 2, "lambda_rd": 0.05},
    "full": {"epochs": 100, "steps": 100, "seq_len": 4, "lambda_rd": 0.05},
}


def synthetic_batch(batch_size: int, device: torch.device):
    image = torch.rand(batch_size, 3, 480, 640, device=device)
    lidar = torch.rand(batch_size, 1024, 3, device=device)
    imu = torch.randn(batch_size, 6, device=device)
    audio = torch.rand(batch_size, 128 * 128, device=device)
    return image, lidar, imu, audio


def save_checkpoint(
    model,
    optimizer,
    out_dir: Path,
    *,
    version: str,
    args: argparse.Namespace,
    history: list,
    epoch: int,
    step_global: int,
    tag: str = "",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"lydlr_compressor_v{version}{tag}.pth"
    ckpt_path = out_dir / name
    meta_path = out_dir / f"metadata_lydlr_compressor_v{version}{tag}.json"

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "architecture": "EnhancedMultimodalCompressor",
        "architecture_version": 2,
        "latent_dim": args.latent_dim,
        "history_len": args.history_len,
        "keyframe_period": args.keyframe_period,
        "lambda_rd": args.lambda_rd,
        "epoch": epoch,
        "step_global": step_global,
        "history": history,
        "args": {
            "epochs": args.epochs,
            "steps": args.steps,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lambda_rd": args.lambda_rd,
            "preset": args.preset,
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(payload, ckpt_path)

    # Rolling latest pointer for easy resume
    latest = out_dir / "lydlr_compressor_v2_full_latest.pth"
    torch.save(payload, latest)

    meta = {
        "version": version,
        "tag": tag,
        "architecture": "EnhancedMultimodalCompressor",
        "architecture_version": 2,
        "objective": "D + lambda_rd * R",
        "lambda_rd": args.lambda_rd,
        "epoch": epoch,
        "step_global": step_global,
        "history_tail": history[-5:],
        "plan": "docs/architecture/NEURAL_COMPRESSION_RD_PLAN.md",
        "handoff": "docs/guides/FULL_RD_TRAIN_HANDOFF.md",
        "latest_path": str(latest),
        "checkpoint_path": str(ckpt_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    (out_dir / "FULL_TRAIN_STATUS.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {ckpt_path}", flush=True)
    print(f"saved {latest} (resume pointer)", flush=True)
    return ckpt_path


def train(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)

    model = EnhancedMultimodalCompressor(
        history_len=args.history_len,
        keyframe_period=args.keyframe_period,
        edge_fast=False,
        latent_dim=args.latent_dim,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history: list = []
    start_epoch = 0
    step_global = 0

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as exc:
                print(f"optimizer state not restored ({exc}); continuing with fresh optimizer")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        step_global = int(ckpt.get("step_global", 0))
        history = list(ckpt.get("history", []))
        print(
            f"resumed from {resume_path} at epoch {start_epoch}/{args.epochs} "
            f"(missing={len(missing)} unexpected={len(unexpected)})"
        )

    steps_per_epoch = args.steps if not args.smoke else 2
    epochs = args.epochs if not args.smoke else 1
    seq_len = args.seq_len if not args.smoke else 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    version = f"2_full_{stamp}"
    out_dir = Path(args.out_dir)
    t_run0 = time.perf_counter()

    model.train()
    for epoch in range(start_epoch, epochs):
        model.reset_temporal_state()
        epoch_metrics = []
        t_epoch0 = time.perf_counter()

        # Mild λ anneal: start softer on rate, tighten later
        progress = epoch / max(epochs - 1, 1)
        lambda_rd = args.lambda_rd * (0.5 + 0.5 * progress)

        for step in range(steps_per_epoch):
            # New clip each step: Markov scene state (see TRAINING_DATA_APPLIED_MATH.md)
            scene = init_scene(args.batch_size, device, height=480, width=640, num_blobs=5)
            step_loss = 0.0
            step_metrics = []
            prev_image = None
            phi_vals = []

            for t in range(seq_len):
                scene, obs = step_scene(scene, cut_prob=args.cut_prob)
                image, lidar, imu, audio = obs["image"], obs["lidar"], obs["imu"], obs["audio"]
                if prev_image is not None:
                    phi_vals.append(relative_residual(prev_image, image))
                prev_image = image.detach()
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
                    lambda_rd=lambda_rd,
                    temporal_to_latent=model.temporal_to_latent,
                    quant_indices=packed.get("quant_indices"),
                )
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step_loss += float(loss.detach())
                step_metrics.append(metrics)
                step_global += 1

            avg_step = {
                k: sum(m[k] for m in step_metrics) / len(step_metrics)
                for k in step_metrics[0]
            }
            avg_step["phi_residual"] = sum(phi_vals) / max(len(phi_vals), 1) if phi_vals else 0.0
            epoch_metrics.append(avg_step)

        if not epoch_metrics:
            print(f"epoch {epoch+1}/{epochs}  SKIPPED (no finite steps)", flush=True)
            continue
        avg = {
            k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics)
            for k in epoch_metrics[0]
        }
        avg["lambda_rd_effective"] = float(lambda_rd)
        avg["epoch_sec"] = time.perf_counter() - t_epoch0
        avg["sec_per_step"] = avg["epoch_sec"] / max(steps_per_epoch * seq_len, 1)
        history.append({"epoch": epoch, **avg})

        elapsed = time.perf_counter() - t_run0
        remaining_epochs = max(epochs - epoch - 1, 0)
        eta = remaining_epochs * avg["epoch_sec"]
        print(
            f"epoch {epoch+1}/{epochs}  "
            f"D={avg['distortion']:.4f}  "
            f"Rproxy={avg['rate_bits']:.3f}  "
            f"Rtrue={avg.get('true_rate_bits', 0):.1f}  "
            f"L={avg['total']:.4f}  λ={lambda_rd:.4f}  "
            f"φ={avg.get('phi_residual', 0):.3f}  "
            f"Drec={avg.get('recon_loss', 0):.4f}  "
            f"KL={avg.get('kl_loss', 0):.3f}  "
            f"{avg['sec_per_step']:.2f}s/step  "
            f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m",
            flush=True,
        )

        if args.save_every and ((epoch + 1) % args.save_every == 0 or epoch + 1 == epochs):
            save_checkpoint(
                model,
                optimizer,
                out_dir,
                version=version,
                args=args,
                history=history,
                epoch=epoch,
                step_global=step_global,
                tag=f"_e{epoch+1}",
            )

    # Final save
    return save_checkpoint(
        model,
        optimizer,
        out_dir,
        version=version,
        args=args,
        history=history,
        epoch=epochs - 1,
        step_global=step_global,
        tag="_final",
    )


def main():
    p = argparse.ArgumentParser(description="RD compressor training (v2)")
    p.add_argument("--preset", choices=sorted(PRESETS.keys()), default="", help="smoke|short|full")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--steps", type=int, default=20, help="optimizer steps per epoch")
    p.add_argument("--seq-len", type=int, default=1, help="temporal frames per step")
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
    p.add_argument("--save-every", type=int, default=10, help="checkpoint every N epochs")
    p.add_argument("--resume", type=str, default="", help="path to .pth to continue")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--cut-prob", type=float, default=0.03, help="scene-cut probability per frame")
    p.add_argument("--smoke", action="store_true", help="one tiny epoch for CI")
    args = p.parse_args()

    if args.smoke:
        args.preset = "smoke"
    if args.preset:
        cfg = PRESETS[args.preset]
        args.epochs = cfg["epochs"]
        args.steps = cfg["steps"]
        args.seq_len = cfg["seq_len"]
        args.lambda_rd = cfg["lambda_rd"]
        print(f"preset={args.preset} -> epochs={args.epochs} steps={args.steps} seq_len={args.seq_len}", flush=True)

    train(args)


if __name__ == "__main__":
    main()
