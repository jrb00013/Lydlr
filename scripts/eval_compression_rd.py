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
Evaluate neural compression on synthetic multimodal batches.

Reports dimensionless RD operating points from
docs/architecture/NEURAL_COMPRESSION_RD_PLAN.md:
  - distortion D (MSE / PSNR / SSIM)
  - rate R (entropy bits proxy)
  - ratio ρ (raw float bytes / latent payload bytes)
  - latency L (ms)

Usage:
  PYTHONPATH=ros2/src/lydlr_ai python3 scripts/eval_compression_rd.py --frames 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2" / "src" / "lydlr_ai"))

from lydlr_ai.model.compressor import (  # noqa: E402
    EnhancedMultimodalCompressor,
    unpack_compressor_output,
)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def _ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    """Lightweight SSIM on grayscale mean channels (no skimage required)."""
    x = a.mean(axis=0)
    y = b.mean(axis=0)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x, mu_y = x.mean(), y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return float(num / den)


def eval_model(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = EnhancedMultimodalCompressor(
        history_len=args.history_len,
        keyframe_period=args.keyframe_period,
        edge_fast=args.edge_fast,
    ).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"loaded checkpoint missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()
    model.reset_temporal_state()

    rows = []
    with torch.no_grad():
        for t in range(args.frames):
            image = (torch.rand(1, 3, 480, 640, device=device) + 0.015 * t).clamp(0, 1)
            lidar = torch.rand(1, 1024, 3, device=device)
            imu = torch.randn(1, 6, device=device)
            audio = torch.rand(1, 128 * 128, device=device)

            t0 = time.perf_counter()
            packed = unpack_compressor_output(
                model(
                    image,
                    lidar,
                    imu,
                    audio,
                    target_quality=args.target_quality,
                    edge_fast=args.edge_fast,
                )
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            recon = packed["recon_img"].clamp(0, 1)
            img_np = image[0].cpu().numpy()
            rec_np = recon[0].cpu().numpy()
            # Match spatial size if progressive scale differs
            if rec_np.shape != img_np.shape:
                rec_t = torch.nn.functional.interpolate(
                    recon, size=image.shape[-2:], mode="bilinear", align_corners=False
                )
                rec_np = rec_t[0].cpu().numpy()

            compressed = packed["compressed"].cpu().numpy().astype(np.float32)
            raw_bytes = (
                image.numel() * 4
                + lidar.numel() * 4
                + imu.numel() * 4
                + audio.numel() * 4
            )
            coded_bytes = compressed.nbytes
            rate_bits = float(packed["rate_bits"].mean().cpu())

            row = {
                "frame": t,
                "is_keyframe": bool(packed["is_keyframe"]),
                "mse": float(np.mean((img_np - rec_np) ** 2)),
                "psnr": _psnr(img_np, rec_np),
                "ssim": _ssim_simple(img_np, rec_np),
                "rate_bits": rate_bits,
                "ratio": raw_bytes / max(coded_bytes, 1),
                "latency_ms": latency_ms,
                "edge_fast": bool(args.edge_fast),
            }
            rows.append(row)

    summary = {
        "frames": len(rows),
        "mean_psnr": float(np.mean([r["psnr"] for r in rows])),
        "mean_ssim": float(np.mean([r["ssim"] for r in rows])),
        "mean_mse": float(np.mean([r["mse"] for r in rows])),
        "mean_rate_bits": float(np.mean([r["rate_bits"] for r in rows])),
        "mean_ratio": float(np.mean([r["ratio"] for r in rows])),
        "p50_latency_ms": float(np.median([r["latency_ms"] for r in rows])),
        "keyframe_fraction": float(np.mean([r["is_keyframe"] for r in rows])),
        "edge_fast": bool(args.edge_fast),
        "lambda_note": "Train with scripts/train_rd_compressor.py --lambda-rd to sweep RD curve",
        "plan": "docs/architecture/NEURAL_COMPRESSION_RD_PLAN.md",
        "frames_detail": rows,
    }
    return summary


def main():
    p = argparse.ArgumentParser(description="RD compression eval harness")
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--history-len", type=int, default=4)
    p.add_argument("--keyframe-period", type=int, default=8)
    p.add_argument("--target-quality", type=float, default=0.8)
    p.add_argument("--edge-fast", action="store_true")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    summary = eval_model(args)
    text = json.dumps({k: v for k, v in summary.items() if k != "frames_detail"}, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
