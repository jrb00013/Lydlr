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
  - rate R_proxy (entropy bits, differentiable)
  - rate R_true (packed quantizer index bits — claim this for link budget)
  - ratio ρ (raw float bytes / latent float payload bytes)
  - latency L (ms)

See docs/architecture/TRUE_RATE_APPLIED_MATH.md.

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
from lydlr_ai.model.true_rate import rate_report  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from structured_synthetic_data import init_scene, step_scene, relative_residual  # noqa: E402


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
        # Eval needs recon for PSNR unless explicitly skipped
        skip_recon=bool(getattr(args, "skip_recon", False)),
        use_fp16=bool(getattr(args, "fp16", False)),
        pretrained_backbone=False,
    ).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"loaded checkpoint missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()
    model.reset_temporal_state()

    rows = []
    scene = init_scene(1, device, height=480, width=640, num_blobs=5)
    prev_image = None
    phis = []
    with torch.no_grad():
        for t in range(args.frames):
            scene, obs = step_scene(scene, cut_prob=getattr(args, "cut_prob", 0.03))
            image, lidar, imu, audio = obs["image"], obs["lidar"], obs["imu"], obs["audio"]
            if prev_image is not None:
                phis.append(relative_residual(prev_image, image))
            prev_image = image

            t0 = time.perf_counter()
            packed = unpack_compressor_output(
                model(
                    image,
                    lidar,
                    imu,
                    audio,
                    target_quality=args.target_quality,
                    edge_fast=args.edge_fast,
                    skip_recon=bool(args.skip_recon),
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
            tr_stats, packed_idx = rate_report(
                packed["rate_bits"],
                packed.get("quant_indices"),
                num_levels=256,
            )
            true_payload = len(packed_idx) if packed_idx else coded_bytes

            row = {
                "frame": t,
                "is_keyframe": bool(packed["is_keyframe"]),
                "mse": float(np.mean((img_np - rec_np) ** 2)),
                "psnr": _psnr(img_np, rec_np),
                "ssim": _ssim_simple(img_np, rec_np),
                "rate_bits": tr_stats["proxy_rate_bits"],  # legacy alias = proxy
                "proxy_rate_bits": tr_stats["proxy_rate_bits"],
                "true_rate_bits": tr_stats["true_rate_bits"],
                "fixed_length_bits": tr_stats["fixed_length_bits"],
                "proxy_vs_true_ratio": tr_stats["proxy_vs_true_ratio"],
                "packed_index_bytes": float(true_payload),
                "ratio": raw_bytes / max(coded_bytes, 1),
                "ratio_true": raw_bytes / max(true_payload, 1),
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
        "mean_proxy_rate_bits": float(np.mean([r["proxy_rate_bits"] for r in rows])),
        "mean_true_rate_bits": float(np.mean([r["true_rate_bits"] for r in rows])),
        "mean_fixed_length_bits": float(np.mean([r["fixed_length_bits"] for r in rows])),
        "mean_proxy_vs_true_ratio": float(
            np.nanmean([r["proxy_vs_true_ratio"] for r in rows])
        ),
        "mean_ratio": float(np.mean([r["ratio"] for r in rows])),
        "mean_ratio_true": float(np.mean([r["ratio_true"] for r in rows])),
        "p50_latency_ms": float(np.median([r["latency_ms"] for r in rows])),
        "keyframe_fraction": float(np.mean([r["is_keyframe"] for r in rows])),
        "edge_fast": bool(args.edge_fast),
        "mean_phi_residual": float(np.mean(phis)) if phis else 0.0,
        "data": "structured_synthetic",
        "lambda_note": "Train with scripts/train_rd_compressor.py --lambda-rd to sweep RD curve",
        "plan": "docs/architecture/NEURAL_COMPRESSION_RD_PLAN.md",
        "true_rate_note": "docs/architecture/TRUE_RATE_APPLIED_MATH.md — claim mean_true_rate_bits for wire",
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
    p.add_argument(
        "--skip-recon",
        action="store_true",
        help="Uplink-only path (no VAE decode). Use on Jetson for latency/stability.",
    )
    p.add_argument("--fp16", action="store_true", help="Autocast fp16 on CUDA")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--cut-prob", type=float, default=0.03)
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
