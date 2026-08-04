#!/usr/bin/env python3
# This file is part of the Lydlr project.
#
# Graduated Jetson Orin probe — find what kills the board WITHOUT starting at full load.
#
# Usage on Orin:
#   PYTHONPATH=ros2/src/lydlr_ai python3 scripts/orin_safe_probe.py --level 3
#   PYTHONPATH=ros2/src/lydlr_ai python3 scripts/orin_safe_probe.py --level 5
#
# Levels:
#   0  host health (mem, nvpmodel, dmesg clues)
#   1  tiny CUDA matmul
#   2  compressor CPU @ 480x640 edge_fast (skip_recon)
#   3  compressor CUDA @ 480x640 edge_fast skip_recon + fp16  ← uplink path
#   4  CUDA ResNet/VAE-encode only
#   5  CUDA full recon (skip_recon=False) — only after 1–4 survive
"""Safe graduated probes for Jetson stability."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2" / "src" / "lydlr_ai"))


def _run(cmd: str) -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.STDOUT, timeout=10
        )
    except Exception as exc:
        return f"(failed: {exc})"


def level0() -> None:
    print("=== L0 host health ===", flush=True)
    print(_run("free -h | head -2"))
    print(_run("nvpmodel -q 2>/dev/null | head -8 || true"))
    print(_run("tegrastats --interval 1000 | head -1 || true"))
    print("recent nvgpu/oom/thermal:")
    print(
        _run(
            "dmesg -T 2>/dev/null | grep -iE 'oom|nvgpu|xid|thermal|throttl|reset|kill' | tail -30 || true"
        )
    )


def level1() -> None:
    print("=== L1 tiny CUDA matmul ===", flush=True)
    import torch
    from lydlr_ai.model.compressor import configure_jetson_runtime

    configure_jetson_runtime(fp16=True)
    assert torch.cuda.is_available(), "CUDA not available"
    print("device", torch.cuda.get_device_name(0), flush=True)
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        c = a @ b
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / 50
    print(f"matmul_ok {ms:.2f} ms/iter  result_mean={float(c.mean()):.4f}", flush=True)


def _load_model(device: str, edge_fast: bool, skip_recon: bool, use_fp16: bool):
    import torch
    from lydlr_ai.model.compressor import (
        EnhancedMultimodalCompressor,
        configure_jetson_runtime,
    )

    if device == "cuda":
        configure_jetson_runtime(fp16=use_fp16)
    model = EnhancedMultimodalCompressor(
        edge_fast=edge_fast,
        skip_recon=skip_recon,
        use_fp16=use_fp16,
        pretrained_backbone=False,
    ).to(device)
    model.eval()
    ckpt = os.environ.get("LYDLR_CKPT", "")
    if ckpt and Path(ckpt).exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        sd = (
            state["model_state_dict"]
            if isinstance(state, dict) and "model_state_dict" in state
            else state
        )
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(
            f"loaded ckpt missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    return model


def _forward(h: int, w: int, device: str, edge_fast: bool, skip_recon: bool, use_fp16: bool) -> None:
    import torch
    from lydlr_ai.model.compressor import unpack_compressor_output

    model = _load_model(device, edge_fast, skip_recon, use_fp16)
    img = torch.randn(1, 3, h, w, device=device)
    lidar = torch.randn(1, 3072, device=device)
    imu = torch.randn(1, 6, device=device)
    audio = torch.randn(1, 16384, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        packed = unpack_compressor_output(
            model(
                img,
                lidar,
                imu,
                audio,
                edge_fast=edge_fast,
                skip_recon=skip_recon,
                target_quality=0.8,
            )
        )
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    recon = packed["recon_img"]
    print(
        f"forward_ok {device} {h}x{w} edge_fast={edge_fast} skip_recon={skip_recon} "
        f"{ms:.1f}ms recon={tuple(recon.shape)} "
        f"Rproxy={float(packed['rate_bits'].mean()):.2f}",
        flush=True,
    )


def level4_encode_only() -> None:
    print("=== L4 CUDA VAE encode-only ===", flush=True)
    import torch
    from lydlr_ai.model.compressor import EnhancedVAE, configure_jetson_runtime

    configure_jetson_runtime(fp16=True)
    vae = EnhancedVAE(latent_dim=64, pretrained_backbone=False).cuda().eval()
    ckpt = os.environ.get("LYDLR_CKPT", "")
    if ckpt and Path(ckpt).exists():
        state = torch.load(ckpt, map_location="cuda", weights_only=False)
        sd = state["model_state_dict"]
        # load only vae.* keys
        vae_sd = {
            k[len("vae.") :]: v for k, v in sd.items() if k.startswith("vae.")
        }
        missing, unexpected = vae.load_state_dict(vae_sd, strict=False)
        print(f"vae loaded missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    x = torch.randn(1, 3, 480, 640, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        mu, logvar = vae.encode(x)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    print(f"encode_ok {ms:.1f}ms mu={tuple(mu.shape)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--level", type=int, default=3, help="Run levels 0..N inclusive")
    p.add_argument(
        "--ckpt",
        type=str,
        default=str(ROOT / "models" / "lydlr_compressor_v2_full_latest.pth"),
    )
    args = p.parse_args()
    if Path(args.ckpt).exists():
        os.environ["LYDLR_CKPT"] = args.ckpt

    level0()
    if args.level < 1:
        return
    level1()
    if args.level < 2:
        return
    print("=== L2 CPU 480x640 edge_fast skip_recon ===", flush=True)
    _forward(480, 640, "cpu", True, True, False)
    if args.level < 3:
        return
    print("=== L3 CUDA 480x640 edge_fast skip_recon fp16 ===", flush=True)
    _forward(480, 640, "cuda", True, True, True)
    if args.level < 4:
        return
    level4_encode_only()
    if args.level < 5:
        return
    print("=== L5 CUDA 480x640 WITH recon (dangerous) ===", flush=True)
    _forward(480, 640, "cuda", True, False, True)
    print("ALL LEVELS OK", flush=True)


if __name__ == "__main__":
    main()
