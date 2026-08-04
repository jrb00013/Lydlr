#!/usr/bin/env python3
# This file is part of the Lydlr project.
#
# Graduated Jetson Orin probe — find what kills the board WITHOUT starting at full load.
#
# Usage on Orin:
#   PYTHONPATH=ros2/src/lydlr_ai python3 scripts/orin_safe_probe.py
#   PYTHONPATH=ros2/src/lydlr_ai python3 scripts/orin_safe_probe.py --level 3
#
# Levels:
#   0  host health (mem, nvpmodel, dmesg clues)
#   1  tiny CUDA matmul
#   2  compressor CPU forward @ 64x64
#   3  compressor CUDA forward @ 64x64 edge_fast
#   4  compressor CUDA @ 240x320 edge_fast
#   5  compressor CUDA @ 480x640 edge_fast (only after 1–4 survive)
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
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT, timeout=10)
    except Exception as exc:
        return f"(failed: {exc})"


def level0() -> None:
    print("=== L0 host health ===")
    print(_run("free -h | head -2"))
    print(_run("nvpmodel -q 2>/dev/null | head -8 || true"))
    print(_run("tegrastats --interval 1000 | head -1 || true"))
    print("recent nvgpu/oom/thermal:")
    print(_run("dmesg -T 2>/dev/null | grep -iE 'oom|nvgpu|xid|thermal|throttl|reset|kill' | tail -30 || true"))
    print(_run("journalctl -b -p err --no-pager 2>/dev/null | tail -20 || true"))


def level1() -> None:
    print("=== L1 tiny CUDA matmul ===")
    import torch

    assert torch.cuda.is_available(), "CUDA not available"
    print("device", torch.cuda.get_device_name(0))
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        c = a @ b
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / 50
    print(f"matmul_ok {ms:.2f} ms/iter  result_mean={float(c.mean()):.4f}")


def _forward(h: int, w: int, device: str, edge_fast: bool) -> None:
    import torch
    from lydlr_ai.model.compressor import EnhancedMultimodalCompressor, unpack_compressor_output

    model = EnhancedMultimodalCompressor(edge_fast=edge_fast).to(device)
    model.eval()
    ckpt = os.environ.get("LYDLR_CKPT", "")
    if ckpt and Path(ckpt).exists():
        state = torch.load(ckpt, map_location=device)
        sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded ckpt missing={len(missing)} unexpected={len(unexpected)}")

    img = torch.randn(1, 3, h, w, device=device)
    lidar = torch.randn(1, 3072, device=device)
    imu = torch.randn(1, 6, device=device)
    audio = torch.randn(1, 16384, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        packed = unpack_compressor_output(
            model(img, lidar, imu, audio, edge_fast=edge_fast, target_quality=0.8)
        )
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    recon = packed["recon_img"]
    print(
        f"forward_ok {device} {h}x{w} edge_fast={edge_fast} "
        f"{ms:.1f}ms recon={tuple(recon.shape)} "
        f"Rproxy={float(packed['rate_bits'].mean()):.2f}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--level", type=int, default=3, help="Run levels 0..N inclusive")
    p.add_argument("--ckpt", type=str, default=str(ROOT / "models" / "lydlr_compressor_v2_full_latest.pth"))
    args = p.parse_args()
    if Path(args.ckpt).exists():
        os.environ["LYDLR_CKPT"] = args.ckpt

    level0()
    if args.level < 1:
        return
    level1()
    if args.level < 2:
        return
    print("=== L2 CPU 64x64 ===")
    _forward(64, 64, "cpu", True)
    if args.level < 3:
        return
    print("=== L3 CUDA 64x64 edge_fast ===")
    _forward(64, 64, "cuda", True)
    if args.level < 4:
        return
    print("=== L4 CUDA 240x320 edge_fast ===")
    _forward(240, 320, "cuda", True)
    if args.level < 5:
        return
    print("=== L5 CUDA 480x640 edge_fast ===")
    _forward(480, 640, "cuda", True)
    print("ALL LEVELS OK")


if __name__ == "__main__":
    main()
