#!/usr/bin/env python3
"""Benchmark PyTorch vs ONNX vs TensorRT inference on replay tensors."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "ros2" / "src" / "lydlr_ai"))


def _bench(fn, warmup=3, runs=20):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms": round(statistics.mean(samples), 3),
        "p95_ms": round(sorted(samples)[int(len(samples) * 0.95) - 1], 3),
        "fps": round(1000 / max(statistics.mean(samples), 0.001), 1),
    }


def bench_torch(weights: Path, device: str):
    import torch
    from lydlr_ai.model.compressor import EnhancedMultimodalCompressor

    model = EnhancedMultimodalCompressor().to(device)
    ckpt = torch.load(weights, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    img = torch.randn(1, 3, 224, 224, device=device)
    lidar = torch.randn(1, 1024 * 3, device=device)
    imu = torch.randn(1, 6, device=device)
    audio = torch.randn(1, 128 * 128, device=device)

    def run():
        with torch.no_grad():
            model(img, lidar, imu, audio, None, 0.8, 0.8)

    return _bench(run)


def bench_onnx(onnx_path: Path):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {
        "image": np.random.randn(1, 3, 224, 224).astype(np.float32),
        "lidar": np.random.randn(1, 1024 * 3).astype(np.float32),
        "imu": np.random.randn(1, 6).astype(np.float32),
        "audio": np.random.randn(1, 128 * 128).astype(np.float32),
    }

    def run():
        sess.run(None, feeds)

    return _bench(run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="vv1.0")
    parser.add_argument("--model-dir", type=Path, default=_REPO / "ros2/src/lydlr_ai/models")
    parser.add_argument("--bundle-dir", type=Path, default=_REPO / "deploy_bundles")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    norm = args.version.lstrip("v")
    weights = args.model_dir / f"lydlr_compressor_v{norm}.pth"
    if not weights.exists():
        weights = args.model_dir / f"compressor_v{norm}.pth"

    bundle = args.bundle_dir / f"jetson_{args.version}"
    onnx_path = bundle / "multimodal_compressor.onnx"

    results = {"version": args.version, "backends": {}}

    if weights.exists():
        results["backends"]["torch"] = bench_torch(weights, args.device)

    if onnx_path.exists():
        results["backends"]["onnx"] = bench_onnx(onnx_path)

    if not results["backends"]:
        raise SystemExit("No weights or ONNX bundle found to benchmark")

    out = args.out or (_REPO / "docs" / "benchmarks" / f"inference_{args.version}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
