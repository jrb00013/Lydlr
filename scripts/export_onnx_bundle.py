#!/usr/bin/env python3
# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# Export multimodal compressor to ONNX + Jetson Orin deploy bundle.
#
# Usage:
#   PYTHONPATH=ros2/src/lydlr_ai python3 scripts/export_onnx_bundle.py \
#     --checkpoint models/lydlr_compressor_v2_full_latest.pth \
#     --version v2_full --out deploy_bundles/
"""
ONNX export for Jetson Orin.

The full EnhancedMultimodalCompressor forward returns many tensors and uses
Python control flow (history buffer, keyframe). We export a thin wrapper that
returns stable tensors for TRT/ONNX Runtime:
  compressed (B, D), predicted_quality (B, 1), quant_indices (B, D) as float.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2" / "src" / "lydlr_ai"))


def _build_wrapper(torch, model):
    class CompressorONNXWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.inner.eval()
            self.inner.reset_temporal_state()

        def forward(self, image, lidar, imu, audio):
            out = self.inner(
                image,
                lidar,
                imu,
                audio,
                hidden_state=None,
                compression_level=0.8,
                target_quality=0.8,
                edge_fast=True,
            )
            compressed = out[0]
            predicted_quality = out[7]
            if len(out) >= 12 and out[11] is not None:
                indices = out[11].float()
            else:
                indices = compressed.new_zeros(compressed.shape)
            return compressed, predicted_quality, indices

    return CompressorONNXWrapper(model)


def _resolve_checkpoint(args) -> Path:
    if args.checkpoint:
        p = Path(args.checkpoint)
        if not p.exists():
            raise SystemExit(f"Checkpoint not found: {p}")
        return p

    search_dirs = [
        Path(args.model_dir),
        ROOT / "models",
        ROOT / "ros2" / "src" / "lydlr_ai" / "models",
    ]
    ver = args.version.lstrip("v")
    names = [
        f"lydlr_compressor_v{ver}.pth",
        f"lydlr_compressor_v{ver}_latest.pth",
        f"lydlr_compressor_v2_full_latest.pth",
        f"compressor_v{ver}.pth",
    ]
    for d in search_dirs:
        for name in names:
            cand = d / name
            if cand.exists():
                return cand
    raise SystemExit(
        f"No weights found for version {args.version}. "
        f"Pass --checkpoint explicitly (searched {search_dirs})."
    )


def main():
    parser = argparse.ArgumentParser(description="Export Jetson Orin ONNX bundle")
    parser.add_argument("--version", default="v2_full")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, default=Path("deploy_bundles"))
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--lidar-points", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch required for ONNX export") from exc

    from lydlr_ai.model.compressor import EnhancedMultimodalCompressor

    weights = _resolve_checkpoint(args)
    device = torch.device("cpu")
    model = EnhancedMultimodalCompressor(edge_fast=True).to(device)
    try:
        ckpt = torch.load(weights, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(weights, map_location=device)
    state = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {weights} missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    model.reset_temporal_state()

    wrapper = _build_wrapper(torch, model)
    wrapper.eval()

    bundle_dir = args.out / f"jetson_{args.version}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = bundle_dir / "multimodal_compressor.onnx"

    h, w = args.height, args.width
    dummy_img = torch.randn(1, 3, h, w)
    dummy_lidar = torch.randn(1, args.lidar_points * 3)
    dummy_imu = torch.randn(1, 6)
    dummy_audio = torch.randn(1, 128 * 128)

    print(f"exporting ONNX opset={args.opset} image={h}x{w} …")
    export_kwargs = dict(
        input_names=["image", "lidar", "imu", "audio"],
        output_names=["compressed", "predicted_quality", "quant_indices"],
        opset_version=args.opset,
        dynamic_axes={
            "image": {0: "batch"},
            "lidar": {0: "batch"},
            "imu": {0: "batch"},
            "audio": {0: "batch"},
            "compressed": {0: "batch"},
            "predicted_quality": {0: "batch"},
            "quant_indices": {0: "batch"},
        },
    )
    try:
        torch.onnx.export(
            wrapper,
            (dummy_img, dummy_lidar, dummy_imu, dummy_audio),
            str(onnx_path),
            dynamo=False,
            **export_kwargs,
        )
    except TypeError:
        torch.onnx.export(
            wrapper,
            (dummy_img, dummy_lidar, dummy_imu, dummy_audio),
            str(onnx_path),
            **export_kwargs,
        )

    with torch.no_grad():
        c, q, idx = wrapper(dummy_img, dummy_lidar, dummy_imu, dummy_audio)
    latent_dim = int(c.shape[-1])
    fixed_bits = latent_dim * 8  # L=256 → 1 byte/symbol

    manifest = {
        "artifact_id": f"multimodal_compressor_{args.version}",
        "version": args.version,
        "format": "onnx",
        "target": "jetson_orin",
        "precision": "fp32",
        "inference_backend": "onnx",
        "input_shapes": {
            "image": [1, 3, h, w],
            "lidar": [1, args.lidar_points * 3],
            "imu": [1, 6],
            "audio": [1, 128 * 128],
        },
        "output_names": ["compressed", "predicted_quality", "quant_indices"],
        "latent_dim": latent_dim,
        "codebook_levels": 256,
        "fixed_length_bits_per_sample": fixed_bits,
        "true_rate_note": "Wire claim = packed quant_indices bytes×8 until ANS lands",
        "files": {
            "onnx": onnx_path.name,
            "weights_source": weights.name,
            "weights_path": str(weights),
        },
        "env": {
            "INFERENCE_BACKEND": "onnx",
            "LYDLR_DEPLOY_BUNDLE": str(bundle_dir),
            "MODEL_VERSION": args.version.lstrip("v"),
        },
        "launch_snippet": (
            f"INFERENCE_BACKEND=onnx MODEL_VERSION={args.version.lstrip('v')} "
            "ros2 run lydlr_ai trt_inference_node"
        ),
        "tensorrt_hint": f"./scripts/build_tensorrt_engine.sh {bundle_dir} fp16",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke_shapes": {
            "compressed": list(c.shape),
            "predicted_quality": list(q.shape),
            "quant_indices": list(idx.shape),
        },
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    meta = {
        "version": args.version.lstrip("v"),
        "source_checkpoint": str(weights),
        "latent_dim": latent_dim,
        "image_hw": [h, w],
        "exported_at": manifest["created_at"],
    }
    (bundle_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Bundle ready: {bundle_dir}")
    print(f"  onnx={onnx_path}  latent_dim={latent_dim}  R_fix={fixed_bits} bits")


if __name__ == "__main__":
    main()
