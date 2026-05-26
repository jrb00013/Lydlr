#!/usr/bin/env python3
"""
Export multimodal compressor to ONNX + Jetson deploy bundle manifest.

Usage:
  python scripts/export_onnx_bundle.py --version vv1.0 --out deploy_bundles/
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="vv1.0")
    parser.add_argument("--model-dir", type=Path, default=Path("ros2/src/lydlr_ai/models"))
    parser.add_argument("--out", type=Path, default=Path("deploy_bundles"))
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch required for ONNX export") from exc

    from lydlr_ai.model.compressor import EnhancedMultimodalCompressor

    model_dir = args.model_dir
    candidates = [
        model_dir / f"lydlr_compressor_v{args.version.lstrip('v')}.pth",
        model_dir / f"compressor_v{args.version.lstrip('v')}.pth",
    ]
    weights = next((p for p in candidates if p.exists()), None)
    if not weights:
        raise SystemExit(f"No weights found for version {args.version} in {model_dir}")

    device = torch.device("cpu")
    model = EnhancedMultimodalCompressor().to(device)
    ckpt = torch.load(weights, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    bundle_dir = args.out / f"jetson_{args.version}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = bundle_dir / "multimodal_compressor.onnx"

    dummy_img = torch.randn(1, 3, 224, 224)
    dummy_lidar = torch.randn(1, 1024 * 3)
    dummy_imu = torch.randn(1, 6)
    dummy_audio = torch.randn(1, 128 * 128)

    torch.onnx.export(
        model,
        (dummy_img, dummy_lidar, dummy_imu, dummy_audio, None, 0.8, 0.8),
        str(onnx_path),
        input_names=["image", "lidar", "imu", "audio"],
        output_names=["compressed", "quality"],
        opset_version=17,
        dynamic_axes={
            "image": {0: "batch"},
            "lidar": {0: "batch"},
            "imu": {0: "batch"},
            "audio": {0: "batch"},
        },
    )

    manifest = {
        "artifact_id": f"multimodal_compressor_{args.version}",
        "version": args.version,
        "format": "onnx",
        "target": "jetson_orin",
        "precision": "fp32",
        "files": {
            "onnx": onnx_path.name,
            "weights_source": weights.name,
        },
        "launch_snippet": (
            f"ros2 run lydlr_ai edge_compressor_node  # MODEL_VERSION={args.version}"
        ),
        "tensorrt_hint": f"trtexec --onnx={onnx_path.name} --saveEngine=model.trt --fp16",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Bundle ready: {bundle_dir}")


if __name__ == "__main__":
    main()
