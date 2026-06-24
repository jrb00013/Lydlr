#!/usr/bin/env python3
"""Validate Jetson deploy bundle manifest and optional TensorRT engine."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_MANIFEST_KEYS = ("artifact_id", "version", "format", "target", "files")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_bundle(bundle_dir: Path) -> dict:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    for key in REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            raise SystemExit(f"Manifest missing key: {key}")

    onnx_name = manifest.get("files", {}).get("onnx", "multimodal_compressor.onnx")
    onnx_path = bundle_dir / onnx_name
    if not onnx_path.exists():
        raise SystemExit(f"ONNX file missing: {onnx_path}")

    report = {
        "bundle_dir": str(bundle_dir),
        "artifact_id": manifest["artifact_id"],
        "version": manifest["version"],
        "onnx_sha256": sha256_file(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "valid": True,
    }

    trt_path = bundle_dir / manifest.get("files", {}).get("trt", "multimodal_compressor.trt")
    if trt_path.exists():
        report["trt_sha256"] = sha256_file(trt_path)
        report["trt_bytes"] = trt_path.stat().st_size
        report["inference_backend"] = "trt"
    else:
        report["inference_backend"] = manifest.get("inference_backend", "onnx")

    return manifest, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--update-manifest", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    manifest, report = validate_bundle(args.bundle_dir.resolve())
    if args.update_manifest:
        manifest.setdefault("files", {})
        trt_name = "multimodal_compressor.trt"
        trt_path = args.bundle_dir / trt_name
        if trt_path.exists():
            manifest["files"]["trt"] = trt_name
            manifest["inference_backend"] = "trt"
            manifest["precision"] = manifest.get("precision", "fp16")
        manifest["validated_at"] = datetime.now(timezone.utc).isoformat()
        manifest["checksums"] = {
            "onnx_sha256": report["onnx_sha256"],
        }
        if "trt_sha256" in report:
            manifest["checksums"]["trt_sha256"] = report["trt_sha256"]
        (args.bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"OK {report['artifact_id']} backend={report['inference_backend']}")


if __name__ == "__main__":
    main()
