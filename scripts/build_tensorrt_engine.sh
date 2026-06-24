#!/usr/bin/env bash
# Build TensorRT engine from an ONNX deploy bundle (Jetson / x86 with trtexec).
set -euo pipefail

BUNDLE_DIR="${1:-}"
PRECISION="${2:-fp16}"

if [[ -z "$BUNDLE_DIR" || ! -d "$BUNDLE_DIR" ]]; then
  echo "Usage: $0 <deploy_bundle_dir> [fp16|fp32|int8]" >&2
  exit 1
fi

ONNX="$BUNDLE_DIR/multimodal_compressor.onnx"
ENGINE="$BUNDLE_DIR/multimodal_compressor.trt"
MANIFEST="$BUNDLE_DIR/manifest.json"

if [[ ! -f "$ONNX" ]]; then
  echo "Missing ONNX: $ONNX" >&2
  exit 1
fi

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec not found — install TensorRT or run on Jetson" >&2
  exit 2
fi

TRT_ARGS=(--onnx="$ONNX" --saveEngine="$ENGINE")
case "$PRECISION" in
  fp16) TRT_ARGS+=(--fp16) ;;
  int8) TRT_ARGS+=(--int8) ;;
  fp32) ;;
  *) echo "Unknown precision: $PRECISION" >&2; exit 1 ;;
esac

echo "Building TensorRT engine ($PRECISION)..."
trtexec "${TRT_ARGS[@]}"

python3 "$(dirname "$0")/validate_deploy_bundle.py" "$BUNDLE_DIR" --update-manifest

echo "Engine ready: $ENGINE"
