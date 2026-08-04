# Jetson Orin Demo — Deploy Path

**Status:** ready for hardware once weights are on the device  
**Goal:** run Lydlr multimodal compression on an NVIDIA Jetson Orin and report **countable** uplink bits, not the training entropy proxy.

Related: [TRUE_RATE_APPLIED_MATH.md](../architecture/TRUE_RATE_APPLIED_MATH.md) · [FULL_RD_TRAIN_HANDOFF.md](FULL_RD_TRAIN_HANDOFF.md)

---

## What “demo ready” means

| Layer | Artifact | Claim on demo day |
|-------|----------|-------------------|
| Weights | `lydlr_compressor_v2_full_*.pth` | Trained RD compressor |
| Edge PyTorch | ModelRegistry under `models/<node_id>/` | Hot-swap without rebuild |
| ONNX bundle | `deploy_bundles/jetson_v2_full/` | Cross-device inference |
| TensorRT (optional) | `multimodal_compressor.trt` | Orin latency |
| Rate | packed `quant_indices` | **R_true ≈ 512 bits/sample** (L=256, d=64) until ANS |

Do **not** quote train `Rproxy≈0.16` as the radio bitrate.

---

## Host prep (dev machine)

```bash
cd /path/to/Lydlr
export PYTHONPATH=ros2/src/lydlr_ai

# 1) Eval with true-rate fields
python3 scripts/eval_compression_rd.py \
  --checkpoint models/lydlr_compressor_v2_full_latest.pth \
  --frames 16 --out models/eval_orin_prep.json

# 2) Install into edge registry layout
python3 scripts/install_edge_checkpoint.py \
  --checkpoint models/lydlr_compressor_v2_full_latest.pth \
  --version 2_full --node-id node_0 --smoke

# 3) Export ONNX (480×640 matches train)
python3 scripts/export_onnx_bundle.py \
  --checkpoint models/lydlr_compressor_v2_full_latest.pth \
  --version v2_full --out deploy_bundles/
```

Copy to Orin:

```bash
rsync -av deploy_bundles/jetson_v2_full/ orin:/opt/lydlr/deploy_bundles/jetson_v2_full/
rsync -av models/node_0/lydlr_compressor_v2_full.pth orin:/opt/lydlr/models/node_0/
```

---

## Live IMX + realtime visualization

You already have **Visual Monitoring** in the Lydlr frontend (`VisualMonitoring.js`) fed by:
- `POST /api/nodes/<id>/preview/` (raw / reconstructed / heatmap JPEGs)
- metrics WebSocket / REST

What’s new for Orin is the **live bridge**:

```bash
cd ~/Lydlr && git pull
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH=ros2/src/lydlr_ai
# Optional: feed the main dashboard
export LYDLR_API_URL=http://<dev-machine-or-orin>:8000/api

# CSI IMX477 is often /dev/video0; USB Arducam may be /dev/video1.
# Prefer V4L2 index first. Use --camera csi0 only if V4L2 fails.
python3 scripts/orin_live_imx_demo.py \
  --checkpoint models/lydlr_compressor_v2_full_latest.pth \
  --camera 0 --port 8765
# USB IMX alternate:
# python3 scripts/orin_live_imx_demo.py --camera 1 --port 8765
```

Open **`http://<orin-ip>:8765/`** for realtime raw | VAE recon | heatmap + Rtrue/PSNR/latency.

**Bring-up order (don’t skip):** models load → camera open → first frames. Avoid starting Argus CSI while also slamming first CUDA load; the demo loads models *before* opening the camera.

**Hybrid (Orin-safe):** CUDA compress with `skip_recon`; **CPU** runs full VAE decode for visualization/eval PSNR so we do not re-trigger the CUDA ConvTranspose hang.

| Piece | Role |
|-------|------|
| `scripts/orin_live_imx_demo.py` | Live IMX + hybrid recon + local MJPEG viz |
| Frontend Visual Monitoring | Fleet charts + MJPEG when API is fed |
| `edge_compressor_node` previews | Same JPEG sides over ROS / API |

## Crash lessons (do this first)

Full `480×640` **CUDA + VAE decode** previously **hard-hung** an AGX Orin (network death, no SSH). CPU path was fine. Suspects: first-touch cuDNN on ResNet + ConvTranspose, desktop DRM conflict, and a progressive decode that overshot to ~960×1280.

**Mitigations in code (`edge_fast`):**
- `skip_recon=True` by default (encode/quantize uplink only — no ConvTranspose decode on device)
- `use_fp16=True` autocast on CUDA
- `pretrained_backbone=False` (no ImageNet download at init)
- hardcoded 480×640 ResNet map dims (no dummy full-res init forward)
- `configure_jetson_runtime()` — cudnn.benchmark off
- progressive decode caps spatial overshoot when recon is enabled

**Before any demo load:**

```bash
cd ~/Lydlr && git pull
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH=ros2/src/lydlr_ai
python3 scripts/orin_safe_probe.py --level 3   # CUDA uplink path (skip_recon)
# only if that survives:
python3 scripts/orin_safe_probe.py --level 5   # adds full recon — optional
```

Eval with metrics needs recon:

```bash
python3 scripts/eval_compression_rd.py --checkpoint models/lydlr_compressor_v2_full_latest.pth \
  --frames 4 --edge-fast --fp16   # skip_recon OFF → PSNR meaningful
```

Uplink-only latency:

```bash
python3 scripts/eval_compression_rd.py ... --edge-fast --skip-recon --fp16
```

## On the Orin

1. **JetPack** with CUDA + (optional) TensorRT `trtexec`.
2. Run `orin_safe_probe.py` levels 0→5 (above).
3. Build engine (fp16 recommended) only after probes pass:
   ```bash
   ./scripts/build_tensorrt_engine.sh /opt/lydlr/deploy_bundles/jetson_v2_full fp16
   ```
4. **PyTorch edge path** (simplest first demo):
   ```bash
   export NODE_ID=node_0
   export PYTHONPATH=ros2/src/lydlr_ai
   # Ensure models/node_0/lydlr_compressor_v2_full.pth exists
   ros2 run lydlr_ai edge_compressor_node
   ```
5. **ONNX/TRT path** when `trt_inference_node` is configured:
   ```bash
   export INFERENCE_BACKEND=onnx   # or tensorrt
   export LYDLR_DEPLOY_BUNDLE=/opt/lydlr/deploy_bundles/jetson_v2_full
   export MODEL_VERSION=2_full
   ros2 run lydlr_ai trt_inference_node
   ```

---

## Demo checklist

- [ ] Checkpoint loads (`strict=False` OK; log missing/unexpected counts)
- [ ] One forward at 480×640 (or document resize) with camera/LiDAR/IMU/audio stubs
- [ ] Log **proxy_rate_bits** vs **true_rate_bits** (expect proxy ≪ true until ANS)
- [ ] Show reconstructed preview / quality score
- [ ] Optional: TRT fp16 latency vs PyTorch

---

## Input shapes (train-matched)

| Tensor | Shape |
|--------|--------|
| image | `(1, 3, 480, 640)` |
| lidar | `(1, 3072)` = 1024×3 |
| imu | `(1, 6)` |
| audio | `(1, 16384)` = 128×128 |
| compressed / indices | `(1, 64)` |

`edge_fast=True` skips attention/multiscale for CPU/bandwidth pressure — use it on Orin if latency is tight.
