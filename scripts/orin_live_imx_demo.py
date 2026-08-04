#!/usr/bin/env python3
# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# Live IMX → Lydlr compress → VAE recon → realtime visualization (Orin-safe).
#
# Strategy (learned the hard way):
#   - CUDA: edge_fast + skip_recon (uplink path that survives Jetson)
#   - CPU:  VAE decode for full reconstruction preview / PSNR
#   - Viz:  local MJPEG page + optional POST into existing Visual Monitoring API
#
# Usage on Orin:
#   export PATH="$HOME/.local/bin:$PATH" PYTHONPATH=ros2/src/lydlr_ai
#   python3 scripts/orin_live_imx_demo.py \
#     --checkpoint models/lydlr_compressor_v2_full_latest.pth \
#     --camera 0 --port 8765
#
# Open http://<orin-ip>:8765/  for realtime raw | recon | heatmap + metrics.
# Point LYDLR_API_URL at the control-plane to feed frontend Visual Monitoring.
"""Live IMX Lydlr demo with hybrid CUDA compress + CPU recon + realtime viz."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2" / "src" / "lydlr_ai"))

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("opencv-python required: pip install opencv-python") from exc

import torch

from lydlr_ai.model.compressor import (  # noqa: E402
    EnhancedMultimodalCompressor,
    EnhancedVAE,
    configure_jetson_runtime,
    unpack_compressor_output,
)
from lydlr_ai.model.true_rate import rate_report  # noqa: E402

try:
    from lydlr_ai.utils.preview_reporter import report_preview
    from lydlr_ai.utils.metrics_reporter import report_metrics
except ImportError:  # pragma: no cover
    report_preview = None
    report_metrics = None


class LiveState:
    """Thread-safe latest frames + metrics for the HTTP viz server."""

    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg: Dict[str, bytes] = {}
        self.metrics: Dict = {
            "frames": 0,
            "latency_ms": 0.0,
            "psnr": 0.0,
            "proxy_rate_bits": 0.0,
            "true_rate_bits": 0.0,
            "quality": 0.0,
            "camera": "",
            "mode": "hybrid_cuda_encode_cpu_recon",
        }

    def update(self, sides: Dict[str, bytes], metrics: Dict) -> None:
        with self.lock:
            self.jpeg.update(sides)
            self.metrics.update(metrics)
            self.metrics["ts"] = time.time()

    def get_jpeg(self, side: str) -> Optional[bytes]:
        with self.lock:
            return self.jpeg.get(side)

    def get_metrics(self) -> Dict:
        with self.lock:
            return dict(self.metrics)


STATE = LiveState()


def _gst_pipelines(camera: str, width: int, height: int) -> list:
    """Candidate OpenCV capture pipelines for Jetson IMX / V4L2."""
    pipes = []
    if camera.startswith("csi") or camera in ("0", "nvargus0"):
        sensor = "0" if camera in ("0", "nvargus0", "csi") else camera.replace("csi", "")
        pipes.append(
            f"nvarguscamerasrc sensor-id={sensor} ! "
            f"video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1 ! "
            f"nvvidconv ! video/x-raw,width={width},height={height},format=BGRx ! "
            f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
        )
    # V4L2 device index or path
    if camera.isdigit():
        pipes.append(int(camera))
        pipes.append(
            f"v4l2src device=/dev/video{camera} ! "
            f"video/x-raw,width={width},height={height} ! "
            f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
        )
    if camera.startswith("/dev/video"):
        pipes.append(camera)
        pipes.append(
            f"v4l2src device={camera} ! videoconvert ! "
            f"video/x-raw,format=BGR ! appsink drop=1"
        )
    return pipes


def open_camera(camera: str, width: int, height: int) -> cv2.VideoCapture:
    last_err = None
    for pipe in _gst_pipelines(camera, width, height):
        try:
            if isinstance(pipe, int):
                cap = cv2.VideoCapture(pipe)
            elif isinstance(pipe, str) and pipe.startswith("/dev/"):
                cap = cv2.VideoCapture(pipe)
            else:
                cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
            if cap is not None and cap.isOpened():
                # Prefer trained geometry when V4L2 allows
                if isinstance(pipe, (int, str)) and (
                    isinstance(pipe, int) or pipe.startswith("/dev/")
                ):
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                ok, frame = cap.read()
                if ok and frame is not None:
                    print(f"camera_ok pipe={pipe!r} shape={frame.shape}", flush=True)
                    return cap
                cap.release()
        except Exception as exc:
            last_err = exc
    raise SystemExit(f"Could not open camera={camera!r} (last_err={last_err})")


def bgr_to_tensor(frame_bgr: np.ndarray, h: int, w: int, device: torch.device) -> torch.Tensor:
    if frame_bgr.shape[0] != h or frame_bgr.shape[1] != w:
        frame_bgr = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def tensor_to_jpeg(t: torch.Tensor, max_dim: int = 480) -> bytes:
    arr = t.detach().float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    rgb = (arr * 255.0).astype(np.uint8)
    if max(rgb.shape[0], rgb.shape[1]) > max_dim:
        scale = max_dim / max(rgb.shape[0], rgb.shape[1])
        rgb = cv2.resize(
            rgb,
            (int(rgb.shape[1] * scale), int(rgb.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 75]
    )
    return buf.tobytes() if ok else b""


def heatmap_jpeg(raw: torch.Tensor, recon: torch.Tensor, max_dim: int = 480) -> bytes:
    a = raw.detach().float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    b = recon.detach().float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    diff = np.abs(a - b).mean(axis=2)
    diff_u8 = np.clip(diff * 255.0 * 4.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(diff_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    base = (a * 255.0).astype(np.uint8)
    overlay = cv2.addWeighted(base, 0.55, heat_rgb, 0.45, 0)
    if max(overlay.shape[0], overlay.shape[1]) > max_dim:
        scale = max_dim / max(overlay.shape[0], overlay.shape[1])
        overlay = cv2.resize(
            overlay,
            (int(overlay.shape[1] * scale), int(overlay.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 75]
    )
    return buf.tobytes() if ok else b""


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(torch.mean((a.clamp(0, 1) - b.clamp(0, 1)) ** 2).cpu())
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def load_models(ckpt_path: Path, device: torch.device):
    configure_jetson_runtime(fp16=device.type == "cuda")
    model = EnhancedMultimodalCompressor(
        edge_fast=True,
        skip_recon=True,
        use_fp16=device.type == "cuda",
        pretrained_backbone=False,
    ).to(device).eval()
    # CPU VAE for full recon (avoids CUDA ConvTranspose hang on Orin)
    vae_cpu = EnhancedVAE(latent_dim=64, pretrained_backbone=False).cpu().eval()

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"gpu_model missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        vae_sd = {k[len("vae.") :]: v for k, v in sd.items() if k.startswith("vae.")}
        m2, u2 = vae_cpu.load_state_dict(vae_sd, strict=False)
        print(f"cpu_vae missing={len(m2)} unexpected={len(u2)}", flush=True)
    return model, vae_cpu


HTML = """<!doctype html>
<html><head><meta charset="utf-8"/><title>Lydlr Live IMX</title>
<style>
  :root { color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }
  body { margin: 0; background: #0b1020; color: #e8eefc; }
  header { padding: 1rem 1.25rem; border-bottom: 1px solid #24304d; }
  h1 { margin: 0; font-size: 1.25rem; letter-spacing: 0.04em; }
  .sub { opacity: 0.7; font-size: 0.9rem; margin-top: 0.25rem; }
  .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; padding: 0.75rem; }
  figure { margin: 0; background: #121a2e; border: 1px solid #24304d; border-radius: 8px; overflow: hidden; }
  figcaption { padding: 0.5rem 0.75rem; font-size: 0.85rem; opacity: 0.85; }
  img { width: 100%; display: block; background: #000; aspect-ratio: 4/3; object-fit: contain; }
  .metrics { display: grid; grid-template-columns: repeat(6, 1fr); gap: 0.5rem; padding: 0 0.75rem 1rem; }
  .card { background: #121a2e; border: 1px solid #24304d; border-radius: 8px; padding: 0.75rem; }
  .card b { display: block; font-size: 1.1rem; margin-top: 0.2rem; }
  @media (max-width: 900px) { .grid, .metrics { grid-template-columns: 1fr; } }
</style></head>
<body>
<header>
  <h1>LYDLR · Live IMX</h1>
  <div class="sub">CUDA compress (skip_recon) · CPU VAE recon · realtime preview</div>
</header>
<section class="grid">
  <figure><img id="raw" src="/mjpeg/raw"/><figcaption>Raw camera</figcaption></figure>
  <figure><img id="recon" src="/mjpeg/reconstructed"/><figcaption>VAE reconstruction</figcaption></figure>
  <figure><img id="heat" src="/mjpeg/heatmap"/><figcaption>Error heatmap</figcaption></figure>
</section>
<section class="metrics" id="metrics"></section>
<script>
async function tick() {
  try {
    const m = await (await fetch('/metrics.json?' + Date.now())).json();
    const el = document.getElementById('metrics');
    const cells = [
      ['Frames', m.frames],
      ['Latency', (m.latency_ms||0).toFixed(1) + ' ms'],
      ['PSNR', (m.psnr||0).toFixed(2) + ' dB'],
      ['Rtrue', (m.true_rate_bits||0).toFixed(0) + ' bits'],
      ['Rproxy', (m.proxy_rate_bits||0).toFixed(2)],
      ['Quality', (m.quality||0).toFixed(3)],
    ];
    el.innerHTML = cells.map(([k,v]) => `<div class="card">${k}<b>${v}</b></div>`).join('');
  } catch (e) {}
}
setInterval(tick, 500); tick();
</script>
</body></html>
"""


def make_handler():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                body = HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/metrics.json":
                body = json.dumps(STATE.get_metrics()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path.startswith("/mjpeg/"):
                side = path.split("/")[-1]
                side = {"raw": "raw", "reconstructed": "reconstructed", "heatmap": "heatmap"}.get(
                    side, side
                )
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    while True:
                        jpg = STATE.get_jpeg(side)
                        if jpg:
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                + str(len(jpg)).encode()
                                + b"\r\n\r\n"
                                + jpg
                                + b"\r\n"
                            )
                            self.wfile.flush()
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    return
            if path.startswith("/jpeg/"):
                side = path.split("/")[-1]
                jpg = STATE.get_jpeg(side) or b""
                self.send_response(200 if jpg else 404)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                if jpg:
                    self.wfile.write(jpg)
                return
            self.send_response(404)
            self.end_headers()

    return Handler


def inference_loop(args, stop: threading.Event):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}", flush=True)
    model, vae_cpu = load_models(Path(args.checkpoint), device)
    cap = open_camera(args.camera, args.width, args.height)
    h, w = args.height, args.width
    frame_i = 0

    while not stop.is_set():
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.02)
            continue
        image = bgr_to_tensor(frame, h, w, device)
        lidar = torch.zeros(1, 3072, device=device)
        imu = torch.zeros(1, 6, device=device)
        audio = torch.zeros(1, 16384, device=device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            packed = unpack_compressor_output(
                model(
                    image,
                    lidar,
                    imu,
                    audio,
                    edge_fast=True,
                    skip_recon=True,
                    target_quality=0.8,
                )
            )
            # Full VAE recon on CPU (Orin-safe)
            mu = packed["mu"].detach().float().cpu()
            logvar = packed["logvar"].detach().float().cpu()
            z = vae_cpu.reparameterize(mu, logvar)
            recon = vae_cpu.decode_progressive(z, target_scale=2).clamp(0, 1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        tr, _ = rate_report(packed["rate_bits"], packed.get("quant_indices"), num_levels=256)
        q = float(packed["predicted_quality"].float().mean().cpu())
        image_cpu = image.detach().float().cpu()
        psnr_v = psnr(image_cpu, recon)

        sides = {
            "raw": tensor_to_jpeg(image_cpu),
            "reconstructed": tensor_to_jpeg(recon),
            "heatmap": heatmap_jpeg(image_cpu, recon),
        }
        metrics = {
            "frames": frame_i + 1,
            "latency_ms": latency_ms,
            "psnr": psnr_v,
            "proxy_rate_bits": tr["proxy_rate_bits"],
            "true_rate_bits": tr["true_rate_bits"],
            "quality": q,
            "camera": args.camera,
            "mode": "hybrid_cuda_encode_cpu_recon",
            "device": str(device),
        }
        STATE.update(sides, metrics)

        # Feed existing Visual Monitoring control plane when configured
        if report_preview is not None:
            for side, jpg in sides.items():
                report_preview(args.node_id, side, jpg)
        if report_metrics is not None:
            raw_bytes = image.numel() * 4
            out_bytes = int(tr["true_rate_bits"] / 8.0)
            report_metrics(
                node_id=args.node_id,
                compression_ratio=raw_bytes / max(out_bytes, 1),
                latency_ms=latency_ms,
                quality_score=q,
                bytes_in=raw_bytes,
                bytes_out=out_bytes,
            )

        frame_i += 1
        if frame_i % 30 == 0:
            print(
                f"frame={frame_i} {latency_ms:.1f}ms PSNR={psnr_v:.2f} "
                f"Rtrue={tr['true_rate_bits']:.0f} Rproxy={tr['proxy_rate_bits']:.2f}",
                flush=True,
            )
        if args.max_frames and frame_i >= args.max_frames:
            stop.set()
            break

    cap.release()


def main():
    p = argparse.ArgumentParser(description="Live IMX Lydlr demo + realtime viz")
    p.add_argument("--checkpoint", default=str(ROOT / "models" / "lydlr_compressor_v2_full_latest.pth"))
    p.add_argument("--camera", default="0", help="0|/dev/video0|csi0 — IMX CSI or V4L2")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--node-id", default="node_0")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--bind", default="0.0.0.0")
    args = p.parse_args()

    stop = threading.Event()
    worker = threading.Thread(target=inference_loop, args=(args, stop), daemon=True)
    worker.start()

    server = ThreadingHTTPServer((args.bind, args.port), make_handler())
    print(f"viz http://{args.bind}:{args.port}/  (open from your laptop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop.set()
    finally:
        stop.set()
        server.shutdown()


if __name__ == "__main__":
    main()
