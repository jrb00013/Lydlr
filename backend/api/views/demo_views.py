"""
Demo / synthetic live metrics + preview frames for Visual Monitoring
when ROS edge nodes are not running.
"""
import asyncio
import math
import os
import time
from datetime import datetime, timezone
from typing import Dict, List

from django.http import JsonResponse
from rest_framework import status
from rest_framework.response import Response

from backend.api.views.base import AsyncAPIView, ensure_db_connection
from backend.api.redis_pubsub import publish_message
from backend.api.views.preview_views import store_preview_frame, PREVIEW_SIDES


def _demo_enabled() -> bool:
    return os.getenv("LYDLR_DEMO_METRICS", "1") not in ("0", "false", "False")


def _synth_metric(node_id: str, t: float, vertical: str = "drone") -> Dict:
    pulse = 0.5 + 0.5 * math.sin(t * 0.7 + hash(node_id) % 7)
    storm = 0.5 + 0.5 * math.sin(t * 0.23 + 1.3)
    scale = 1.0 if vertical == "drone" else 0.35
    bytes_in = int((160000 + pulse * 380000 + storm * 70000) * scale)
    ratio = 2.0 + pulse * 7.0 + storm * 1.8
    bytes_out = max(2000, int(bytes_in / ratio))
    return {
        "node_id": node_id,
        "vertical": vertical,
        "compression_ratio": round(ratio, 3),
        "latency_ms": round(10 + storm * 60 + pulse * 15, 2),
        "quality_score": round(max(0.55, 0.94 - storm * 0.2 - pulse * 0.04), 4),
        "bandwidth_estimate": round(0.3 + storm * 0.5, 3),
        "compression_level": round(0.4 + pulse * 0.45, 3),
        "bytes_in": bytes_in,
        "bytes_out": bytes_out,
        "modality_bytes_in": {
            "camera": int(bytes_in * 0.55),
            "lidar": int(bytes_in * 0.22),
            "imu": int(bytes_in * 0.08),
            "audio": int(bytes_in * 0.15),
        },
        "modality_bytes_out": {
            "camera": int(bytes_out * 0.5),
            "lidar": int(bytes_out * 0.25),
            "imu": int(bytes_out * 0.1),
            "audio": int(bytes_out * 0.15),
        },
        "modality_quality": {
            "camera": round(0.9 - storm * 0.1, 3),
            "lidar": round(0.88 - pulse * 0.05, 3),
            "imu": 0.95,
            "audio": round(0.86 - storm * 0.08, 3),
        },
        "controller_mode": "demo",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "demo": True,
    }


def _make_demo_jpeg(side: str, t: float, w: int = 320, h: int = 200) -> bytes:
    """Generate a simple animated JPEG without requiring OpenCV at import time."""
    try:
        import numpy as np
        import cv2
    except ImportError:
        # fall back to tiny placeholder
        from backend.api.views.preview_views import _tiny_placeholder_jpeg
        return _tiny_placeholder_jpeg()

    yy, xx = np.mgrid[0:h, 0:w]
    phase = t * 2.5
    wave = (np.sin(xx * 0.04 + phase) * 40 + np.sin(yy * 0.07 - phase * 0.7) * 30).astype(np.float32)
    base = (90 + wave).clip(0, 255).astype(np.uint8)
    img = np.stack([base, base, base], axis=-1)

    if side == "raw":
        img[:, :, 0] = (img[:, :, 0].astype(np.int16) + 30).clip(0, 255).astype(np.uint8)
        img[:, :, 2] = (img[:, :, 2].astype(np.int16) - 10).clip(0, 255).astype(np.uint8)
    elif side == "reconstructed":
        # softer / slightly blurrier recon
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img[:, :, 1] = (img[:, :, 1].astype(np.int16) + 25).clip(0, 255).astype(np.uint8)
    else:  # heatmap
        diff = (np.abs(np.sin(xx * 0.08 + phase)) * 255).astype(np.uint8)
        heat = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        img = cv2.addWeighted(img, 0.45, heat, 0.55, 0)

    # label strip
    cv2.putText(
        img,
        f"LYDLR {side.upper()}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if not ok:
        from backend.api.views.preview_views import _tiny_placeholder_jpeg
        return _tiny_placeholder_jpeg()
    return buf.tobytes()


class DemoPulseView(AsyncAPIView):
    """
    POST /api/demo/pulse/ — inject one round of synthetic metrics + preview frames.
    GET  /api/demo/pulse/ — same, convenient for browsers/scripts.
    """

    async def _pulse(self, request):
        if not _demo_enabled():
            return JsonResponse(
                {"detail": "Demo metrics disabled (LYDLR_DEMO_METRICS=0)"},
                status=status.HTTP_403_FORBIDDEN,
            )

        db = await ensure_db_connection()
        body = {}
        if hasattr(request, "data") and isinstance(request.data, dict):
            body = request.data
        node_ids: List[str] = body.get("node_ids") or []
        if not node_ids:
            cursor = db.nodes.find({}).limit(8)
            docs = await cursor.to_list(8)
            node_ids = [d.get("node_id") for d in docs if d.get("node_id")]
        if not node_ids:
            node_ids = ["node_0", "node_1", "iot_gateway_01"]

        t = time.time()
        published = []
        for nid in node_ids:
            vertical = "iot" if "iot" in nid else "drone"
            doc = _synth_metric(nid, t, vertical)
            await db.metrics.insert_one(dict(doc))
            await publish_message("metrics_update", doc)
            for side in PREVIEW_SIDES:
                jpeg = await asyncio.to_thread(_make_demo_jpeg, side, t + hash(nid) % 10)
                await store_preview_frame(nid, side, jpeg)
            published.append(nid)

        return Response(
            {
                "ok": True,
                "nodes": published,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def get(self, request):
        return await self._pulse(request)

    async def post(self, request):
        return await self._pulse(request)


class DemoAutopilotView(AsyncAPIView):
    """
    GET /api/demo/status/ — whether demo mode is enabled.
    """

    async def get(self, request):
        return Response({"demo_metrics_enabled": _demo_enabled()})
