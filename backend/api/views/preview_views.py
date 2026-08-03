"""
Live preview frame ingest + MJPEG streaming for Visual Monitoring.
"""
import asyncio
import base64
import time
from typing import Optional

from django.http import JsonResponse, StreamingHttpResponse
from rest_framework import status

from backend.api.connections import redis_client
from backend.api.views.base import AsyncAPIView, ensure_db_connection
from backend.api.redis_pubsub import publish_message


PREVIEW_SIDES = ("raw", "reconstructed", "heatmap")
PREVIEW_TTL_SEC = 8

# Valid 1x1 dark JPEG (base64)
_PLACEHOLDER_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a"
    "HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIy"
    "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIA"
    "AhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEB"
    "AQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAA8A/9k="
)


def _redis_key(node_id: str, side: str) -> str:
    return f"preview:{node_id}:{side}"


def _tiny_placeholder_jpeg() -> bytes:
    return base64.b64decode(_PLACEHOLDER_B64)


async def store_preview_frame(node_id: str, side: str, jpeg_bytes: bytes) -> bool:
    if not redis_client or side not in PREVIEW_SIDES:
        return False
    key = _redis_key(node_id, side)
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    await redis_client.set(key, b64, ex=PREVIEW_TTL_SEC)
    await redis_client.set(f"{key}:ts", str(time.time()), ex=PREVIEW_TTL_SEC)
    return True


async def load_preview_frame(node_id: str, side: str) -> Optional[bytes]:
    if not redis_client or side not in PREVIEW_SIDES:
        return None
    b64 = await redis_client.get(_redis_key(node_id, side))
    if not b64:
        return None
    try:
        return base64.b64decode(b64)
    except Exception:
        return None


class NodePreviewView(AsyncAPIView):
    """POST JPEG preview frames from edge nodes."""

    async def post(self, request, node_id: str):
        await ensure_db_connection()
        body = request.data if hasattr(request, "data") else {}
        if not isinstance(body, dict):
            try:
                import json

                body = json.loads(request.body.decode("utf-8"))
            except Exception:
                body = {}

        side = str(body.get("side", "reconstructed")).lower()
        if side not in PREVIEW_SIDES:
            return JsonResponse(
                {"detail": f"side must be one of {PREVIEW_SIDES}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        data_b64 = body.get("data_b64")
        if not data_b64:
            return JsonResponse(
                {"detail": "data_b64 required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            jpeg_bytes = base64.b64decode(data_b64)
        except Exception:
            return JsonResponse(
                {"detail": "invalid base64"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if len(jpeg_bytes) < 20 or len(jpeg_bytes) > 2_000_000:
            return JsonResponse(
                {"detail": "jpeg size out of range"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        ok = await store_preview_frame(node_id, side, jpeg_bytes)
        if not ok:
            return JsonResponse(
                {"detail": "redis unavailable"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        await publish_message(
            "preview_update",
            {"node_id": node_id, "side": side, "bytes": len(jpeg_bytes)},
        )
        return JsonResponse({"ok": True, "node_id": node_id, "side": side})


class NodePreviewLatestView(AsyncAPIView):
    """GET single latest JPEG (easier for <img> refresh polling than MJPEG)."""

    async def get(self, request, node_id: str):
        side = request.GET.get("side", "reconstructed").lower()
        if side not in PREVIEW_SIDES:
            return JsonResponse(
                {"detail": f"side must be one of {PREVIEW_SIDES}"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        frame = await load_preview_frame(node_id, side)
        if not frame:
            frame = _tiny_placeholder_jpeg()
        from django.http import HttpResponse

        resp = HttpResponse(frame, content_type="image/jpeg")
        resp["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp["X-Lydlr-Preview-Side"] = side
        return resp


class NodePreviewMjpegView(AsyncAPIView):
    """GET multipart MJPEG stream of latest Redis preview frame."""

    async def get(self, request, node_id: str):
        side = request.GET.get("side", "reconstructed").lower()
        if side not in PREVIEW_SIDES:
            return JsonResponse(
                {"detail": f"side must be one of {PREVIEW_SIDES}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        boundary = b"lydlrframe"

        async def frame_generator():
            placeholder = _tiny_placeholder_jpeg()
            while True:
                frame = await load_preview_frame(node_id, side)
                if not frame:
                    frame = placeholder
                yield (
                    b"--" + boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
                await asyncio.sleep(0.2)

        response = StreamingHttpResponse(
            frame_generator(),
            content_type=f"multipart/x-mixed-replace; boundary={boundary.decode()}",
        )
        response["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response["Pragma"] = "no-cache"
        return response


class NodeTopicsView(AsyncAPIView):
    """Return known LYDT / preview topics for a node."""

    async def get(self, request, node_id: str):
        await ensure_db_connection()
        topics = [
            {"name": f"/lydlr/{node_id}/transport/compressed", "type": "compressed", "node": node_id},
            {"name": f"/lydlr/{node_id}/transport/metrics", "type": "metrics", "node": node_id},
            {"name": f"/lydlr/{node_id}/coordination", "type": "coordination", "node": node_id},
            {"name": f"/lydlr/{node_id}/heartbeat", "type": "heartbeat", "node": node_id},
            {"name": f"/lydlr/{node_id}/preview/raw", "type": "preview", "node": node_id},
            {"name": f"/lydlr/{node_id}/preview/reconstructed", "type": "preview", "node": node_id},
            {"name": f"/lydlr/{node_id}/preview/heatmap", "type": "preview", "node": node_id},
            {"name": f"/{node_id}/compressed", "type": "compressed", "node": node_id},
            {"name": f"/{node_id}/metrics", "type": "metrics", "node": node_id},
        ]

        last_seen = {}
        if redis_client:
            for side in PREVIEW_SIDES:
                ts = await redis_client.get(f"{_redis_key(node_id, side)}:ts")
                if ts:
                    last_seen[side] = float(ts)
                    for t in topics:
                        if t["name"].endswith(f"/preview/{side}"):
                            t["last_seen"] = float(ts)
                            t["live"] = (time.time() - float(ts)) < PREVIEW_TTL_SEC

        return JsonResponse({"node_id": node_id, "topics": topics, "preview_last_seen": last_seen})
