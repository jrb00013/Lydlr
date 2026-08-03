"""
Push JPEG preview thumbnails from edge nodes to the control-plane API.
"""
import base64
import json
import logging
import os
import threading
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_API = "http://127.0.0.1:8000/api"


def preview_api_base() -> str:
    base = os.getenv("LYDLR_API_URL", os.getenv("METRICS_API_URL", _DEFAULT_API))
    if "/api/metrics" in base:
        base = base.split("/api/metrics")[0] + "/api"
    elif base.rstrip("/").endswith("8000"):
        base = f"{base.rstrip('/')}/api"
    elif not base.rstrip("/").endswith("/api"):
        if "/api" not in base:
            base = f"{base.rstrip('/')}/api"
    return base.rstrip("/")


def report_preview(
    node_id: str,
    side: str,
    jpeg_bytes: bytes,
    async_send: bool = True,
) -> None:
    """POST a JPEG thumbnail (base64 JSON) for MJPEG gateway."""
    if not jpeg_bytes:
        return
    payload = {
        "side": side,
        "content_type": "image/jpeg",
        "data_b64": base64.b64encode(jpeg_bytes).decode("ascii"),
    }

    def _post():
        url = f"{preview_api_base()}/nodes/{node_id}/preview/"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status not in (200, 201, 204):
                    logger.warning("Preview API returned %s for %s", resp.status, node_id)
        except urllib.error.URLError as exc:
            logger.debug("Preview POST failed for %s: %s", node_id, exc)
        except Exception as exc:
            logger.debug("Preview POST error for %s: %s", node_id, exc)

    if async_send:
        threading.Thread(target=_post, daemon=True).start()
    else:
        _post()
