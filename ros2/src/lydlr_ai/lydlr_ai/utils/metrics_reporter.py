"""
Push edge compression metrics to the Lydlr control-plane API.
Used by ROS2 edge nodes (host or Docker + host networking).
"""
import json
import logging
import os
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_API = "http://127.0.0.1:8000/api/metrics/"


def metrics_api_url() -> str:
    base = os.getenv("METRICS_API_URL", os.getenv("LYDLR_API_URL", _DEFAULT_API))
    if base.endswith("/metrics/"):
        return base
    if base.endswith("/metrics"):
        return f"{base}/"
    if base.endswith("/api"):
        return f"{base}/metrics/"
    if base.rstrip("/").endswith("8000"):
        return f"{base.rstrip('/')}/api/metrics/"
    return _DEFAULT_API


def report_metrics(
    node_id: str,
    compression_ratio: float,
    latency_ms: float,
    quality_score: float,
    bandwidth_estimate: float = 1.0,
    compression_level: float = 0.8,
    vertical: Optional[str] = None,
    async_send: bool = True,
) -> None:
    """POST a metrics sample to the backend."""
    payload: Dict[str, Any] = {
        "node_id": node_id,
        "compression_ratio": float(compression_ratio),
        "latency_ms": float(latency_ms),
        "quality_score": float(quality_score),
        "bandwidth_estimate": float(bandwidth_estimate),
        "compression_level": float(compression_level),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if vertical:
        payload["vertical"] = vertical

    if async_send:
        threading.Thread(
            target=_post_payload,
            args=(payload,),
            daemon=True,
        ).start()
    else:
        _post_payload(payload)


def _post_payload(payload: Dict[str, Any]) -> None:
    url = metrics_api_url()
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status != 200:
                logger.warning("Metrics API returned %s for %s", resp.status, payload.get("node_id"))
    except urllib.error.URLError as exc:
        logger.debug("Metrics POST failed for %s: %s", payload.get("node_id"), exc)
    except Exception as exc:
        logger.debug("Metrics POST error for %s: %s", payload.get("node_id"), exc)
