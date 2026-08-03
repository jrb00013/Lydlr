"""Pure helpers for Redis pub/sub payload shapes (no Redis/Django imports)."""
from typing import Any, Dict


def unwrap_pubsub_payload(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """
    publish_message wraps payloads as {type, data, timestamp}.
    Return the inner metric/event document for WebSocket clients.
    """
    if not isinstance(envelope, dict):
        return {}
    inner = envelope.get("data")
    if isinstance(inner, dict) and (
        "compression_ratio" in inner
        or "node_id" in inner
        or "status" in inner
        or "event" in inner
    ):
        return inner
    if isinstance(inner, dict) and "data" in inner and isinstance(inner["data"], dict):
        return inner["data"]
    return inner if isinstance(inner, dict) else envelope
