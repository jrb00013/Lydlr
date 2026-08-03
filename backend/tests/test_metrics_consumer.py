"""Regression: Redis pub/sub envelopes unwrap to metric docs for WebSocket clients."""
from backend.api.pubsub_utils import unwrap_pubsub_payload


def test_unwrap_pubsub_payload_inner_metric():
    envelope = {
        "type": "metrics_update",
        "data": {
            "node_id": "node_0",
            "compression_ratio": 4.2,
            "latency_ms": 12.5,
            "quality_score": 0.91,
        },
        "timestamp": "1.0",
    }
    metric = unwrap_pubsub_payload(envelope)
    assert metric["node_id"] == "node_0"
    assert metric["compression_ratio"] == 4.2
    assert metric["latency_ms"] == 12.5


def test_unwrap_double_wrapped_envelope():
    outer = {
        "type": "metrics_update",
        "data": {
            "type": "metrics_update",
            "data": {
                "node_id": "iot_gateway_01",
                "compression_ratio": 2.0,
            },
            "timestamp": "2.0",
        },
    }
    metric = unwrap_pubsub_payload(outer)
    assert metric["node_id"] == "iot_gateway_01"
    assert metric["compression_ratio"] == 2.0


def test_unwrap_passthrough_already_flat():
    flat = {"node_id": "node_1", "compression_ratio": 3.0, "status": "ok"}
    assert unwrap_pubsub_payload(flat)["compression_ratio"] == 3.0
