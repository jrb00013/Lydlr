"""Deploy-ack LYDT message round-trip."""

from __future__ import annotations

from lydlr_ai.communication.wire import (
    MSG_DEPLOY_ACK,
    decode_deploy_ack,
    encode_deploy_ack,
    _unpack,
)


def test_deploy_ack_round_trip():
    frame = encode_deploy_ack(
        "drone_01",
        model_version="edge-b",
        status="ok",
        detail="loaded",
        seq=9,
    )
    msg_type, *_ = _unpack(frame)
    assert msg_type == MSG_DEPLOY_ACK
    node_id, meta = decode_deploy_ack(frame)
    assert node_id == "drone_01"
    assert meta["model_version"] == "edge-b"
    assert meta["status"] == "ok"
    assert meta["detail"] == "loaded"
