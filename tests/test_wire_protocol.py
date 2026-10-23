"""Round-trip tests for the LYDT wire protocol."""

from __future__ import annotations

import zlib

import pytest

from lydlr_ai.communication.wire import (
    MSG_COMPRESSED,
    MSG_HEARTBEAT,
    MSG_METRICS,
    CompressedPayload,
    CoordinationPayload,
    MetricsPayload,
    decode_compressed,
    decode_coordination,
    decode_heartbeat,
    decode_metrics,
    decompress_tensor,
    encode_compressed,
    encode_coordination,
    encode_heartbeat,
    encode_metrics,
    from_uint8_array,
    to_uint8_array_bytes,
    _unpack,
)


def test_metrics_round_trip():
    m = MetricsPayload(
        node_id="drone_01",
        vertical="drone",
        model_version="v1.2.3",
        compression_ratio=4.5,
        latency_ms=12.3,
        compression_level=0.8,
        quality_score=0.91,
        bandwidth_estimate=0.55,
        bytes_in=10000,
        bytes_out=2200,
        modality_bytes_in={"camera": 8000, "imu": 2000},
        modality_bytes_out={"camera": 1800, "imu": 400},
        controller_mode="rl",
        rl_action=0.7,
        rl_reward=1.2,
    )
    frame = encode_metrics(m, seq=7)
    back = decode_metrics(frame)
    assert back.node_id == "drone_01"
    assert back.vertical == "drone"
    assert back.compression_ratio == pytest.approx(4.5)
    assert back.bytes_in == 10000
    assert back.modality_bytes_out["camera"] == 1800
    assert back.controller_mode == "rl"
    msg_type, *_ = _unpack(frame)
    assert msg_type == MSG_METRICS


def test_compressed_round_trip():
    raw = b"latent-blob-" + bytes(range(64))
    c = CompressedPayload(
        node_id="iot_gw",
        model_version="edge-a",
        vertical="iot",
        seq=3,
        bytes_in=len(raw),
        bytes_out=0,
        compression_ratio=2.0,
        payload=b"",
    )
    frame = encode_compressed(c, raw)
    back = decode_compressed(frame)
    assert back.node_id == "iot_gw"
    assert back.seq == 3
    assert decompress_tensor(back) == raw
    assert zlib.decompress(back.payload) == raw
    msg_type, *_ = _unpack(frame)
    assert msg_type == MSG_COMPRESSED


def test_coordination_and_heartbeat():
    coord = CoordinationPayload(
        target_compression=0.85,
        allocated_mbps=0.4,
        fleet_avg_compression=3.2,
        fleet_avg_latency_ms=18.0,
        fleet_avg_quality=0.88,
    )
    frame = encode_coordination("fleet_hub", coord, seq=1)
    node_id, back = decode_coordination(frame)
    assert node_id == "fleet_hub"
    assert back.target_compression == pytest.approx(0.85)
    assert back.allocated_mbps == pytest.approx(0.4)

    hb = encode_heartbeat("drone_02", "drone", "v9")
    nid, meta = decode_heartbeat(hb)
    assert nid == "drone_02"
    assert meta["model_version"] == "v9"
    msg_type, *_ = _unpack(hb)
    assert msg_type == MSG_HEARTBEAT


def test_uint8_array_helpers():
    payload = encode_heartbeat("n", "drone", "v")
    as_list = to_uint8_array_bytes(payload)
    assert from_uint8_array(as_list) == payload


def test_bad_magic_raises():
    with pytest.raises(ValueError, match="bad magic"):
        decode_metrics(b"XXXX" + b"\x00" * 40)
