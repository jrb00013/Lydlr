"""
Lydlr Transport Wire Protocol v1 (LYDT)

Packed binary over std_msgs/UInt8MultiArray.data for:
- Framed multimodal compression payloads
- Rich metrics (bytes in/out, vertical, model version)
- Coordination signals from fleet coordinator

Layout (little-endian):
  magic      4s   b'LYDT'
  version    B    1
  msg_type   B    see MSG_*
  flags      H
  seq        I
  ts_sec     d
  node_len   H    + node_id utf-8
  meta_len   H    + meta_json utf-8
  payload_len I  + payload bytes
"""
from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

MAGIC = b"LYDT"
VERSION = 1

MSG_COMPRESSED = 1
MSG_METRICS = 2
MSG_COORDINATION = 3
MSG_HEARTBEAT = 4
MSG_DEPLOY_ACK = 5

_HEADER = struct.Struct("<4sBBHId")  # magic, ver, type, flags, seq, ts
_TAIL = struct.Struct("<HHI")  # node_len, meta_len, payload_len


@dataclass
class MetricsPayload:
    node_id: str
    vertical: str = ""
    model_version: str = ""
    compression_ratio: float = 0.0
    latency_ms: float = 0.0
    compression_level: float = 0.0
    quality_score: float = 0.0
    bandwidth_estimate: float = 1.0
    bytes_in: int = 0
    bytes_out: int = 0

    def to_legacy_floats(self):
        return [
            self.compression_ratio,
            self.latency_ms,
            self.compression_level,
            self.quality_score,
            self.bandwidth_estimate,
        ]


@dataclass
class CoordinationPayload:
    target_compression: float
    allocated_mbps: float
    fleet_avg_compression: float
    fleet_avg_latency_ms: float
    fleet_avg_quality: float


@dataclass
class CompressedPayload:
    node_id: str
    model_version: str
    vertical: str
    seq: int
    bytes_in: int
    bytes_out: int
    compression_ratio: float
    payload: bytes  # zlib compressed tensor blob


def _pack(
    msg_type: int,
    node_id: str,
    meta: Dict[str, Any],
    payload: bytes,
    seq: int = 0,
    flags: int = 0,
    ts: float = 0.0,
) -> bytes:
    import time
    if not ts:
        ts = time.time()
    node_b = node_id.encode("utf-8")
    meta_b = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    header = _HEADER.pack(MAGIC, VERSION, msg_type, flags, seq, ts)
    tail = _TAIL.pack(len(node_b), len(meta_b), len(payload))
    return header + tail + node_b + meta_b + payload


def _unpack(data: bytes) -> Tuple[int, str, Dict[str, Any], bytes, int, float]:
    if len(data) < _HEADER.size + _TAIL.size:
        raise ValueError("truncated LYDT frame")
    magic, ver, msg_type, flags, seq, ts = _HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if ver != VERSION:
        raise ValueError(f"unsupported version {ver}")
    node_len, meta_len, payload_len = _TAIL.unpack_from(data, _HEADER.size)
    off = _HEADER.size + _TAIL.size
    node_id = data[off : off + node_len].decode("utf-8")
    off += node_len
    meta = json.loads(data[off : off + meta_len].decode("utf-8") or "{}")
    off += meta_len
    payload = data[off : off + payload_len]
    return msg_type, node_id, meta, payload, seq, ts


def encode_metrics(m: MetricsPayload, seq: int = 0) -> bytes:
    meta = {
        "vertical": m.vertical,
        "model_version": m.model_version,
        "compression_ratio": m.compression_ratio,
        "latency_ms": m.latency_ms,
        "compression_level": m.compression_level,
        "quality_score": m.quality_score,
        "bandwidth_estimate": m.bandwidth_estimate,
        "bytes_in": m.bytes_in,
        "bytes_out": m.bytes_out,
    }
    return _pack(MSG_METRICS, m.node_id, meta, b"", seq=seq)


def decode_metrics(data: bytes) -> MetricsPayload:
    msg_type, node_id, meta, _, _, _ = _unpack(data)
    if msg_type != MSG_METRICS:
        raise ValueError(f"expected metrics, got {msg_type}")
    return MetricsPayload(
        node_id=node_id,
        vertical=meta.get("vertical", ""),
        model_version=meta.get("model_version", ""),
        compression_ratio=float(meta.get("compression_ratio", 0)),
        latency_ms=float(meta.get("latency_ms", 0)),
        compression_level=float(meta.get("compression_level", 0)),
        quality_score=float(meta.get("quality_score", 0)),
        bandwidth_estimate=float(meta.get("bandwidth_estimate", 1)),
        bytes_in=int(meta.get("bytes_in", 0)),
        bytes_out=int(meta.get("bytes_out", 0)),
    )


def encode_compressed(c: CompressedPayload, raw_tensor: bytes) -> bytes:
    payload = zlib.compress(raw_tensor, level=6)
    meta = {
        "model_version": c.model_version,
        "vertical": c.vertical,
        "bytes_in": c.bytes_in,
        "bytes_out": len(payload),
        "compression_ratio": c.compression_ratio,
    }
    return _pack(MSG_COMPRESSED, c.node_id, meta, payload, seq=c.seq)


def decode_compressed(data: bytes) -> CompressedPayload:
    msg_type, node_id, meta, payload, seq, _ = _unpack(data)
    if msg_type != MSG_COMPRESSED:
        raise ValueError(f"expected compressed, got {msg_type}")
    return CompressedPayload(
        node_id=node_id,
        model_version=meta.get("model_version", ""),
        vertical=meta.get("vertical", ""),
        seq=seq,
        bytes_in=int(meta.get("bytes_in", 0)),
        bytes_out=int(meta.get("bytes_out", len(payload))),
        compression_ratio=float(meta.get("compression_ratio", 0)),
        payload=payload,
    )


def decompress_tensor(frame: CompressedPayload) -> bytes:
    return zlib.decompress(frame.payload)


def encode_coordination(node_id: str, c: CoordinationPayload, seq: int = 0) -> bytes:
    return _pack(MSG_COORDINATION, node_id, asdict(c), b"", seq=seq)


def decode_coordination(data: bytes) -> Tuple[str, CoordinationPayload]:
    msg_type, node_id, meta, _, _, _ = _unpack(data)
    if msg_type != MSG_COORDINATION:
        raise ValueError(f"expected coordination, got {msg_type}")
    return node_id, CoordinationPayload(**meta)


def encode_heartbeat(node_id: str, vertical: str, model_version: str) -> bytes:
    return _pack(
        MSG_HEARTBEAT,
        node_id,
        {"vertical": vertical, "model_version": model_version},
        b"",
    )


def decode_heartbeat(data: bytes) -> Tuple[str, Dict[str, Any]]:
    msg_type, node_id, meta, _, _, _ = _unpack(data)
    if msg_type != MSG_HEARTBEAT:
        raise ValueError(f"expected heartbeat, got {msg_type}")
    return node_id, meta


def to_uint8_array_bytes(frame: bytes) -> list:
    return list(frame)


def from_uint8_array(msg_data) -> bytes:
    return bytes(msg_data)
