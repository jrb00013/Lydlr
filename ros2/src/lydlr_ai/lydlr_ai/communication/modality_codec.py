"""
Multimodal encoding helpers for edge compression and stream framing.

- Audio → mel spectrogram features
- LiDAR voxel downsampling by compression level
- IMU delta encoding for compact uplink
- LYMS framed multiplex for async_stream_splitter
"""
from __future__ import annotations

import struct
from typing import Dict, Optional, Tuple

import numpy as np

MODALITY_ORDER = ("camera", "lidar", "imu", "audio")
LYMS_MAGIC = b"LYMS"
LYMS_VERSION = 1
_HEADER = struct.Struct("<4sBH")  # magic, version, num_chunks
_CHUNK_HEAD = struct.Struct("<HII")  # name_len, data_len, flags


def audio_to_mel(
    waveform: np.ndarray,
    *,
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 512,
    hop_length: int = 256,
) -> np.ndarray:
    """Convert 1-D waveform to flattened mel spectrogram (float32)."""
    wave = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if wave.size == 0:
        return np.zeros(n_mels * n_mels, dtype=np.float32)

    try:
        import librosa

        mel = librosa.feature.melspectrogram(
            y=wave,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        flat = mel_db.astype(np.float32).reshape(-1)
        target = n_mels * n_mels
        if flat.size >= target:
            return flat[:target]
        out = np.zeros(target, dtype=np.float32)
        out[: flat.size] = flat
        return out
    except ImportError:
        # Lightweight fallback: block energy bins (always n_mels² features)
        bins = n_mels * n_mels
        chunk = max(1, wave.size // bins)
        energies = [
            float(np.sqrt(np.mean(wave[i : i + chunk] ** 2) + 1e-8))
            for i in range(0, wave.size, chunk)
        ]
        out = np.zeros(bins, dtype=np.float32)
        n = min(len(energies), bins)
        out[:n] = np.array(energies[:n], dtype=np.float32)
        return out


def downsample_lidar(points: np.ndarray, compression_level: float) -> np.ndarray:
    """Reduce LiDAR point count as compression becomes more aggressive."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1)
    if pts.size == 0:
        return pts

    level = float(np.clip(compression_level, 0.1, 0.98))
    keep_ratio = max(0.12, 1.0 - level * 0.75)
    target = max(4, int(len(pts) * keep_ratio))
    if target >= len(pts):
        return pts

    stride = max(1, len(pts) // target)
    return pts[::stride][:target]


def encode_imu_delta(prev: Optional[np.ndarray], curr: np.ndarray) -> Tuple[bytes, np.ndarray]:
    """Pack IMU as delta from previous sample (6 floats). Returns (bytes, curr)."""
    curr_v = np.asarray(curr, dtype=np.float32).reshape(-1)[:6]
    if prev is None:
        payload = curr_v.tobytes()
    else:
        prev_v = np.asarray(prev, dtype=np.float32).reshape(-1)[:6]
        delta = curr_v - prev_v
        payload = delta.tobytes()
    return payload, curr_v


def decode_imu_delta(prev: Optional[np.ndarray], delta_bytes: bytes) -> np.ndarray:
    delta = np.frombuffer(delta_bytes, dtype=np.float32)
    if prev is None:
        return delta.reshape(-1)[:6]
    prev_v = np.asarray(prev, dtype=np.float32).reshape(-1)[:6]
    return (prev_v + delta).astype(np.float32)


def frame_multimodal_payload(chunks: Dict[str, bytes]) -> bytes:
    """Pack modality chunks into a single LYMS frame."""
    ordered_names = [m for m in MODALITY_ORDER if m in chunks and chunks[m]]
    extras = [k for k in chunks if k not in MODALITY_ORDER and chunks[k]]
    ordered = [(m, chunks[m]) for m in ordered_names + extras]
    buf = bytearray(_HEADER.pack(LYMS_MAGIC, LYMS_VERSION, len(ordered)))
    for name, data in ordered:
        name_b = name.encode("utf-8")
        buf.extend(_CHUNK_HEAD.pack(len(name_b), len(data), 0))
        buf.extend(name_b)
        buf.extend(data)
    return bytes(buf)


def split_multimodal_payload(data: bytes) -> Dict[str, bytes]:
    """Unpack LYMS frame into modality chunks."""
    if len(data) < _HEADER.size:
        raise ValueError("truncated LYMS frame")
    magic, version, n_chunks = _HEADER.unpack_from(data, 0)
    if magic != LYMS_MAGIC:
        raise ValueError(f"bad LYMS magic {magic!r}")
    if version != LYMS_VERSION:
        raise ValueError(f"unsupported LYMS version {version}")

    out: Dict[str, bytes] = {}
    off = _HEADER.size
    for _ in range(n_chunks):
        if off + _CHUNK_HEAD.size > len(data):
            raise ValueError("truncated LYMS chunk header")
        name_len, data_len, _flags = _CHUNK_HEAD.unpack_from(data, off)
        off += _CHUNK_HEAD.size
        name = data[off : off + name_len].decode("utf-8")
        off += name_len
        out[name] = data[off : off + data_len]
        off += data_len
    return out
