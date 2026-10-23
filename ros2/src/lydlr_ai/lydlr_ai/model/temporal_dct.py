# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
"""Vision codec: temporal residual + 8x8 DCT + countable rANS bits.

Image-only clips have no LiDAR/IMU/audio, so the multimodal network cannot
be the rate path. This coder is the Lydlr vision operating point: predict
from the previous reconstruction, transform the residual, quantize, and
entropy-code the symbols. Rate is the length of the self-describing payload.
"""

from __future__ import annotations

import struct
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from lydlr_ai.model.entropy_coder import rans_decode, rans_encode

MAGIC = b"LDCT"
VERSION = 1
_HEAD = struct.Struct("<4sBBfHHH")  # magic, ver, pad, qstep, n, h, w
_BLOCK = 8

_K = np.arange(8, dtype=np.float64)[:, None]
_I = np.arange(8, dtype=np.float64)[None, :]
_M = np.cos(np.pi * (2 * _I + 1) * _K / 16.0)
_M[0] *= np.sqrt(1.0 / 8.0)
_M[1:] *= np.sqrt(2.0 / 8.0)
_M = _M.astype(np.float32)
_MT = np.ascontiguousarray(_M.T)


def _as_frames(frames: Sequence[np.ndarray]) -> List[np.ndarray]:
    arr = np.asarray(frames)
    if arr.ndim == 4:
        return [np.asarray(arr[i], dtype=np.uint8) for i in range(arr.shape[0])]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return [np.asarray(arr, dtype=np.uint8)]
    raise ValueError(f"expected NxHxWx3 frames, got {arr.shape}")


def _pad8(frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
    h, w = frame.shape[:2]
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    if ph == 0 and pw == 0:
        return frame, h, w
    out = np.pad(frame, ((0, ph), (0, pw), (0, 0)), mode="edge")
    return out, h, w


def _blocks(ch: np.ndarray) -> np.ndarray:
    h, w = ch.shape
    return (
        ch.reshape(h // 8, 8, w // 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(-1, 8, 8)
    )


def _unblocks(blocks: np.ndarray, h: int, w: int) -> np.ndarray:
    nby, nbx = h // 8, w // 8
    return (
        blocks.reshape(nby, nbx, 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(h, w)
    )


def _dct_image(img: np.ndarray) -> np.ndarray:
    out = np.empty_like(img, dtype=np.float32)
    for c in range(3):
        b = _blocks(img[:, :, c])
        d = _M @ b @ _MT
        out[:, :, c] = _unblocks(d, img.shape[0], img.shape[1])
    return out


def _idct_image(coef: np.ndarray) -> np.ndarray:
    out = np.empty_like(coef, dtype=np.float32)
    for c in range(3):
        b = _blocks(coef[:, :, c])
        d = _MT @ b @ _M
        out[:, :, c] = _unblocks(d, coef.shape[0], coef.shape[1])
    return out


def _pmf_u16(symbols: np.ndarray) -> np.ndarray:
    counts = np.bincount(symbols.reshape(-1), minlength=256).astype(np.int64)
    total = int(counts.sum())
    u = np.zeros(256, dtype=np.int64)
    if total <= 0:
        u[:] = 1
    else:
        observed = counts > 0
        u[observed] = np.maximum(1, np.rint(counts[observed] * 65535.0 / total).astype(np.int64))
    drift = int(u.sum()) - 65535
    if drift > 0:
        order = np.argsort(-u)
        for i in order:
            take = min(drift, int(u[i] - (1 if counts[i] > 0 else 0)))
            if take > 0:
                u[i] -= take
                drift -= take
            if drift == 0:
                break
    elif drift < 0:
        u[int(np.argmax(counts))] += -drift
    return u.astype(np.uint16)


def _pmf_from_u16(u: np.ndarray) -> np.ndarray:
    p = np.asarray(u, dtype=np.float64)
    s = float(p.sum())
    if s <= 0:
        return np.full(256, 1.0 / 256.0)
    return p / s


def encode_temporal_dct(
    frames: Sequence[np.ndarray],
    *,
    qstep: float = 16.0,
) -> Tuple[bytes, List[np.ndarray], dict]:
    """Encode RGB frames. Returns payload, reconstructions, rate stats."""
    flist = _as_frames(frames)
    if not flist:
        raise ValueError("encode_temporal_dct: empty frame list")
    qstep = float(qstep)
    if qstep <= 0:
        raise ValueError("qstep must be positive")

    orig_h, orig_w = flist[0].shape[:2]
    padded = []
    for f in flist:
        p, _, _ = _pad8(np.asarray(f, dtype=np.uint8))
        padded.append(p)
    H, W = padded[0].shape[:2]

    prev = None
    recons_full: List[np.ndarray] = []
    sym_chunks: List[np.ndarray] = []
    for frame in padded:
        x = frame.astype(np.float32)
        pred = prev if prev is not None else np.full_like(x, 128.0)
        coef = _dct_image(x - pred)
        q = np.clip(np.rint(coef / qstep), -127, 127).astype(np.int16)
        sym_chunks.append((q + 127).astype(np.uint8))
        recon = np.clip(pred + _idct_image(q.astype(np.float32) * qstep), 0, 255)
        prev = recon
        recons_full.append(recon)

    symbols = np.concatenate([s.reshape(-1) for s in sym_chunks]).astype(np.uint8)
    u16 = _pmf_u16(symbols)
    pmf = _pmf_from_u16(u16)
    coded = rans_encode(symbols, pmf)
    head = _HEAD.pack(MAGIC, VERSION, 0, qstep, len(flist), H, W)
    payload = head + u16.tobytes() + coded

    recons = [r[:orig_h, :orig_w].astype(np.uint8) for r in recons_full]
    pixels = orig_h * orig_w
    total_bits = len(payload) * 8
    stats = {
        "bpp": float(total_bits / (len(flist) * pixels)),
        "bits_per_frame": float(total_bits / len(flist)),
        "total_bits": float(total_bits),
        "payload_bytes": len(payload),
        "n_frames": len(flist),
        "qstep": qstep,
        "rate_source": "rans_bits",
        "codec_path": "temporal_dct_rans",
    }
    return payload, recons, stats


def decode_temporal_dct(payload: bytes) -> List[np.ndarray]:
    """Decode a payload from ``encode_temporal_dct``."""
    if len(payload) < _HEAD.size + 512:
        raise ValueError("truncated LDCT payload")
    magic, ver, _pad, qstep, n, h, w = _HEAD.unpack_from(payload, 0)
    if magic != MAGIC:
        raise ValueError(f"bad LDCT magic {magic!r}")
    if ver != VERSION:
        raise ValueError(f"unsupported LDCT version {ver}")
    off = _HEAD.size
    u16 = np.frombuffer(payload[off : off + 512], dtype="<u2").copy()
    off += 512
    pmf = _pmf_from_u16(u16)
    n_sym = int(n) * int(h) * int(w) * 3
    symbols = rans_decode(payload[off:], n_sym, pmf).reshape(n, h, w, 3)
    prev = None
    recons: List[np.ndarray] = []
    for i in range(n):
        q = symbols[i].astype(np.int16) - 127
        pred = prev if prev is not None else np.full((h, w, 3), 128.0, dtype=np.float32)
        recon = np.clip(pred + _idct_image(q.astype(np.float32) * float(qstep)), 0, 255)
        prev = recon
        recons.append(recon.astype(np.uint8))
    return recons


def sweep_temporal_dct(
    frames: Sequence[np.ndarray],
    qsteps: Iterable[float] = (4, 8, 12, 16, 24, 32, 48, 64, 96),
) -> List[dict]:
    """Operating points with countable bpp and reconstruction arrays."""
    points = []
    for q in qsteps:
        payload, recons, stats = encode_temporal_dct(frames, qstep=float(q))
        points.append({"payload": payload, "recons": recons, **stats})
    return points
