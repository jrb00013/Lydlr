# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
"""Vision codec: temporal residual + 8x8 DCT + countable rANS bits.

Image-only clips have no LiDAR/IMU/audio, so the multimodal network cannot
be the rate path. This coder is the Lydlr vision operating point: predict
from the previous reconstruction, transform the residual in YCbCr, apply a
mild frequency-weighted quantizer, and entropy-code the symbols. Rate is the
length of the self-describing payload.

Version 2 (current encode): YCbCr residual + mild AC emphasis.
Version 1 payloads (RGB, uniform qstep) still decode.
"""

from __future__ import annotations

import struct
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from lydlr_ai.model.entropy_coder import rans_decode, rans_encode

MAGIC = b"LDCT"
VERSION = 2
VERSION_RGB_UNIFORM = 1
_HEAD = struct.Struct("<4sBBfHHH")  # magic, ver, flags, qstep, n, h, w
_BLOCK = 8

_K = np.arange(8, dtype=np.float64)[:, None]
_I = np.arange(8, dtype=np.float64)[None, :]
_M = np.cos(np.pi * (2 * _I + 1) * _K / 16.0)
_M[0] *= np.sqrt(1.0 / 8.0)
_M[1:] *= np.sqrt(2.0 / 8.0)
_M = _M.astype(np.float32)
_MT = np.ascontiguousarray(_M.T)

# Mild frequency weights: keep DC tight, gently coarsen high AC so residual
# zeros concentrate and rANS compresses harder without JPEG-table overkill.
_FY = np.arange(8, dtype=np.float32)
_FX, _FY = np.meshgrid(_FY, _FY)
_FREQ_W = (1.0 + 0.08 * (_FX + _FY)).astype(np.float32)
_CHROMA_Q = 1.15


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


def _rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float32)
    r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    return np.stack([y, cb, cr], axis=2)


def _ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    y = ycbcr[:, :, 0]
    cb = ycbcr[:, :, 1] - 128.0
    cr = ycbcr[:, :, 2] - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.stack([r, g, b], axis=2)


def _quant_volume(h: int, w: int, qstep: float) -> np.ndarray:
    """Tile mild frequency weights; chroma planes slightly coarser."""
    nby, nbx = h // 8, w // 8
    luma = np.tile(_FREQ_W * float(qstep), (nby, nbx))
    chroma = luma * _CHROMA_Q
    return np.stack([luma, chroma, chroma], axis=2).astype(np.float32)


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


def model_confidence(*, zero_frac: float, psnr_db: Optional[float] = None) -> float:
    """Prediction lock + optional recon fidelity → [0, 1] confidence.

    Near-zero residual symbols mean the temporal predictor is locked; PSNR
    (when known) adds fidelity. Both levers rise together on correlated video.
    """
    z = float(np.clip(zero_frac, 0.0, 1.0))
    if psnr_db is None:
        return float(np.clip(0.15 + 0.85 * z, 0.0, 1.0))
    fidelity = float(np.clip(float(psnr_db) / 45.0, 0.0, 1.0))
    return float(np.clip(0.40 * z + 0.60 * fidelity, 0.0, 1.0))


def _confidence_from_symbols(symbols: np.ndarray) -> Tuple[float, float]:
    zero_frac = float(np.mean(np.asarray(symbols, dtype=np.uint8) == 127))
    return zero_frac, model_confidence(zero_frac=zero_frac)


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
    Q = _quant_volume(H, W, qstep)

    prev = None
    recons_full: List[np.ndarray] = []
    sym_chunks: List[np.ndarray] = []
    for frame in padded:
        x = _rgb_to_ycbcr(frame)
        pred = prev if prev is not None else np.full_like(x, 128.0)
        coef = _dct_image(x - pred)
        q = np.clip(np.rint(coef / Q), -127, 127).astype(np.int16)
        sym_chunks.append((q + 127).astype(np.uint8))
        recon_yc = pred + _idct_image(q.astype(np.float32) * Q)
        prev = recon_yc
        recons_full.append(np.clip(_ycbcr_to_rgb(recon_yc), 0, 255))

    symbols = np.concatenate([s.reshape(-1) for s in sym_chunks]).astype(np.uint8)
    u16 = _pmf_u16(symbols)
    pmf = _pmf_from_u16(u16)
    coded = rans_encode(symbols, pmf)
    head = _HEAD.pack(MAGIC, VERSION, 0, qstep, len(flist), H, W)
    payload = head + u16.tobytes() + coded

    recons = [r[:orig_h, :orig_w].astype(np.uint8) for r in recons_full]
    pixels = orig_h * orig_w
    total_bits = len(payload) * 8
    raw_bytes = int(len(flist) * orig_h * orig_w * 3)
    zero_frac, conf = _confidence_from_symbols(symbols)
    stats = {
        "bpp": float(total_bits / (len(flist) * pixels)),
        "bits_per_frame": float(total_bits / len(flist)),
        "total_bits": float(total_bits),
        "payload_bytes": len(payload),
        "bytes_in": raw_bytes,
        "bytes_out": len(payload),
        "compression_ratio": float(raw_bytes) / float(max(len(payload), 1)),
        "n_frames": len(flist),
        "qstep": qstep,
        "zero_frac": zero_frac,
        "confidence": conf,
        "rate_source": "rans_bits",
        "codec_path": "temporal_dct_rans_v2",
        "color_space": "ycbcr",
        "freq_weighted": True,
    }
    return payload, recons, stats


def encode_frame(
    frame: np.ndarray,
    prev: np.ndarray | None = None,
    *,
    qstep: float = 16.0,
) -> Tuple[bytes, np.ndarray, dict]:
    """Encode one RGB frame against the previous reconstruction.

    ``prev`` is the prior reconstruction (or None for an I-frame predicted
    from mid-grey). The payload is a one-frame LDCT blob; its length is the
    countable rate.
    """
    qstep = float(qstep)
    if qstep <= 0:
        raise ValueError("qstep must be positive")
    src = np.asarray(frame, dtype=np.uint8)
    padded, orig_h, orig_w = _pad8(src)
    H, W = padded.shape[:2]
    Q = _quant_volume(H, W, qstep)
    x = _rgb_to_ycbcr(padded)
    if prev is None:
        pred = np.full(x.shape, 128.0, dtype=np.float32)
    else:
        prev_p, _, _ = _pad8(np.asarray(prev, dtype=np.uint8))
        if prev_p.shape != padded.shape:
            pred = np.full(x.shape, 128.0, dtype=np.float32)
        else:
            pred = _rgb_to_ycbcr(prev_p)
    coef = _dct_image(x - pred)
    q = np.clip(np.rint(coef / Q), -127, 127).astype(np.int16)
    symbols = (q + 127).astype(np.uint8).reshape(-1)
    u16 = _pmf_u16(symbols)
    pmf = _pmf_from_u16(u16)
    coded = rans_encode(symbols, pmf)
    head = _HEAD.pack(MAGIC, VERSION, 0, qstep, 1, H, W)
    payload = head + u16.tobytes() + coded
    recon_yc = pred + _idct_image(q.astype(np.float32) * Q)
    recon = np.clip(_ycbcr_to_rgb(recon_yc), 0, 255)[:orig_h, :orig_w].astype(np.uint8)
    raw_bytes = int(src.size)
    zero_frac, conf = _confidence_from_symbols(symbols)
    stats = {
        "bpp": float(len(payload) * 8 / max(orig_h * orig_w, 1)),
        "bytes_in": raw_bytes,
        "bytes_out": len(payload),
        "compression_ratio": float(raw_bytes) / float(max(len(payload), 1)),
        "qstep": qstep,
        "zero_frac": zero_frac,
        "confidence": conf,
        "rate_source": "rans_bits",
        "codec_path": "temporal_dct_rans_v2",
        "color_space": "ycbcr",
        "freq_weighted": True,
        "keyframe": prev is None,
    }
    return payload, recon, stats


def _decode_symbols_to_frames(
    symbols: np.ndarray,
    *,
    n: int,
    h: int,
    w: int,
    qstep: float,
    version: int,
) -> List[np.ndarray]:
    q_frames = symbols.reshape(n, h, w, 3).astype(np.int16) - 127
    prev = None
    recons: List[np.ndarray] = []
    if int(version) >= VERSION:
        Q = _quant_volume(h, w, float(qstep))
        for i in range(n):
            pred = prev if prev is not None else np.full((h, w, 3), 128.0, dtype=np.float32)
            recon_yc = pred + _idct_image(q_frames[i].astype(np.float32) * Q)
            prev = recon_yc
            recons.append(np.clip(_ycbcr_to_rgb(recon_yc), 0, 255).astype(np.uint8))
        return recons
    # v1: RGB residual, uniform qstep
    for i in range(n):
        pred = prev if prev is not None else np.full((h, w, 3), 128.0, dtype=np.float32)
        recon = np.clip(pred + _idct_image(q_frames[i].astype(np.float32) * float(qstep)), 0, 255)
        prev = recon
        recons.append(recon.astype(np.uint8))
    return recons


def decode_frame(payload: bytes, prev: np.ndarray | None = None) -> np.ndarray:
    """Invert ``encode_frame``. ``prev`` must be the same prediction the encoder used."""
    if len(payload) < _HEAD.size + 512:
        raise ValueError("truncated LDCT payload")
    magic, ver, _flags, qstep, n, h, w = _HEAD.unpack_from(payload, 0)
    if magic != MAGIC or int(n) != 1:
        raise ValueError("decode_frame expects a one-frame LDCT payload")
    if int(ver) not in (VERSION_RGB_UNIFORM, VERSION):
        raise ValueError(f"unsupported LDCT version {ver}")
    off = _HEAD.size
    u16 = np.frombuffer(payload[off : off + 512], dtype="<u2").copy()
    symbols = rans_decode(payload[off + 512 :], int(h) * int(w) * 3, _pmf_from_u16(u16))
    q = symbols.reshape(int(h), int(w), 3).astype(np.int16) - 127
    if int(ver) >= VERSION:
        Q = _quant_volume(int(h), int(w), float(qstep))
        if prev is None:
            pred = np.full((int(h), int(w), 3), 128.0, dtype=np.float32)
        else:
            prev_p, _, _ = _pad8(np.asarray(prev, dtype=np.uint8))
            if prev_p.shape != (int(h), int(w), 3):
                pred = np.full((int(h), int(w), 3), 128.0, dtype=np.float32)
            else:
                pred = _rgb_to_ycbcr(prev_p)
        recon_yc = pred + _idct_image(q.astype(np.float32) * Q)
        return np.clip(_ycbcr_to_rgb(recon_yc), 0, 255).astype(np.uint8)
    if prev is None:
        pred = np.full((int(h), int(w), 3), 128.0, dtype=np.float32)
    else:
        prev_p, _, _ = _pad8(np.asarray(prev, dtype=np.uint8))
        if prev_p.shape != (int(h), int(w), 3):
            pred = np.full((int(h), int(w), 3), 128.0, dtype=np.float32)
        else:
            pred = prev_p.astype(np.float32)
    recon = np.clip(pred + _idct_image(q.astype(np.float32) * float(qstep)), 0, 255)
    return recon.astype(np.uint8)


def decode_temporal_dct(payload: bytes) -> List[np.ndarray]:
    """Decode a payload from ``encode_temporal_dct``."""
    if len(payload) < _HEAD.size + 512:
        raise ValueError("truncated LDCT payload")
    magic, ver, _flags, qstep, n, h, w = _HEAD.unpack_from(payload, 0)
    if magic != MAGIC:
        raise ValueError(f"bad LDCT magic {magic!r}")
    if int(ver) not in (VERSION_RGB_UNIFORM, VERSION):
        raise ValueError(f"unsupported LDCT version {ver}")
    off = _HEAD.size
    u16 = np.frombuffer(payload[off : off + 512], dtype="<u2").copy()
    off += 512
    pmf = _pmf_from_u16(u16)
    n_sym = int(n) * int(h) * int(w) * 3
    symbols = rans_decode(payload[off:], n_sym, pmf)
    return _decode_symbols_to_frames(
        symbols, n=int(n), h=int(h), w=int(w), qstep=float(qstep), version=int(ver)
    )


def sweep_temporal_dct(
    frames: Sequence[np.ndarray],
    qsteps: Iterable[float] = (2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 96),
) -> List[dict]:
    """Operating points with countable bpp and reconstruction arrays.

    Includes fine qsteps (2–3) so matched-bpp JPEG comparisons remain possible
    after the v2 rate drop, plus coarse points for below-floor wins.
    """
    points = []
    for q in qsteps:
        payload, recons, stats = encode_temporal_dct(frames, qstep=float(q))
        points.append({"payload": payload, "recons": recons, **stats})
    return points
