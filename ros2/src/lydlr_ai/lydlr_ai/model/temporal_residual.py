# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
"""Temporal JPEG residual baseline: exploit clip correlation vs intra JPEG."""

from __future__ import annotations

import io
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np

from lydlr_ai.model.bench_baselines import psnr as _psnr

FrameSeq = Union[Sequence[np.ndarray], np.ndarray]

DEFAULT_JPEG_QUALITIES: Tuple[int, ...] = tuple(range(5, 100, 5))


def _as_frame_list(frames: FrameSeq) -> List[np.ndarray]:
    arr = np.asarray(frames)
    if arr.ndim == 4:
        return [np.asarray(arr[i], dtype=np.uint8) for i in range(arr.shape[0])]
    if arr.ndim == 3:
        return [np.asarray(f, dtype=np.uint8) for f in frames]
    raise ValueError(f"expected NxHxWx3 or list of HxWx3 frames, got shape {arr.shape}")


def _jpeg_encode_rgb(frame: np.ndarray, quality: int) -> bytes:
    q = int(max(1, min(100, quality)))
    try:
        import cv2

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            raise RuntimeError("cv2.imencode JPEG failed")
        return bytes(buf)
    except Exception:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, "JPEG", quality=q)
        return buf.getvalue()


def _jpeg_decode_rgb(payload: bytes) -> np.ndarray:
    try:
        import cv2

        arr = np.frombuffer(payload, dtype=np.uint8)
        out = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if out is None:
            raise RuntimeError("cv2.imdecode failed")
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    except Exception:
        from PIL import Image

        with Image.open(io.BytesIO(payload)) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _clip_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def _avg_bpp(payloads: Sequence[bytes], h: int, w: int) -> float:
    n_pix = float(h * w)
    total_bits = sum(8 * len(p) for p in payloads)
    return total_bits / (len(payloads) * n_pix)


def encode_jpeg_intra(
    frames: FrameSeq,
    quality: int,
) -> Tuple[List[bytes], List[np.ndarray], float]:
    """Independent JPEG per frame. bits_per_frame = mean countable payload bits."""
    flist = _as_frame_list(frames)
    if not flist:
        raise ValueError("encode_jpeg_intra: empty frame list")
    h, w = flist[0].shape[:2]
    payloads: List[bytes] = []
    recons: List[np.ndarray] = []
    for f in flist:
        pl = _jpeg_encode_rgb(f, quality)
        payloads.append(pl)
        recons.append(_jpeg_decode_rgb(pl))
    bpp = _avg_bpp(payloads, h, w)
    bits_per_frame = bpp * h * w
    return payloads, recons, float(bits_per_frame)


def encode_jpeg_residual(
    frames: FrameSeq,
    quality: int,
    *,
    residual_scale: float = 1.0,
) -> Tuple[List[bytes], List[np.ndarray], float]:
    """I-frame JPEG + JPEG residuals vs previous reconstruction."""
    flist = _as_frame_list(frames)
    if not flist:
        raise ValueError("encode_jpeg_residual: empty frame list")
    scale = float(residual_scale)
    if scale <= 0:
        raise ValueError("residual_scale must be positive")
    h, w = flist[0].shape[:2]
    payloads: List[bytes] = []
    recons: List[np.ndarray] = []

    pl0 = _jpeg_encode_rgb(flist[0], quality)
    payloads.append(pl0)
    prev = _jpeg_decode_rgb(pl0).astype(np.float64)
    recons.append(_clip_u8(prev))

    for f in flist[1:]:
        diff = (f.astype(np.float64) - prev) * scale + 128.0
        residual_img = _clip_u8(diff)
        pl = _jpeg_encode_rgb(residual_img, quality)
        payloads.append(pl)
        dec_res = _jpeg_decode_rgb(pl).astype(np.float64)
        prev = np.clip(prev + (dec_res - 128.0) / scale, 0.0, 255.0)
        recons.append(_clip_u8(prev))

    bpp = _avg_bpp(payloads, h, w)
    bits_per_frame = bpp * h * w
    return payloads, recons, float(bits_per_frame)


def synthetic_correlated_clip(
    n: int = 16,
    h: int = 240,
    w: int = 320,
    seed: int = 0,
) -> np.ndarray:
    """Slow pan + soft blobs; high temporal correlation (uint8 NxHxWx3)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = np.zeros((h, w, 3), dtype=np.float32)
    for _ in range(6):
        cx = rng.uniform(0, w)
        cy = rng.uniform(0, h)
        sx = rng.uniform(w * 0.15, w * 0.45)
        sy = rng.uniform(h * 0.15, h * 0.45)
        col = rng.uniform(0.2, 0.85, size=3)
        g = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        bg += g[..., None] * col.astype(np.float32)
    bg = bg / (bg.max() + 1e-6) * 180.0 + 40.0

    blobs = []
    for _ in range(5):
        blobs.append(
            (
                rng.uniform(0.2 * w, 0.8 * w),
                rng.uniform(0.2 * h, 0.8 * h),
                rng.uniform(12, 28),
                rng.uniform(0.3, 0.95, size=3),
            )
        )

    ox, oy = 0.0, 0.0
    frames: List[np.ndarray] = []
    for _ in range(n):
        ox += 0.35
        oy += 0.12
        img = np.roll(bg, (int(oy) % h, int(ox) % w), axis=(0, 1))
        layer = img.copy()
        for bx, by, br, bc in blobs:
            cx = (bx + ox * 0.15) % w
            cy = (by + oy * 0.1) % h
            dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
            mask = np.exp(-dist2 / (2.0 * br * br))
            layer += mask[..., None] * (bc.astype(np.float32) * 90.0)
        layer = np.clip(layer, 0, 255).astype(np.uint8)
        frames.append(layer)
    return np.stack(frames, axis=0)


def _mean_psnr(frames: FrameSeq, recons: Sequence[np.ndarray]) -> float:
    flist = _as_frame_list(frames)
    vals = [_psnr(f, r) for f, r in zip(flist, recons)]
    return float(np.mean(vals))


def _interpolate_intra_at_bpp(
    intra_curve: List[Tuple[int, float, float]],
    target_bpp: float,
) -> Tuple[float, int, bool]:
    """Return (jpeg_psnr, jpeg_quality, matched_within_5pct)."""
    if not intra_curve:
        return 0.0, 0, False
    sorted_curve = sorted(intra_curve, key=lambda x: x[1])
    bpps = [x[1] for x in sorted_curve]
    if target_bpp <= bpps[0]:
        q, bpp, p = sorted_curve[0]
        ok = abs(bpp - target_bpp) / max(target_bpp, 1e-9) <= 0.05
        return p, q, ok
    if target_bpp >= bpps[-1]:
        q, bpp, p = sorted_curve[-1]
        ok = abs(bpp - target_bpp) / max(target_bpp, 1e-9) <= 0.05
        return p, q, ok
    for i in range(len(sorted_curve) - 1):
        q0, b0, p0 = sorted_curve[i]
        q1, b1, p1 = sorted_curve[i + 1]
        if b0 <= target_bpp <= b1 or b1 <= target_bpp <= b0:
            if abs(b0 - target_bpp) / max(target_bpp, 1e-9) <= 0.05:
                return p0, q0, True
            if abs(b1 - target_bpp) / max(target_bpp, 1e-9) <= 0.05:
                return p1, q1, True
            if abs(b1 - b0) < 1e-12:
                return p0, q0, False
            t = (target_bpp - b0) / (b1 - b0)
            psnr_interp = p0 + t * (p1 - p0)
            q_interp = int(round(q0 + t * (q1 - q0)))
            return float(psnr_interp), q_interp, False
    q, bpp, p = min(sorted_curve, key=lambda x: abs(x[1] - target_bpp))
    ok = abs(bpp - target_bpp) / max(target_bpp, 1e-9) <= 0.05
    return p, q, ok


def match_rate_beat_jpeg(
    frames: FrameSeq,
    target_bpp: float | None = None,
    jpeg_qualities: Iterable[int] = DEFAULT_JPEG_QUALITIES,
) -> dict:
    """Sweep intra vs residual JPEG; compare PSNR at matched average bpp."""
    flist = _as_frame_list(frames)
    if not flist:
        raise ValueError("match_rate_beat_jpeg: empty frame list")
    h, w = flist[0].shape[:2]
    n_pix = float(h * w)
    qualities = [int(q) for q in jpeg_qualities]

    intra_curve: List[Tuple[int, float, float]] = []
    intra_by_q: dict[int, dict] = {}
    for q in qualities:
        payloads, recons, bpf = encode_jpeg_intra(flist, q)
        bpp = _avg_bpp(payloads, h, w)
        p = _mean_psnr(flist, recons)
        intra_curve.append((q, bpp, p))
        intra_by_q[q] = {"bpp": bpp, "psnr": p, "bits_per_frame": bpf}

    residual_rows: List[dict] = []
    beats: List[dict] = []
    best_margin_db = -1e9
    best_row: dict | None = None

    for q in qualities:
        payloads, recons, bpf = encode_jpeg_residual(flist, q)
        res_bpp = _avg_bpp(payloads, h, w)
        res_psnr = _mean_psnr(flist, recons)
        jpeg_psnr, jpeg_q, matched_5 = _interpolate_intra_at_bpp(intra_curve, res_bpp)
        margin = res_psnr - jpeg_psnr
        beats_jpeg = margin > 0.0 and matched_5
        row = {
            "residual_quality": q,
            "residual_bpp": res_bpp,
            "residual_psnr": res_psnr,
            "residual_bits_per_frame": bpf,
            "matched_jpeg_quality": jpeg_q,
            "matched_jpeg_psnr": jpeg_psnr,
            "matched_jpeg_bpp": intra_by_q.get(jpeg_q, {}).get("bpp", res_bpp),
            "bpp_match_within_5pct": matched_5,
            "psnr_margin_db": margin,
            "beats_jpeg_at_matched_bpp": beats_jpeg,
        }
        residual_rows.append(row)
        if beats_jpeg:
            beats.append(row)
        if margin > best_margin_db:
            best_margin_db = margin
            best_row = row

    if target_bpp is not None:
        target_bpp = float(target_bpp)

    return {
        "n_frames": len(flist),
        "height": h,
        "width": w,
        "pixels_per_frame": int(n_pix),
        "target_bpp": target_bpp,
        "jpeg_intra_curve": [
            {"quality": q, "bpp": bpp, "psnr": p} for q, bpp, p in intra_curve
        ],
        "residual_operating_points": residual_rows,
        "beats_jpeg_within_5pct_bpp": beats,
        "any_residual_beats_jpeg": len(beats) > 0,
        "best_psnr_margin_db": float(best_margin_db),
        "best_operating_point": best_row,
    }
