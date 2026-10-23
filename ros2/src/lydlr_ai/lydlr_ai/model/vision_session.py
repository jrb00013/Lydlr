# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
"""Rolling vision encode session over the checked-in correlated clip.

Each tick feeds the next frame through the temporal DCT codec and returns
countable byte counts plus JPEG previews. No ROS, database, or checkpoint.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from lydlr_ai.model.bench_baselines import psnr
from lydlr_ai.model.temporal_dct import decode_frame, encode_frame

_ROOT = Path(__file__).resolve().parents[5]
_FIXTURE = _ROOT / "scripts" / "fixture_correlated_clip.npz"


def _jpeg(rgb: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(
        buf, format="JPEG", quality=80
    )
    return buf.getvalue()


def _load_frames(max_width: int = 96, n: int = 12) -> np.ndarray:
    data = np.load(_FIXTURE, allow_pickle=True)
    raw = data["frames"] if "frames" in data.files else data[data.files[0]]
    frames = np.asarray(raw, dtype=np.uint8)[:n]
    h, w = frames.shape[1:3]
    if w <= max_width:
        return frames
    from PIL import Image

    new_w = max(8, max_width - (max_width % 8))
    new_h = max(8, int(round(h * new_w / w)))
    new_h -= new_h % 8
    out = [
        np.asarray(Image.fromarray(f).resize((new_w, new_h), Image.Resampling.BILINEAR), dtype=np.uint8)
        for f in frames
    ]
    return np.stack(out, axis=0)


def _heat(src: np.ndarray, recon: np.ndarray) -> np.ndarray:
    delta = np.mean(np.abs(src.astype(np.int16) - recon.astype(np.int16)), axis=2)
    heat = np.zeros_like(src)
    heat[:, :, 0] = np.clip(delta * 6, 0, 255).astype(np.uint8)
    heat[:, :, 1] = np.clip(180 - delta * 2, 0, 255).astype(np.uint8)
    heat[:, :, 2] = np.clip(40 + delta, 0, 255).astype(np.uint8)
    return heat


class VisionSession:
    def __init__(self, *, qstep: float = 16.0, max_width: int = 96, n_frames: int = 12):
        self.qstep = float(qstep)
        self.frames = _load_frames(max_width=max_width, n=n_frames)
        self.index = 0
        self.prev: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def tick(self) -> Dict:
        with self._lock:
            frame = self.frames[self.index % len(self.frames)]
            if self.index % len(self.frames) == 0:
                self.prev = None
            payload, recon, stats = encode_frame(frame, self.prev, qstep=self.qstep)
            decoded = decode_frame(payload, self.prev)
            if decoded.shape != recon.shape or not np.array_equal(decoded, recon):
                raise RuntimeError("vision codec decode diverged from encoder reconstruction")
            psnr_db = float(psnr(frame, recon))
            self.prev = recon
            self.index += 1
            metric = {
                "node_id": "vision_codec",
                "vertical": "drone",
                "codec_path": stats["codec_path"],
                "rate_source": stats["rate_source"],
                "bytes_in": stats["bytes_in"],
                "bytes_out": stats["bytes_out"],
                "compression_ratio": round(stats["compression_ratio"], 3),
                "bpp": round(stats["bpp"], 4),
                "psnr": round(psnr_db, 3),
                "quality_score": round(min(1.0, psnr_db / 50.0), 4),
                "compression_level": round(min(0.98, self.qstep / 100.0), 3),
                "latency_ms": 0.0,
                "bandwidth_estimate": round(stats["bytes_out"] * 8 / 1000.0, 3),
                "keyframe": stats["keyframe"],
                "modality_bytes_in": {"camera": stats["bytes_in"]},
                "modality_bytes_out": {"camera": stats["bytes_out"]},
                "demo": False,
            }
            previews = {
                "raw": _jpeg(frame),
                "reconstructed": _jpeg(recon),
                "heatmap": _jpeg(_heat(frame, recon)),
            }
            return {"metric": metric, "previews": previews, "payload": payload}


_default: Optional[VisionSession] = None
_default_lock = threading.Lock()


def get_session() -> VisionSession:
    global _default
    with _default_lock:
        if _default is None:
            _default = VisionSession()
        return _default
