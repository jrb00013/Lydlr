# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
"""Lightweight NPZ clip statistics for fixture sanity checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

PathLike = Union[str, Path]


def load_frames(path: PathLike, *, max_frames: Optional[int] = None) -> np.ndarray:
    """Load HxWx3 uint8 frames from a Lydlr fixture NPZ."""
    data = np.load(path, allow_pickle=True)
    if "frames" in data:
        raw = data["frames"]
    else:
        raw = data[list(data.files)[0]]
    raw = np.asarray(raw, dtype=object)
    if max_frames is not None:
        raw = raw[: max_frames]
    if raw.ndim == 1 and raw.size and isinstance(raw[0], dict):
        imgs = [np.asarray(entry["image"], dtype=np.uint8) for entry in raw]
        return np.stack(imgs, axis=0)
    frames = np.asarray(raw)
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    return frames


def mean_abs_temporal_delta(frames: np.ndarray) -> float:
    """Mean |Δpixel| across consecutive frames; low ⇒ temporally correlated."""
    if len(frames) < 2:
        return 0.0
    diffs = [
        float(np.mean(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))))
        for i in range(1, len(frames))
    ]
    return float(np.mean(diffs))


def is_temporally_correlated(
    frames: np.ndarray,
    *,
    max_mean_abs_delta: float = 25.0,
) -> bool:
    return mean_abs_temporal_delta(frames) <= max_mean_abs_delta


def summarize_clip(path: PathLike, *, max_frames: int = 32) -> Dict[str, Any]:
    frames = load_frames(path, max_frames=max_frames)
    n, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    delta = mean_abs_temporal_delta(frames)
    return {
        "path": str(path),
        "n_frames": n,
        "height": h,
        "width": w,
        "channels": int(frames.shape[3]) if frames.ndim == 4 else 1,
        "mean_abs_temporal_delta": delta,
        "temporally_correlated": delta <= 25.0,
        "mean_pixel": float(np.mean(frames)),
        "std_pixel": float(np.std(frames)),
    }
