# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black

from __future__ import annotations

import numpy as np

from lydlr_ai.model.bench_baselines import mse
from lydlr_ai.model.temporal_residual import (
    encode_jpeg_intra,
    encode_jpeg_residual,
    match_rate_beat_jpeg,
    synthetic_correlated_clip,
)


def test_synthetic_residual_beats_jpeg_at_matched_rate():
    frames = synthetic_correlated_clip(n=16, h=120, w=160, seed=42)
    report = match_rate_beat_jpeg(
        frames,
        jpeg_qualities=tuple(range(10, 96, 5)),
    )
    assert report["any_residual_beats_jpeg"], (
        f"expected residual win; best margin {report['best_psnr_margin_db']:.3f} dB"
    )
    assert report["best_psnr_margin_db"] > 0.0


def test_residual_roundtrip_mse_low_at_high_quality():
    frames = synthetic_correlated_clip(n=8, h=64, w=64, seed=1)
    _, recons, _ = encode_jpeg_residual(frames, quality=95)
    flist = [frames[i] for i in range(frames.shape[0])]
    err = float(np.mean([mse(f, r) for f, r in zip(flist, recons)]))
    assert err < 0.002, f"high-Q residual MSE too large: {err}"


def test_intra_roundtrip_sane():
    frames = synthetic_correlated_clip(n=4, h=32, w=32, seed=0)
    _, recons, bpf = encode_jpeg_intra(frames, quality=80)
    assert bpf > 0
    assert len(recons) == frames.shape[0]
