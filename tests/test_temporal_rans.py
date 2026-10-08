# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
from __future__ import annotations

import numpy as np

from lydlr_ai.model.bench_baselines import mse, psnr
from lydlr_ai.model.temporal_rans import encode_temporal_rans
from lydlr_ai.model.temporal_residual import synthetic_correlated_clip


def test_temporal_rans_roundtrip_reasonable():
    frames = synthetic_correlated_clip(n=8, h=32, w=32, seed=5)
    payload, recons, stats = encode_temporal_rans(frames, steps=128)
    assert stats["rate_source"] == "rans_bits"
    assert payload
    assert len(recons) == len(frames)
    errs = [mse(f, r) for f, r in zip(frames, recons)]
    assert np.mean(errs) < 0.05  # reasonable quality


def test_temporal_rans_has_countable_bits():
    frames = synthetic_correlated_clip(n=4, h=16, w=16, seed=10)
    payload, recons, stats = encode_temporal_rans(frames, steps=256)
    assert stats["total_bits"] == len(payload) * 8
    assert stats["bpp"] > 0
