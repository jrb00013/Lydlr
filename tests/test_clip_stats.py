"""Tests for NPZ clip summary helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lydlr_ai.utils.clip_stats import (
    is_temporally_correlated,
    load_frames,
    mean_abs_temporal_delta,
    summarize_clip,
)

ROOT = Path(__file__).resolve().parents[1]
CORRELATED = ROOT / "scripts" / "fixture_correlated_clip.npz"


@pytest.mark.skipif(not CORRELATED.is_file(), reason="correlated fixture missing")
def test_load_and_summarize_correlated_fixture():
    frames = load_frames(CORRELATED, max_frames=8)
    assert frames.ndim == 4
    assert frames.dtype == np.uint8
    assert frames.shape[0] == 8
    summary = summarize_clip(CORRELATED, max_frames=8)
    assert summary["n_frames"] == 8
    assert summary["temporally_correlated"] is True
    assert mean_abs_temporal_delta(frames) < 25.0
    assert is_temporally_correlated(frames)


def test_uncorrelated_frames_flagged():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, size=(6, 16, 16, 3), dtype=np.uint8)
    assert mean_abs_temporal_delta(frames) > 25.0
    assert not is_temporally_correlated(frames)
