"""Countable rate: rANS payload must beat fixed-length on peaked latents."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lydlr_ai.model.entropy_coder import rans_decode
from lydlr_ai.model.true_rate import (
    countable_rate_from_indices,
    fixed_length_rate_bits,
    pack_indices_rans,
    rate_report,
)


def test_rans_pack_roundtrip_and_beats_fixed_length():
    # Peaked indices → rANS should beat 8 bits/symbol fixed packing.
    rng = np.random.default_rng(0)
    idx = rng.choice(np.arange(256), size=(4, 64), p=_peaked_pmf()).astype(np.uint8)
    t = torch.from_numpy(idx)
    packed, probs = pack_indices_rans(t, num_levels=256)
    back = rans_decode(packed, idx.size, probs)
    np.testing.assert_array_equal(back, idx.reshape(-1))

    stats = countable_rate_from_indices(t, num_levels=256, prefer_rans=True)
    fixed = fixed_length_rate_bits(64, 256)
    assert stats["rate_source"] == "rans_bits"
    assert stats["true_rate_bits"] < fixed
    assert stats["true_rate_bits"] < 8 * 64  # per sample


def test_rate_report_per_sample_and_source():
    idx = torch.zeros((2, 32), dtype=torch.long)
    idx[:, :2] = 3
    proxy = torch.tensor([1.0, 1.2])
    stats, payload = rate_report(proxy, idx, num_levels=256)
    assert stats["rate_source"] == "rans_bits"
    assert len(payload) > 0
    # Per-sample: total payload bits / batch
    assert stats["true_rate_bits"] == pytest.approx(len(payload) * 8 / 2)
    assert stats["packed_bytes"] == pytest.approx(len(payload) / 2)
    assert stats["true_rate_bits"] < stats["fixed_length_bits"]
    assert 0.0 < stats["proxy_vs_true_ratio"] < 2.0 or stats["proxy_rate_bits"] == 0.0


def test_fixed_length_fallback_path():
    idx = torch.arange(16, dtype=torch.long).view(1, 16) % 256
    stats = countable_rate_from_indices(idx, num_levels=256, prefer_rans=False)
    assert stats["rate_source"] == "fixed_length_bits"
    assert stats["true_rate_bits"] == pytest.approx(16 * 8)


def _peaked_pmf() -> np.ndarray:
    p = np.full(256, 1e-6, dtype=np.float64)
    p[120:136] = 1.0
    p /= p.sum()
    return p
