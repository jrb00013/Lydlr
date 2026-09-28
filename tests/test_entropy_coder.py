"""Tests for pure-numpy rANS entropy coder."""

import numpy as np
import pytest

from lydlr_ai.model.entropy_coder import (
    RansCoder,
    calibrate_temperature,
    empirical_pmf,
    rans_decode,
    rans_encode,
)


def _entropy_bits(indices: np.ndarray, probs: np.ndarray) -> float:
    p = probs[np.asarray(indices, dtype=np.int64)]
    return float(-np.sum(np.log2(np.clip(p, 1e-300, 1.0))))


def test_round_trip_uniform():
    rng = np.random.default_rng(0)
    probs = np.ones(256, dtype=np.float64) / 256.0
    indices = rng.integers(0, 256, size=512, dtype=np.uint8)
    payload = rans_encode(indices, probs)
    back = rans_decode(payload, indices.size, probs)
    np.testing.assert_array_equal(back, indices)


def test_round_trip_skewed():
    rng = np.random.default_rng(1)
    probs = np.zeros(256, dtype=np.float64)
    probs[0] = 0.7
    probs[1] = 0.2
    probs[2:10] = 0.1 / 8.0
    indices = rng.choice(256, size=4096, p=probs).astype(np.uint8)
    payload = rans_encode(indices, probs)
    back = rans_decode(payload, indices.size, probs)
    np.testing.assert_array_equal(back, indices)


def test_coded_rate_near_entropy_long_sequence():
    rng = np.random.default_rng(2)
    probs = np.zeros(256, dtype=np.float64)
    probs[0] = 0.5
    probs[1] = 0.25
    probs[2:32] = 0.25 / 30.0
    n = 4096
    indices = rng.choice(256, size=n, p=probs).astype(np.uint8)
    payload = rans_encode(indices, probs)
    coded_bits = len(payload) * 8
    ent = _entropy_bits(indices, probs)
    overhead = coded_bits - ent
    assert overhead >= 0
    assert coded_bits <= ent * 1.05 + 64


def test_empty_indices():
    probs = np.ones(256, dtype=np.float64) / 256.0
    payload = rans_encode(np.asarray([], dtype=np.uint8), probs)
    assert payload in (b"", payload)  # minimal
    assert len(payload) <= 4
    back = rans_decode(payload or _empty_decode_guard(), 0, probs)
    assert back.size == 0


def _empty_decode_guard():
    return b""


def test_empirical_pmf_and_rans_coder_class():
    indices = np.array([0, 0, 1, 2], dtype=np.uint8)
    pmf = empirical_pmf(indices, 4)
    assert pmf.shape == (4,)
    assert np.isclose(pmf.sum(), 1.0)
    coder = RansCoder()
    payload = coder.encode(indices, pmf)
    back = coder.decode(payload, len(indices), pmf)
    np.testing.assert_array_equal(back, indices)


def test_calibrate_temperature_runs():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(128, 16))
    indices = rng.integers(0, 16, size=128, dtype=np.uint8)
    t = calibrate_temperature(logits, indices)
    assert t > 0


def test_skewed_coded_bits_example():
    """Expose coded_bits vs entropy for reporting."""
    rng = np.random.default_rng(42)
    probs = np.zeros(256, dtype=np.float64)
    probs[0] = 0.85
    probs[1] = 0.10
    probs[2:8] = 0.05 / 6.0
    indices = rng.choice(256, size=4096, p=probs).astype(np.uint8)
    payload = rans_encode(indices, probs)
    coded_bits = len(payload) * 8
    ent = _entropy_bits(indices, probs)
    print(f"skewed pmf: coded_bits={coded_bits:.1f} entropy={ent:.1f} overhead={coded_bits - ent:.1f}")
    assert coded_bits >= ent
