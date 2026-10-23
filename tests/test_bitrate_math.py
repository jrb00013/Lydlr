"""Tests for pure bitrate / bpp helpers."""

from __future__ import annotations

import pytest

from lydlr_ai.utils.bitrate_math import (
    bits_from_bytes,
    bpp,
    compression_ratio,
    harmonic_mean,
    kbps_from_bytes,
    match_within_tol,
    mean_bpp,
    savings_fraction,
)


def test_bits_and_bpp():
    assert bits_from_bytes(10) == 80
    assert bpp(100, 10, 10) == pytest.approx(8.0)  # 800 bits / 100 px
    assert mean_bpp([b"abcd", b"efgh"], 8, 8) == pytest.approx(0.5)  # 32 bits / 64 px


def test_kbps_and_match():
    assert kbps_from_bytes(1250, 0.1) == pytest.approx(100.0)
    assert kbps_from_bytes(100, 0) == 0.0
    assert match_within_tol(1.0, 1.04, rel_tol=0.05)
    assert not match_within_tol(1.0, 1.2, rel_tol=0.05)


def test_ratio_and_savings():
    assert compression_ratio(1000, 250) == pytest.approx(4.0)
    assert compression_ratio(100, 0) == 0.0
    assert savings_fraction(1000, 250) == pytest.approx(0.75)
    assert savings_fraction(0, 10) == 0.0


def test_harmonic_mean():
    assert harmonic_mean([2, 4, 4]) == pytest.approx(3.0)
    assert harmonic_mean([]) == 0.0
    assert harmonic_mean([0, -1]) == 0.0
