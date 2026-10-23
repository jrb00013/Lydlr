# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
"""Pure bitrate / bpp helpers with no torch dependency."""

from __future__ import annotations

from typing import Iterable, Sequence


def bits_from_bytes(nbytes: int) -> int:
    return int(nbytes) * 8


def bpp(payload_bytes: int, height: int, width: int, *, channels: int = 1) -> float:
    """Bits per pixel for a countable payload over an HxW (optionally Cx) frame."""
    pixels = max(int(height) * int(width) * max(int(channels), 1), 1)
    return bits_from_bytes(payload_bytes) / float(pixels)


def mean_bpp(payloads: Sequence[bytes], height: int, width: int) -> float:
    if not payloads:
        return 0.0
    total_bits = sum(bits_from_bytes(len(p)) for p in payloads)
    pixels = max(int(height) * int(width), 1)
    return total_bits / (len(payloads) * pixels)


def kbps_from_bytes(nbytes: int, duration_sec: float) -> float:
    if duration_sec <= 0:
        return 0.0
    return bits_from_bytes(nbytes) / duration_sec / 1000.0


def match_within_tol(actual: float, target: float, *, rel_tol: float = 0.05) -> bool:
    if target <= 0:
        return abs(actual) <= 1e-9
    return abs(actual - target) / target <= rel_tol


def compression_ratio(bytes_in: int, bytes_out: int) -> float:
    if bytes_out <= 0:
        return 0.0
    return float(bytes_in) / float(bytes_out)


def savings_fraction(bytes_in: int, bytes_out: int) -> float:
    if bytes_in <= 0:
        return 0.0
    return max(0.0, float(bytes_in - bytes_out) / float(bytes_in))


def harmonic_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v and v > 0]
    if not vals:
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)
