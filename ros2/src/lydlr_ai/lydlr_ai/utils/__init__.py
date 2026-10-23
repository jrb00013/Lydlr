# Lydlr utilities package
"""Shared helpers (bitrate math, clip stats, LYD format, metrics)."""

from lydlr_ai.utils.bitrate_math import (  # noqa: F401
    bits_from_bytes,
    bpp,
    compression_ratio,
    kbps_from_bytes,
    mean_bpp,
    savings_fraction,
)
