# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# Countable wire rate vs differentiable entropy proxy.
# See docs/architecture/TRUE_RATE_APPLIED_MATH.md

"""True (countable) rate helpers for quantized latents."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch


def bits_per_index(num_levels: int) -> int:
    """Fixed-length bits needed to store one symbol from a codebook of size L."""
    if num_levels <= 1:
        return 0
    return int(math.ceil(math.log2(num_levels)))


def fixed_length_rate_bits(latent_dim: int, num_levels: int = 256) -> float:
    """Hard upper bound on index payload without entropy coding: d * ceil(log2 L)."""
    return float(latent_dim * bits_per_index(num_levels))


def pack_indices_u8(indices: torch.Tensor, num_levels: int = 256) -> bytes:
    """
    Pack quantization indices into a wire payload.

    For L <= 256 each index fits in one uint8 → nbytes = d (per batch row we pack
    the first sample for edge demos; batch packing concatenates rows).
    """
    idx = indices.detach().cpu().numpy().astype(np.int64)
    if num_levels <= 256:
        return np.clip(idx, 0, 255).astype(np.uint8).tobytes()
    # Wider codes: store as little-endian uint16
    return np.clip(idx, 0, 65535).astype("<u2").tobytes()


def unpack_indices_u8(
    payload: bytes,
    *,
    batch: int,
    latent_dim: int,
    num_levels: int = 256,
) -> np.ndarray:
    if num_levels <= 256:
        arr = np.frombuffer(payload, dtype=np.uint8)
    else:
        arr = np.frombuffer(payload, dtype="<u2")
    return arr.reshape(batch, latent_dim)


def countable_rate_from_indices(
    indices: torch.Tensor,
    num_levels: int = 256,
) -> Dict[str, float]:
    """
    Measure countable rate for a batch of index tensors (B, D).

    Returns per-batch-mean bits and the fixed-length ceiling.
    """
    if indices is None:
        return {
            "true_rate_bits": 0.0,
            "fixed_length_bits": 0.0,
            "packed_bytes": 0.0,
            "bits_per_symbol": float(bits_per_index(num_levels)),
        }
    b, d = indices.shape[0], indices.shape[-1]
    packed = pack_indices_u8(indices, num_levels=num_levels)
    packed_bits = float(len(packed) * 8) / max(b, 1)
    fixed = fixed_length_rate_bits(d, num_levels)
    return {
        "true_rate_bits": packed_bits,  # actual packed payload bits / sample
        "fixed_length_bits": fixed,
        "packed_bytes": float(len(packed)) / max(b, 1),
        "bits_per_symbol": float(bits_per_index(num_levels)),
        "latent_dim": float(d),
    }


def rate_report(
    proxy_bits: torch.Tensor,
    indices: torch.Tensor | None,
    num_levels: int = 256,
) -> Tuple[Dict[str, float], bytes]:
    """Combine proxy entropy estimate with countable packed indices."""
    proxy = float(proxy_bits.mean().detach().cpu()) if proxy_bits is not None and proxy_bits.numel() else 0.0
    if indices is None:
        stats = {
            "proxy_rate_bits": proxy,
            "true_rate_bits": 0.0,
            "fixed_length_bits": 0.0,
            "packed_bytes": 0.0,
            "proxy_vs_true_ratio": float("nan"),
        }
        return stats, b""
    countable = countable_rate_from_indices(indices, num_levels=num_levels)
    packed = pack_indices_u8(indices, num_levels=num_levels)
    true_b = countable["true_rate_bits"]
    stats = {
        "proxy_rate_bits": proxy,
        "true_rate_bits": true_b,
        "fixed_length_bits": countable["fixed_length_bits"],
        "packed_bytes": countable["packed_bytes"],
        "proxy_vs_true_ratio": proxy / max(true_b, 1e-8),
    }
    return stats, packed
