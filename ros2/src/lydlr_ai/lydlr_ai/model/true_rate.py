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


def pack_indices_rans(
    indices: np.ndarray | torch.Tensor,
    *,
    num_levels: int = 256,
    symbol_probs: np.ndarray | torch.Tensor | None = None,
) -> Tuple[bytes, np.ndarray]:
    """Entropy-code indices with order-0 rANS. Returns payload and the PMF used."""
    from lydlr_ai.model.entropy_coder import empirical_pmf, rans_encode

    if isinstance(indices, torch.Tensor):
        idx_np = indices.detach().cpu().numpy().reshape(-1).astype(np.uint8)
    else:
        idx_np = np.asarray(indices, dtype=np.uint8).reshape(-1)
    alphabet = max(int(num_levels), int(idx_np.max()) + 1 if idx_np.size else 1)
    if symbol_probs is None:
        probs = empirical_pmf(idx_np, alphabet)
    else:
        if isinstance(symbol_probs, torch.Tensor):
            probs = symbol_probs.detach().float().cpu().numpy().astype(np.float64).reshape(-1)
        else:
            probs = np.asarray(symbol_probs, dtype=np.float64).reshape(-1)
        if probs.size < alphabet:
            probs = np.pad(probs, (0, alphabet - probs.size))
        probs = probs[:alphabet]
        s = float(probs.sum())
        probs = probs / s if s > 0 else empirical_pmf(idx_np, alphabet)
    return rans_encode(idx_np, probs), probs


def countable_rate_from_indices(
    indices: torch.Tensor,
    num_levels: int = 256,
    *,
    prefer_rans: bool = True,
) -> Dict[str, float]:
    """
    Measure countable rate for a batch of index tensors (B, D).

    Returns **per-sample** bits. Prefers rANS when ``prefer_rans`` is true so a
    peaked latent distribution reports below the fixed-length ceiling.
    """
    if indices is None:
        return {
            "true_rate_bits": 0.0,
            "fixed_length_bits": 0.0,
            "packed_bytes": 0.0,
            "bits_per_symbol": float(bits_per_index(num_levels)),
            "rate_source": "none",
        }
    b = max(int(indices.shape[0]), 1)
    d = int(indices.shape[-1])
    fixed = fixed_length_rate_bits(d, num_levels)

    if prefer_rans:
        try:
            packed, _probs = pack_indices_rans(indices, num_levels=num_levels)
            packed_bits = float(len(packed) * 8) / b
            return {
                "true_rate_bits": packed_bits,
                "fixed_length_bits": fixed,
                "packed_bytes": float(len(packed)) / b,
                "bits_per_symbol": packed_bits / max(d, 1),
                "latent_dim": float(d),
                "rate_source": "rans_bits",
            }
        except Exception:
            pass

    packed = pack_indices_u8(indices, num_levels=num_levels)
    packed_bits = float(len(packed) * 8) / b
    return {
        "true_rate_bits": packed_bits,
        "fixed_length_bits": fixed,
        "packed_bytes": float(len(packed)) / b,
        "bits_per_symbol": float(bits_per_index(num_levels)),
        "latent_dim": float(d),
        "rate_source": "fixed_length_bits",
    }


def rate_report(
    proxy_bits: torch.Tensor,
    indices: torch.Tensor | None,
    num_levels: int = 256,
    symbol_probs: torch.Tensor | np.ndarray | None = None,
) -> Tuple[Dict[str, float], bytes]:
    """Combine proxy entropy estimate with countable wire payload.

    Prefer rANS (``lydlr_ai.model.entropy_coder.rans_encode``) when available so
    ``true_rate_bits`` is the length of a real bitstream **per sample**, not
    fixed-length packing. Falls back to uint8 packing and labels ``rate_source``.
    """
    proxy = (
        float(proxy_bits.mean().detach().cpu())
        if proxy_bits is not None and getattr(proxy_bits, "numel", lambda: 0)()
        else 0.0
    )
    if indices is None:
        stats = {
            "proxy_rate_bits": proxy,
            "true_rate_bits": 0.0,
            "fixed_length_bits": 0.0,
            "packed_bytes": 0.0,
            "proxy_vs_true_ratio": float("nan"),
            "rate_source": "none",
        }
        return stats, b""

    b = max(int(indices.shape[0]), 1)
    fixed = fixed_length_rate_bits(int(indices.shape[-1]), num_levels)
    packed = b""
    rate_source = "fixed_length_bits"
    true_b = float(fixed)

    try:
        packed, _probs = pack_indices_rans(
            indices, num_levels=num_levels, symbol_probs=symbol_probs
        )
        true_b = float(len(packed) * 8) / b
        rate_source = "rans_bits"
    except Exception:
        packed = pack_indices_u8(indices, num_levels=num_levels)
        true_b = float(len(packed) * 8) / b
        rate_source = "fixed_length_bits"

    stats = {
        "proxy_rate_bits": proxy,
        "true_rate_bits": true_b,
        "fixed_length_bits": fixed,
        "packed_bytes": float(len(packed)) / b,
        "proxy_vs_true_ratio": proxy / max(true_b, 1e-8),
        "rate_source": rate_source,
    }
    return stats, packed
