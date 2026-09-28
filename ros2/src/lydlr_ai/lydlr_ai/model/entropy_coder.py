"""Pure-numpy rANS entropy coding for discrete symbol indices."""

from __future__ import annotations

import struct
from typing import Union

import numpy as np

# 16-bit renormalization lower bound; 32-bit rANS state.
RANS_BYTE_L = 1 << 16
TABLE_LOG = 12
TABLE_SIZE = 1 << TABLE_LOG
TABLE_MASK = TABLE_SIZE - 1

_HEADER = struct.Struct(">I")


def empirical_pmf(indices: np.ndarray, alphabet_size: int) -> np.ndarray:
    """Normalized histogram PMF over ``alphabet_size`` symbols."""
    flat = np.asarray(indices, dtype=np.int64).reshape(-1)
    counts = np.bincount(flat, minlength=int(alphabet_size)).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return np.full(int(alphabet_size), 1.0 / float(alphabet_size), dtype=np.float64)
    return counts / total


def _quantize_cdf(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build ``freq`` and ``cumul`` (length K+1) summing to ``TABLE_SIZE``."""
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    k = probs.size
    if k == 0:
        raise ValueError("probabilities must be non-empty")

    freq = np.floor(probs * TABLE_SIZE).astype(np.int64)
    # At least one slot for any symbol with non-zero model mass (near-zero safe).
    freq = np.maximum(freq, (probs > 0).astype(np.int64))

    total = int(freq.sum())
    if total == 0:
        freq = np.ones(k, dtype=np.int64)
        total = k

    while total > TABLE_SIZE:
        idx = int(np.argmax(freq))
        if freq[idx] <= 1:
            break
        freq[idx] -= 1
        total -= 1

    while total < TABLE_SIZE:
        idx = int(np.argmax(probs))
        freq[idx] += 1
        total += 1

    if total != TABLE_SIZE:
        # Last-resort uniform fill if rounding fight stalls.
        deficit = TABLE_SIZE - int(freq.sum())
        if deficit > 0:
            order = np.argsort(-probs)
            for j in order:
                if deficit <= 0:
                    break
                freq[j] += 1
                deficit -= 1
        elif deficit < 0:
            order = np.argsort(probs)
            for j in order:
                if deficit >= 0:
                    break
                if freq[j] > 1:
                    freq[j] -= 1
                    deficit += 1

    cumul = np.empty(k + 1, dtype=np.int64)
    cumul[0] = 0
    np.cumsum(freq, out=cumul[1:])
    if cumul[-1] != TABLE_SIZE:
        raise RuntimeError("internal CDF table does not sum to TABLE_SIZE")
    return freq, cumul


def _symbol_from_cdf(cf: int, cumul: np.ndarray) -> int:
    # cumul[s] <= cf < cumul[s+1]
    return int(np.searchsorted(cumul, cf, side="right") - 1)


def rans_encode(indices: np.ndarray, probabilities: np.ndarray) -> bytes:
    """Encode ``indices`` (uint8) with a fixed order-0 ``probabilities`` PMF."""
    sym = np.asarray(indices, dtype=np.uint8).reshape(-1)
    if sym.size == 0:
        return b""

    max_sym = int(sym.max())
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probs.size <= max_sym:
        pad = max_sym + 1 - probs.size
        probs = np.pad(probs, (0, pad), mode="constant")
    s = float(probs.sum())
    if s <= 0:
        probs = np.ones(probs.size, dtype=np.float64) / float(probs.size)
    else:
        probs = probs / s

    freq, cumul = _quantize_cdf(probs)
    x = np.uint32(RANS_BYTE_L)
    stream: list[int] = []

    for s_val in sym[::-1]:
        s_i = int(s_val)
        f = int(freq[s_i])
        if f <= 0:
            raise ValueError(f"symbol {s_i} has zero frequency in CDF table")

        x_max = np.uint32(((RANS_BYTE_L >> TABLE_LOG) << 8) * f)
        while x >= x_max:
            stream.append(int(x & np.uint32(0xFF)))
            x >>= np.uint32(8)

        q = x // np.uint32(f)
        r = x % np.uint32(f)
        x = (q << np.uint32(TABLE_LOG)) + r + np.uint32(cumul[s_i])

    return _HEADER.pack(int(x)) + bytes(stream)


def rans_decode(payload: bytes, n_symbols: int, probabilities: np.ndarray) -> np.ndarray:
    """Decode ``n_symbols`` indices from ``payload`` using the same PMF as encode."""
    n = int(n_symbols)
    if n == 0:
        return np.asarray([], dtype=np.uint8)
    if not payload:
        raise ValueError("empty payload for non-zero n_symbols")

    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    s = float(probs.sum())
    if s <= 0:
        probs = np.ones(probs.size, dtype=np.float64) / float(probs.size)
    else:
        probs = probs / s

    freq, cumul = _quantize_cdf(probs)

    if len(payload) < 4:
        raise ValueError("payload too short for rANS state")
    x = np.uint32(_HEADER.unpack_from(payload, 0)[0])
    stream = payload[4:]
    pos = len(stream) - 1

    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        while x < np.uint32(RANS_BYTE_L):
            if pos < 0:
                raise ValueError("truncated rANS stream during decode")
            x = (x << np.uint32(8)) | np.uint32(stream[pos])
            pos -= 1

        cf = int(x & np.uint32(TABLE_MASK))
        sym = _symbol_from_cdf(cf, cumul)
        f = int(freq[sym])
        x = np.uint32(f) * (x >> np.uint32(TABLE_LOG)) + np.uint32(cf) - np.uint32(cumul[sym])
        out[i] = sym

    return out


encode_indices = rans_encode


class RansCoder:
    """Thin OO wrapper around :func:`rans_encode` / :func:`rans_decode`."""

    def encode(self, indices: np.ndarray, probabilities: np.ndarray) -> bytes:
        return rans_encode(indices, probabilities)

    def decode(self, payload: bytes, n: int, probabilities: np.ndarray) -> np.ndarray:
        return rans_decode(payload, n, probabilities)


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _per_symbol_probs(logits_or_probs: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (matrix probs shape (n,K), vector p[i] for each index)."""
    x = np.asarray(logits_or_probs, dtype=np.float64)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    n = idx.size

    if x.ndim == 1:
        p = x.reshape(-1)
        if np.any(p < 0) or not np.isclose(p.sum(), 1.0, rtol=1e-4, atol=1e-6):
            # Treat as logits over alphabet.
            p = _softmax(x.reshape(1, -1), axis=1)[0]
        else:
            p = p / p.sum()
        probs = np.tile(p.reshape(1, -1), (n, 1))
    elif x.ndim == 2:
        if x.shape[0] != n:
            raise ValueError("logits_or_probs rows must match len(indices)")
        row_sum = x.sum(axis=1, keepdims=True)
        if np.all(x >= 0) and np.allclose(row_sum, 1.0, rtol=1e-3, atol=1e-5):
            probs = x / row_sum
        else:
            probs = _softmax(x, axis=1)
    else:
        raise ValueError("logits_or_probs must be 1-D or 2-D")

    k = probs.shape[1]
    idx = np.clip(idx, 0, k - 1)
    p_i = probs[np.arange(n), idx]
    return probs, p_i


def calibrate_temperature(logits_or_probs: Union[np.ndarray, list], indices: np.ndarray) -> float:
    """Find temperature ``T`` minimizing ``|coded_bits - true_nll_bits|``.

    For 2-D logits, applies ``softmax(logits / T)`` row-wise and uses a shared
    order-0 PMF (mean row) for rANS, matching typical bench usage.
    """
    idx = np.asarray(indices, dtype=np.uint8).reshape(-1)
    if idx.size == 0:
        return 1.0

    x = np.asarray(logits_or_probs, dtype=np.float64)
    if x.ndim == 1:
        base = x.reshape(-1)
        is_prob = np.all(base >= 0) and np.isclose(base.sum(), 1.0, rtol=1e-4, atol=1e-6)

        def pmf_at(t: float) -> np.ndarray:
            if is_prob:
                p = np.power(base, 1.0 / t)
            else:
                p = _softmax((base / t).reshape(1, -1), axis=1)[0]
            return p / p.sum()

    elif x.ndim == 2:
        if x.shape[0] != idx.size:
            raise ValueError("logits rows must match len(indices)")

        def pmf_at(t: float) -> np.ndarray:
            p = _softmax(x / t, axis=1)
            return p.mean(axis=0)

    else:
        raise ValueError("logits_or_probs must be 1-D or 2-D")

    _, p_i = _per_symbol_probs(x if x.ndim == 2 else pmf_at(1.0), idx)
    true_nll = float(-np.sum(np.log2(np.clip(p_i, 1e-300, 1.0))))

    def objective(t: float) -> float:
        pmf = pmf_at(t)
        payload = rans_encode(idx, pmf)
        coded = float(len(payload) * 8)
        return abs(coded - true_nll)

    lo, hi = 0.05, 20.0
    for _ in range(40):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if objective(m1) < objective(m2):
            hi = m2
        else:
            lo = m1
    return float((lo + hi) * 0.5)


def rans_encode_constriction(indices: np.ndarray, probabilities: np.ndarray) -> bytes:
    """Optional cross-check using ``constriction`` when installed."""
    try:
        import constriction
    except ImportError as exc:
        raise ImportError("constriction is not installed") from exc

    sym = np.asarray(indices, dtype=np.uint8).reshape(-1)
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    probs = probs / probs.sum()
    freqs = np.maximum(np.round(probs * TABLE_SIZE).astype(np.int32), 1)
    total = int(freqs.sum())
    diff = TABLE_SIZE - total
    if diff != 0:
        freqs[int(np.argmax(probs))] += diff

    encoder = constriction.stream.stack.AnsCoder()
    for s in sym[::-1]:
        encoder.encode(constriction.stream.model.Categorical(freqs, perfect=False), int(s))
    return encoder.get_compressed()


__all__ = [
    "RansCoder",
    "calibrate_temperature",
    "empirical_pmf",
    "encode_indices",
    "rans_decode",
    "rans_encode",
    "rans_encode_constriction",
]
