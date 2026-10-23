"""Round-trip and rate tests for the temporal DCT + rANS vision codec."""

from __future__ import annotations

import numpy as np

from lydlr_ai.model.bench_baselines import psnr
from lydlr_ai.model.temporal_dct import decode_temporal_dct, encode_temporal_dct
from lydlr_ai.model.temporal_residual import encode_jpeg_intra, synthetic_correlated_clip


def test_roundtrip_matches_encoder_recon():
    frames = synthetic_correlated_clip(n=4, h=32, w=32, seed=1)
    payload, recons, stats = encode_temporal_dct(frames, qstep=12.0)
    back = decode_temporal_dct(payload)
    assert stats["rate_source"] == "rans_bits"
    assert stats["payload_bytes"] == len(payload)
    assert stats["bpp"] > 0
    for a, b in zip(recons, back):
        np.testing.assert_array_equal(a, b)
    assert float(np.mean([psnr(f, r) for f, r in zip(frames, recons)])) > 30.0


def test_coarser_qstep_uses_fewer_bits():
    frames = synthetic_correlated_clip(n=4, h=32, w=32, seed=2)
    fine = encode_temporal_dct(frames, qstep=8.0)[2]["bpp"]
    coarse = encode_temporal_dct(frames, qstep=48.0)[2]["bpp"]
    assert coarse < fine


def test_beats_intra_jpeg_on_correlated_clip():
    frames = synthetic_correlated_clip(n=6, h=48, w=64, seed=3)
    h, w = frames.shape[1:3]
    jpeg = []
    for q in range(5, 96, 10):
        payloads, jrecons, _bpf = encode_jpeg_intra(frames, q)
        bpp = sum(8 * len(p) for p in payloads) / (len(payloads) * h * w)
        jpeg.append((bpp, float(np.mean([psnr(f, r) for f, r in zip(frames, jrecons)]))))
    jmin_bpp, jmin_psnr = min(jpeg, key=lambda t: t[0])
    wins = False
    for qstep in (8.0, 16.0, 32.0, 64.0):
        _payload, recons, stats = encode_temporal_dct(frames, qstep=qstep)
        lp = float(np.mean([psnr(f, r) for f, r in zip(frames, recons)]))
        peers = [p for b, p in jpeg if abs(b - stats["bpp"]) / max(stats["bpp"], 1e-9) <= 0.15]
        if peers and lp > max(peers) + 0.5:
            wins = True
        if stats["bpp"] < jmin_bpp and lp >= jmin_psnr:
            wins = True
    assert wins
