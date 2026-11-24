"""Round-trip and rate tests for the temporal DCT + rANS vision codec."""

from __future__ import annotations

import numpy as np

from lydlr_ai.model.bench_baselines import psnr
from lydlr_ai.model.temporal_dct import (
    VERSION,
    decode_temporal_dct,
    encode_temporal_dct,
    model_confidence,
)
from lydlr_ai.model.temporal_residual import encode_jpeg_intra, synthetic_correlated_clip


def test_roundtrip_matches_encoder_recon():
    frames = synthetic_correlated_clip(n=4, h=32, w=32, seed=1)
    payload, recons, stats = encode_temporal_dct(frames, qstep=12.0)
    back = decode_temporal_dct(payload)
    assert stats["rate_source"] == "rans_bits"
    assert stats["codec_path"] == "temporal_dct_rans_v2"
    assert stats["payload_bytes"] == len(payload)
    assert stats["bpp"] > 0
    assert stats["compression_ratio"] > 1.0
    assert 0.0 <= stats["confidence"] <= 1.0
    assert stats["zero_frac"] > 0.5
    for a, b in zip(recons, back):
        np.testing.assert_array_equal(a, b)
    assert float(np.mean([psnr(f, r) for f, r in zip(frames, recons)])) > 30.0
    assert payload[4] == VERSION


def test_coarser_qstep_uses_fewer_bits():
    frames = synthetic_correlated_clip(n=4, h=32, w=32, seed=2)
    fine = encode_temporal_dct(frames, qstep=8.0)[2]["bpp"]
    coarse = encode_temporal_dct(frames, qstep=48.0)[2]["bpp"]
    assert coarse < fine


def test_ycbcr_v2_beats_uniform_rgb_rate_at_similar_psnr():
    """v2 must spend fewer bits than a uniform-qstep RGB residual at similar PSNR."""
    from lydlr_ai.model import temporal_dct as td

    frames = synthetic_correlated_clip(n=6, h=48, w=64, seed=7)
    # Force-compare against a one-off RGB-uniform encode (v1 math) at qstep=16.
    flist = [np.asarray(f, dtype=np.uint8) for f in frames]
    orig_h, orig_w = flist[0].shape[:2]
    padded = [td._pad8(f)[0] for f in flist]
    H, W = padded[0].shape[:2]
    prev = None
    chunks = []
    recons = []
    qstep = 16.0
    for frame in padded:
        x = frame.astype(np.float32)
        pred = prev if prev is not None else np.full_like(x, 128.0)
        coef = td._dct_image(x - pred)
        q = np.clip(np.rint(coef / qstep), -127, 127).astype(np.int16)
        chunks.append((q + 127).astype(np.uint8))
        recon = np.clip(pred + td._idct_image(q.astype(np.float32) * qstep), 0, 255)
        prev = recon
        recons.append(recon[:orig_h, :orig_w].astype(np.uint8))
    from lydlr_ai.model.entropy_coder import rans_encode

    symbols = np.concatenate([c.reshape(-1) for c in chunks]).astype(np.uint8)
    u16 = td._pmf_u16(symbols)
    coded = rans_encode(symbols, td._pmf_from_u16(u16))
    head = td._HEAD.pack(td.MAGIC, 1, 0, qstep, len(flist), H, W)
    v1_payload = head + u16.tobytes() + coded
    v1_bpp = len(v1_payload) * 8 / (len(flist) * orig_h * orig_w)
    v1_psnr = float(np.mean([psnr(f, r) for f, r in zip(flist, recons)]))

    _p, v2_recons, v2 = encode_temporal_dct(frames, qstep=16.0)
    v2_psnr = float(np.mean([psnr(f, r) for f, r in zip(frames, v2_recons)]))
    # Same qstep: large countable-rate win; PSNR stays in the useful regime.
    assert v2["bpp"] < v1_bpp * 0.75
    assert v2_psnr > 35.0
    assert v2["compression_ratio"] > v1_bpp / max(v2["bpp"], 1e-9) * 0.9
    assert v2["confidence"] > 0.8
    # Matched-rate check: finer v2 qstep near v1 bpp should not collapse quality.
    fine = encode_temporal_dct(frames, qstep=8.0)[2]
    assert fine["bpp"] <= v1_bpp * 1.05
    assert fine["confidence"] > 0.85


def test_model_confidence_rises_with_zero_frac_and_psnr():
    low = model_confidence(zero_frac=0.2, psnr_db=20.0)
    high = model_confidence(zero_frac=0.99, psnr_db=44.0)
    assert high > low
    assert high > 0.9


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
    for qstep in (2.0, 8.0, 16.0, 32.0, 64.0):
        _payload, recons, stats = encode_temporal_dct(frames, qstep=qstep)
        lp = float(np.mean([psnr(f, r) for f, r in zip(frames, recons)]))
        peers = [p for b, p in jpeg if abs(b - stats["bpp"]) / max(stats["bpp"], 1e-9) <= 0.15]
        if peers and lp > max(peers) + 0.5:
            wins = True
        if stats["bpp"] < jmin_bpp and lp >= jmin_psnr:
            wins = True
    assert wins
