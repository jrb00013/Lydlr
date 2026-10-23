"""Tests for multimodal framing helpers (IMU delta, LiDAR downsample, LYMS)."""

from __future__ import annotations

import numpy as np
import pytest

from lydlr_ai.communication.modality_codec import (
    audio_to_mel,
    decode_imu_delta,
    downsample_lidar,
    encode_imu_delta,
    frame_multimodal_payload,
    split_multimodal_payload,
)


def test_imu_delta_round_trip():
    prev = None
    curr = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float32)
    payload, stored = encode_imu_delta(prev, curr)
    back = decode_imu_delta(None, payload)
    np.testing.assert_allclose(back, curr, rtol=1e-5)

    nxt = curr + np.array([0.5, 0, 0, 0, 0, 0], dtype=np.float32)
    payload2, _ = encode_imu_delta(stored, nxt)
    restored = decode_imu_delta(stored, payload2)
    np.testing.assert_allclose(restored, nxt, rtol=1e-5)


def test_downsample_lidar_reduces_points():
    pts = np.arange(1000, dtype=np.float32)
    light = downsample_lidar(pts, 0.2)
    heavy = downsample_lidar(pts, 0.9)
    assert heavy.size < light.size <= pts.size
    assert heavy.size >= 4


def test_audio_to_mel_fallback_shape():
    rng = np.random.default_rng(0)
    wave = rng.standard_normal(8000).astype(np.float32)
    mel = audio_to_mel(wave, n_mels=16)
    assert mel.dtype == np.float32
    assert mel.size == 16 * 16


def test_empty_audio_and_lidar():
    mel = audio_to_mel(np.array([], dtype=np.float32), n_mels=8)
    assert mel.size == 64
    assert downsample_lidar(np.array([], dtype=np.float32), 0.5).size == 0


def test_lyms_frame_round_trip():
    chunks = {
        "camera": b"cam-bytes",
        "imu": b"imu-bytes",
        "extra": b"x",
    }
    frame = frame_multimodal_payload(chunks)
    back = split_multimodal_payload(frame)
    assert back["camera"] == b"cam-bytes"
    assert back["imu"] == b"imu-bytes"
    assert back["extra"] == b"x"


def test_lyms_bad_magic():
    with pytest.raises(ValueError, match="bad LYMS magic"):
        split_multimodal_payload(b"NOPE" + b"\x00" * 8)
