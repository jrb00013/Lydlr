"""Tests for multimodal codec (LYMS framing, audio mel, LiDAR, IMU delta)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_mod(name, rel_path):
    path = Path(__file__).resolve().parents[2] / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_imu_delta_roundtrip():
    mc = _load_mod("modality_codec", "ros2/src/lydlr_ai/lydlr_ai/communication/modality_codec.py")
    prev = None
    samples = [np.array([1, 2, 3, 0, 0, 0], dtype=np.float32), np.array([1.1, 2.2, 3.3, 0, 0, 0], dtype=np.float32)]
    decoded_prev = None
    for s in samples:
        delta, prev = mc.encode_imu_delta(prev, s)
        decoded = mc.decode_imu_delta(decoded_prev, delta)
        decoded_prev = decoded
        np.testing.assert_allclose(decoded, s, rtol=1e-5)


def test_lyms_frame_roundtrip():
    mc = _load_mod("modality_codec", "ros2/src/lydlr_ai/lydlr_ai/communication/modality_codec.py")
    chunks = {
        "camera": b"img-bytes",
        "lidar": b"lidar-bytes",
        "imu": b"imu-bytes",
        "audio": b"audio-bytes",
    }
    framed = mc.frame_multimodal_payload(chunks)
    out = mc.split_multimodal_payload(framed)
    assert out == chunks


def test_lidar_downsample():
    mc = _load_mod("modality_codec", "ros2/src/lydlr_ai/lydlr_ai/communication/modality_codec.py")
    pts = np.arange(100, dtype=np.float32)
    small = mc.downsample_lidar(pts, compression_level=0.9)
    large = mc.downsample_lidar(pts, compression_level=0.2)
    assert len(small) < len(large)


def test_audio_to_mel_shape():
    mc = _load_mod("modality_codec", "ros2/src/lydlr_ai/lydlr_ai/communication/modality_codec.py")
    wave = np.random.randn(4096).astype(np.float32)
    mel = mc.audio_to_mel(wave)
    assert mel.shape == (128 * 128,)
    assert mel.dtype == np.float32
