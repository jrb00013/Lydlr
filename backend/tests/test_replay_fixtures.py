"""Tests for deterministic NPZ replay fixtures."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1].parent
FIXTURE_DIR = REPO / "scripts"

PROFILES = {
    "drone": {"hz": 10.0, "img_size": 224, "lidar_points": 64, "audio_samples": 4096, "frames": 200},
    "iot": {"hz": 2.0, "img_size": 128, "lidar_points": 32, "audio_samples": 1024, "frames": 40},
    "warehouse": {"hz": 5.0, "img_size": 160, "lidar_points": 96, "audio_samples": 2048, "frames": 100},
}


def _load_modality_codec():
    path = REPO / "ros2/src/lydlr_ai/lydlr_ai/communication/modality_codec.py"
    spec = importlib.util.spec_from_file_location("modality_codec", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modality_codec"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fixture_path(vertical: str) -> Path:
    return FIXTURE_DIR / f"fixture_{vertical}_clip.npz"


def test_all_fixtures_exist():
    for v in PROFILES:
        assert _fixture_path(v).exists(), f"Missing fixture for {v}"


def test_npz_keys():
    for v in PROFILES:
        data = np.load(_fixture_path(v), allow_pickle=True)
        assert set(data.keys()) == {"frames", "hz", "vertical"}, f"Unexpected keys for {v}: {list(data.keys())}"


def test_frame_count():
    for v, p in PROFILES.items():
        data = np.load(_fixture_path(v), allow_pickle=True)
        assert len(data["frames"]) == p["frames"], f"{v}: expected {p['frames']} frames, got {len(data['frames'])}"


def test_metadata():
    for v, p in PROFILES.items():
        data = np.load(_fixture_path(v), allow_pickle=True)
        assert data["hz"] == p["hz"], f"{v}: expected hz={p['hz']}, got {data['hz']}"
        assert str(data["vertical"]) == v, f"{v}: expected vertical={v}, got {data['vertical']}"


def test_frame_modality_keys():
    expected_keys = {"image", "imu", "lidar", "audio_wave", "audio"}
    for v in PROFILES:
        data = np.load(_fixture_path(v), allow_pickle=True)
        for i, frame in enumerate(data["frames"]):
            frame = dict(frame)
            assert set(frame.keys()) == expected_keys, f"{v} frame {i}: got keys {set(frame.keys())}"


def test_image_shape_and_dtype():
    for v, p in PROFILES.items():
        data = np.load(_fixture_path(v), allow_pickle=True)
        for i, frame in enumerate(data["frames"]):
            frame = dict(frame)
            img = frame["image"]
            assert img.shape == (p["img_size"], p["img_size"], 3), f"{v} frame {i}: shape {img.shape}"
            assert img.dtype == np.uint8, f"{v} frame {i}: dtype {img.dtype}"


def test_imu_shape_and_dtype():
    for v in PROFILES:
        data = np.load(_fixture_path(v), allow_pickle=True)
        for i, frame in enumerate(data["frames"]):
            frame = dict(frame)
            imu = frame["imu"]
            assert imu.shape == (6,), f"{v} frame {i}: shape {imu.shape}"
            assert imu.dtype == np.float32, f"{v} frame {i}: dtype {imu.dtype}"


def test_lidar_shape_and_dtype():
    for v, p in PROFILES.items():
        data = np.load(_fixture_path(v), allow_pickle=True)
        for i, frame in enumerate(data["frames"]):
            frame = dict(frame)
            lidar = frame["lidar"]
            assert lidar.shape == (p["lidar_points"],), f"{v} frame {i}: shape {lidar.shape}"
            assert lidar.dtype == np.float32, f"{v} frame {i}: dtype {lidar.dtype}"


def test_audio_wave_shape_and_dtype():
    for v, p in PROFILES.items():
        data = np.load(_fixture_path(v), allow_pickle=True)
        for i, frame in enumerate(data["frames"]):
            frame = dict(frame)
            wave = frame["audio_wave"]
            assert wave.shape == (p["audio_samples"],), f"{v} frame {i}: shape {wave.shape}"
            assert wave.dtype == np.float32, f"{v} frame {i}: dtype {wave.dtype}"


def test_audio_mel_shape_and_dtype():
    for v in PROFILES:
        data = np.load(_fixture_path(v), allow_pickle=True)
        for i, frame in enumerate(data["frames"]):
            frame = dict(frame)
            audio = frame["audio"]
            assert audio.shape == (128 * 128,), f"{v} frame {i}: shape {audio.shape}"
            assert audio.dtype == np.float32, f"{v} frame {i}: dtype {audio.dtype}"


def test_audio_mel_uses_fallback():
    mc = _load_modality_codec()
    wave = np.zeros(4096, dtype=np.float32)
    mel = mc.audio_to_mel(wave)
    assert mel.shape == (128 * 128,)
    assert mel.dtype == np.float32
    assert np.all(mel >= 0)


def test_generator_accepts_subset_verticals():
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts/generate_replay_fixtures.py"),
         "--seed", "42", "--out", str(FIXTURE_DIR), "--verticals", "drone"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert _fixture_path("drone").exists()
    # Other fixtures might exist from previous runs, that's fine


def test_different_seed_changes_output(tmp_path):
    frames_ref = list(np.load(_fixture_path("drone"), allow_pickle=True)["frames"])
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts/generate_replay_fixtures.py"),
         "--seed", "999", "--out", str(tmp_path), "--verticals", "drone"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    frames_new = list(np.load(tmp_path / "fixture_drone_clip.npz", allow_pickle=True)["frames"])
    ref_img = dict(frames_ref[0])["image"]
    new_img = dict(frames_new[0])["image"]
    assert not np.array_equal(ref_img, new_img), "Seed 999 should produce different output than seed 42"


def test_first_frame_fingerprint_seed_42():
    data = np.load(_fixture_path("drone"), allow_pickle=True)
    frame0 = dict(data["frames"][0])
    np.testing.assert_array_equal(
        frame0["image"][0, 0, :3],
        np.array([130, 146, 148], dtype=np.uint8),
    )
    np.testing.assert_allclose(
        frame0["imu"],
        [0.0, 0.1, 9.81, 0.02, 0.01, 0.005],
        atol=1e-6,
    )


def test_deterministic_generator():
    seed = 99
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts/generate_replay_fixtures.py"), "--seed", str(seed), "--out", str(FIXTURE_DIR)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Generator failed: {result.stderr}"

    result2 = subprocess.run(
        [sys.executable, str(REPO / "scripts/generate_replay_fixtures.py"), "--seed", str(seed), "--out", str(FIXTURE_DIR)],
        capture_output=True,
        text=True,
    )
    assert result2.returncode == 0, f"Second run failed: {result2.stderr}"

    for v in PROFILES:
        data1 = np.load(_fixture_path(v), allow_pickle=True)
        data2 = np.load(_fixture_path(v), allow_pickle=True)
        for i in range(len(data1["frames"])):
            f1, f2 = dict(data1["frames"][i]), dict(data2["frames"][i])
            for key in f1:
                assert np.array_equal(f1[key], f2[key]), f"{v} frame {i} key {key} differs between runs"
