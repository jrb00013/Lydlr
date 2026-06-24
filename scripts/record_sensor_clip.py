#!/usr/bin/env python3
"""Record a short multimodal sensor clip to NPZ for sensor_ingest replay."""
import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_audio_to_mel():
    path = _REPO_ROOT / "ros2/src/lydlr_ai/lydlr_ai/communication/modality_codec.py"
    spec = importlib.util.spec_from_file_location("modality_codec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.audio_to_mel


audio_to_mel = _load_audio_to_mel()


PROFILES = {
    "drone": {"hz": 10.0, "img_size": 224, "lidar_points": 64, "audio_samples": 4096, "frames": 200},
    "iot": {"hz": 2.0, "img_size": 128, "lidar_points": 32, "audio_samples": 1024, "frames": 40},
    "warehouse": {"hz": 5.0, "img_size": 160, "lidar_points": 96, "audio_samples": 2048, "frames": 100},
}


def synth_audio_wave(tick: int, n_samples: int) -> np.ndarray:
    t = np.linspace(0, 1, n_samples, dtype=np.float32)
    wave = 0.02 * np.sin(2 * math.pi * (220 + 30 * math.sin(tick * 0.1)) * t)
    wave += 0.005 * np.random.randn(n_samples).astype(np.float32)
    return wave.astype(np.float32)


def generate_clip(vertical: str):
    p = PROFILES[vertical]
    h = w = p["img_size"]
    frames = []
    for i in range(p["frames"]):
        t = i / p["hz"]
        img = np.clip(
            128 + 35 * np.sin(t) + 20 * np.random.randn(h, w, 3),
            0,
            255,
        ).astype(np.uint8)
        wave = synth_audio_wave(i, p["audio_samples"])
        frames.append(
            {
                "image": img,
                "imu": np.array(
                    [0.15 * math.sin(t), 0.1 * math.cos(t), 9.81, 0.02, 0.01, 0.005],
                    dtype=np.float32,
                ),
                "lidar": (np.random.randn(p["lidar_points"]) * 0.5).astype(np.float32),
                "audio_wave": wave,
                "audio": audio_to_mel(wave),
            }
        )
    return frames, p["hz"]


def main():
    parser = argparse.ArgumentParser(description="Record demo sensor clip for Lydlr replay")
    parser.add_argument("--vertical", choices=list(PROFILES.keys()), default="drone")
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "data" / "demo_clips",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"{args.vertical}_clip.npz"
    frames, hz = generate_clip(args.vertical)
    np.savez(dest, frames=np.array(frames, dtype=object), hz=hz, vertical=args.vertical)
    print(f"Wrote {len(frames)} frames @ {hz} Hz → {dest}")


if __name__ == "__main__":
    main()
