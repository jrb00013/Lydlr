#!/usr/bin/env python3
"""Record a short multimodal sensor clip to NPZ for sensor_ingest replay."""
import argparse
import math
from pathlib import Path

import numpy as np


PROFILES = {
    "drone": {"hz": 10.0, "img_size": 224, "lidar_points": 64, "audio_samples": 4096, "frames": 200},
    "iot": {"hz": 2.0, "img_size": 128, "lidar_points": 32, "audio_samples": 1024, "frames": 40},
}


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
        frames.append(
            {
                "image": img,
                "imu": np.array(
                    [0.15 * math.sin(t), 0.1 * math.cos(t), 9.81, 0.02, 0.01, 0.005],
                    dtype=np.float32,
                ),
                "lidar": (np.random.randn(p["lidar_points"]) * 0.5).astype(np.float32),
                "audio": (np.random.randn(p["audio_samples"]) * 0.01).astype(np.float32),
            }
        )
    return frames, p["hz"]


def main():
    parser = argparse.ArgumentParser(description="Record demo sensor clip for Lydlr replay")
    parser.add_argument("--vertical", choices=["drone", "iot"], default="drone")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "demo_clips",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"{args.vertical}_clip.npz"
    frames, hz = generate_clip(args.vertical)
    np.savez(dest, frames=np.array(frames, dtype=object), hz=hz, vertical=args.vertical)
    print(f"Wrote {len(frames)} frames @ {hz} Hz → {dest}")


if __name__ == "__main__":
    main()
