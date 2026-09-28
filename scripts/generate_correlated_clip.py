#!/usr/bin/env python3
# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
"""Generate a temporally correlated RGB clip fixture for beat-JPEG benchmarks.

Unlike scripts/fixture_drone_clip.npz (independent multimodal samples), this
writes consecutive video frames so residual coding can beat intra JPEG.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2" / "src" / "lydlr_ai"))

from lydlr_ai.model.temporal_residual import synthetic_correlated_clip  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--height", type=int, default=96)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "scripts" / "fixture_correlated_clip.npz",
    )
    args = p.parse_args()

    frames = np.asarray(
        synthetic_correlated_clip(
            n=args.frames, h=args.height, w=args.width, seed=args.seed
        ),
        dtype=np.uint8,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        frames=frames,
        hz=np.array(10.0),
        kind=np.array("correlated_video"),
        seed=np.array(args.seed),
    )
    mad = float(
        np.mean(
            np.abs(
                frames[1:].astype(np.float32) - frames[:-1].astype(np.float32)
            )
        )
    )
    print(
        f"wrote {args.out} shape={frames.shape} bytes={args.out.stat().st_size} mad={mad:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
