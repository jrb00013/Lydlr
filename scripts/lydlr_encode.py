#!/usr/bin/env python3
# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
"""Encode an RGB clip with the Lydlr vision codec and print countable rate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2" / "src" / "lydlr_ai"))

from lydlr_ai.model.bench_baselines import psnr  # noqa: E402
from lydlr_ai.model.temporal_dct import decode_temporal_dct, encode_temporal_dct  # noqa: E402


def _load(path: Path, n: int) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    raw = data["frames"] if "frames" in data.files else data[data.files[0]]
    return np.asarray(raw, dtype=np.uint8)[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--qstep", type=float, default=16.0)
    parser.add_argument("--out", type=Path, default=None, help="Write the LDCT payload")
    args = parser.parse_args()

    frames = _load(args.clip, args.frames)
    payload, recons, stats = encode_temporal_dct(frames, qstep=args.qstep)
    back = decode_temporal_dct(payload)
    if any(not np.array_equal(a, b) for a, b in zip(recons, back)):
        print("decode mismatch", file=sys.stderr)
        return 1
    stats["psnr"] = float(np.mean([psnr(f, r) for f, r in zip(frames, recons)]))
    if args.out:
        args.out.write_bytes(payload)
        stats["wrote"] = str(args.out)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
