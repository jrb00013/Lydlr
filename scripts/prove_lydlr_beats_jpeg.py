#!/usr/bin/env python3
# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
"""Prove the Lydlr vision codec beats intra JPEG at matched bpp.

Vision path: temporal residual + 8x8 DCT + rANS. Rate is countable payload
bytes (header included). No neural checkpoint is required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2" / "src" / "lydlr_ai"))

from lydlr_ai.model.bench_baselines import psnr, ssim  # noqa: E402
from lydlr_ai.model.temporal_dct import sweep_temporal_dct  # noqa: E402
from lydlr_ai.model.temporal_residual import encode_jpeg_intra  # noqa: E402


def _load(fixture: Path, n: int) -> tuple[np.ndarray, str]:
    data = np.load(fixture, allow_pickle=True)
    raw = data["frames"] if "frames" in data else data[data.files[0]]
    frames = np.asarray(raw, dtype=np.uint8)[:n]
    return frames, fixture.name


def _downsample(frames: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frames.shape[1:3]
    if w <= max_width:
        return frames
    import cv2

    new_w = max_width
    new_h = max(8, int(round(h * new_w / w)))
    out = []
    for f in frames:
        bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        small = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    return np.stack(out, axis=0)


def _jpeg_curve(frames: np.ndarray) -> list[dict]:
    h, w = frames.shape[1:3]
    rows = []
    for q in range(5, 96, 5):
        payloads, recons, _bpf = encode_jpeg_intra(frames, q)
        bpp = sum(8 * len(p) for p in payloads) / (len(payloads) * h * w)
        rows.append({
            "quality": q,
            "bpp": float(bpp),
            "psnr": float(np.mean([psnr(f, r) for f, r in zip(frames, recons)])),
            "ssim": float(np.mean([ssim(f, r) for f, r in zip(frames, recons)])),
        })
    return rows


def _jpeg_psnr_at(curve: list[dict], target_bpp: float) -> tuple[float, bool]:
    """Linear interpolate JPEG PSNR at target_bpp. False if outside the curve."""
    curve = sorted(curve, key=lambda r: r["bpp"])
    if target_bpp < curve[0]["bpp"] * 0.90 or target_bpp > curve[-1]["bpp"] * 1.10:
        # Still allow comparison against the nearest endpoint inside 10%.
        nearest = min(curve, key=lambda r: abs(r["bpp"] - target_bpp))
        rel = abs(nearest["bpp"] - target_bpp) / max(target_bpp, 1e-9)
        return float(nearest["psnr"]), rel <= 0.10
    for a, b in zip(curve, curve[1:]):
        if a["bpp"] <= target_bpp <= b["bpp"] or b["bpp"] <= target_bpp <= a["bpp"]:
            span = b["bpp"] - a["bpp"]
            t = 0.0 if span == 0 else (target_bpp - a["bpp"]) / span
            return float(a["psnr"] + t * (b["psnr"] - a["psnr"])), True
    nearest = min(curve, key=lambda r: abs(r["bpp"] - target_bpp))
    rel = abs(nearest["bpp"] - target_bpp) / max(target_bpp, 1e-9)
    return float(nearest["psnr"]), rel <= 0.10


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=ROOT / "scripts" / "fixture_correlated_clip.npz",
    )
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--max-width", type=int, default=96)
    parser.add_argument("--min-margin-db", type=float, default=1.0)
    parser.add_argument(
        "--publish",
        type=Path,
        default=None,
        help="Also write the proof JSON here (shipped with the console)",
    )
    parser.add_argument(
        "--check-published",
        type=Path,
        default=None,
        help="Fail if this file's matched margin drifts more than --margin-tol dB",
    )
    parser.add_argument("--margin-tol", type=float, default=2.0)
    args = parser.parse_args()

    frames, source = _load(args.fixture, args.frames)
    frames = _downsample(frames, args.max_width)
    jpeg = _jpeg_curve(frames)
    points = sweep_temporal_dct(frames)

    wins = []
    print(f"clip={source} shape={frames.shape}")
    print(f"{'qstep':>6} | {'bpp':>8} | {'psnr':>8} | {'jpeg_psnr':>9} | margin")
    for pt in points:
        recons = pt["recons"]
        psnr_l = float(np.mean([psnr(f, r) for f, r in zip(frames, recons)]))
        jpeg_psnr, matched = _jpeg_psnr_at(jpeg, pt["bpp"])
        margin = psnr_l - jpeg_psnr
        jpeg_min = min(jpeg, key=lambda r: r["bpp"])
        below_floor = (
            pt["bpp"] <= jpeg_min["bpp"]
            and psnr_l >= jpeg_min["psnr"] + args.min_margin_db
        )
        beat = (matched and margin >= args.min_margin_db) or below_floor
        kind = "matched" if matched and margin >= args.min_margin_db else (
            "below_floor" if below_floor else ""
        )
        if beat:
            wins.append({
                "qstep": pt["qstep"],
                "bpp": pt["bpp"],
                "psnr": psnr_l,
                "jpeg_psnr": jpeg_psnr if kind == "matched" else jpeg_min["psnr"],
                "margin_db": margin if kind == "matched" else psnr_l - jpeg_min["psnr"],
                "kind": kind,
            })
        flag = kind or ("near" if matched else "out-of-range")
        print(
            f"{pt['qstep']:6.1f} | {pt['bpp']:8.4f} | {psnr_l:8.2f} | "
            f"{jpeg_psnr:9.2f} | {margin:+5.2f} {flag}"
        )

    out = ROOT / "scripts" / "results" / "lydlr_beats_jpeg.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "success": bool(wins),
        "clip": source,
        "shape": list(frames.shape),
        "min_margin_db": args.min_margin_db,
        "wins": wins,
        "best_matched_margin_db": max(
            (w["margin_db"] for w in wins if w["kind"] == "matched"),
            default=None,
        ),
        "best_margin_db": max(
            (w["margin_db"] for w in wins if w["kind"] == "matched"),
            default=None,
        ),
    }
    text = json.dumps(payload, indent=2) + "\n"
    out.write_text(text, encoding="utf-8")
    if args.publish:
        args.publish.parent.mkdir(parents=True, exist_ok=True)
        args.publish.write_text(text, encoding="utf-8")
        print(f"published {args.publish}")
    print(
        f"wrote {out} success={payload['success']} "
        f"best_matched_margin_db={payload['best_matched_margin_db']}"
    )
    matched_wins = [w for w in wins if w["kind"] == "matched"]
    if not matched_wins:
        return 1
    if args.check_published:
        published = json.loads(args.check_published.read_text(encoding="utf-8"))
        fresh = float(payload["best_matched_margin_db"])
        shipped = float(published.get("best_matched_margin_db") or 0)
        drift = abs(fresh - shipped)
        print(f"published_margin={shipped:.3f} fresh={fresh:.3f} drift={drift:.3f}")
        if drift > args.margin_tol:
            print(f"published proof drifted more than {args.margin_tol} dB", file=sys.stderr)
            return 1
        if not published.get("success"):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
