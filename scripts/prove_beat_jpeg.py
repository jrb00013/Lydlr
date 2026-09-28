#!/usr/bin/env python3
# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
"""Prove temporal JPEG residual beats intra JPEG on correlated clip average bpp."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2" / "src" / "lydlr_ai"))

from lydlr_ai.model.bench_baselines import psnr, ssim  # noqa: E402
from lydlr_ai.model.temporal_residual import (  # noqa: E402
    encode_jpeg_intra,
    encode_jpeg_residual,
    match_rate_beat_jpeg,
    synthetic_correlated_clip,
)


def _mean_abs_delta(frames: np.ndarray) -> float:
    if len(frames) < 2:
        return 0.0
    diffs = [
        float(np.mean(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))))
        for i in range(1, len(frames))
    ]
    return float(np.mean(diffs))


def _load_frames(fixture: Path, n: int, *, force_synthetic: bool = False) -> tuple[np.ndarray, str]:
    """Load clip frames. Fall back to synthetic when fixture is not temporally correlated.

    The bundled drone/iot/warehouse fixtures store independent multimodal samples, not
    consecutive video. Residual coding only beats JPEG when frames are correlated, so
    we detect high mean |Δpixel| and switch to synthetic_correlated_clip (locksmith axis).
    """
    if force_synthetic or not fixture.is_file():
        return synthetic_correlated_clip(n=n), "synthetic_correlated"

    data = np.load(fixture, allow_pickle=True)
    if "frames" in data:
        raw = data["frames"]
    else:
        key = list(data.files)[0]
        raw = data[key]
    raw = np.asarray(raw, dtype=object)[:n]
    if raw.ndim == 1 and raw.size and isinstance(raw[0], dict):
        imgs = [np.asarray(entry["image"], dtype=np.uint8) for entry in raw]
        frames = np.stack(imgs, axis=0)
    else:
        frames = np.asarray(raw)
        if frames.ndim != 4:
            raise ValueError(f"unsupported fixture frame layout: {frames.shape}")
        frames = frames.astype(np.uint8)

    # Independent samples typically have mean |Δ| ≫ 15; correlated video is usually < 8.
    mad = _mean_abs_delta(frames)
    if mad > 12.0:
        syn = synthetic_correlated_clip(n=n)
        return syn, f"synthetic_correlated (fixture {fixture.name} mad={mad:.1f} uncorrelated)"
    return frames, f"fixture:{fixture.name}"


def _downsample(frames: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frames.shape[1:3]
    if w <= max_width:
        return frames
    try:
        import cv2

        new_w = max_width
        new_h = max(1, int(round(h * new_w / w)))
        out = []
        for f in frames:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            small = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            out.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        return np.stack(out, axis=0)
    except Exception:
        from PIL import Image

        new_w = max_width
        new_h = max(1, int(round(h * new_w / w)))
        out = []
        for f in frames:
            im = Image.fromarray(f).resize((new_w, new_h), Image.Resampling.BILINEAR)
            out.append(np.asarray(im, dtype=np.uint8))
        return np.stack(out, axis=0)


def _row_metrics(frames: np.ndarray, quality: int, codec: str) -> dict:
    if codec == "jpeg_intra":
        payloads, recons, bpf = encode_jpeg_intra(frames, quality)
    else:
        payloads, recons, bpf = encode_jpeg_residual(frames, quality)
    h, w = frames.shape[1:3]
    bpp = sum(8 * len(p) for p in payloads) / (len(payloads) * h * w)
    psnrs = [psnr(f, r) for f, r in zip(frames, recons)]
    ssims = [ssim(f, r) for f, r in zip(frames, recons)]
    return {
        "codec": codec,
        "quality": quality,
        "bpp": float(bpp),
        "bits_per_frame": float(bpf),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prove residual JPEG beats intra on clip bpp")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=ROOT / "scripts" / "fixture_correlated_clip.npz",
        help="Temporally correlated NxHxWx3 clip (default: fixture_correlated_clip.npz)",
    )
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--max-width", type=int, default=320)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force in-memory synthetic_correlated_clip (ignore fixture)",
    )
    args = parser.parse_args()

    frames, source = _load_frames(args.fixture, args.frames, force_synthetic=args.synthetic)
    frames = _downsample(frames, args.max_width)
    print(f"clip_source={source} shape={frames.shape} mad={_mean_abs_delta(frames):.2f}")

    report = match_rate_beat_jpeg(frames)

    table_rows: list[dict] = []
    for pt in report["residual_operating_points"]:
        q = pt["residual_quality"]
        res_m = _row_metrics(frames, q, "jpeg_residual")
        jq = pt["matched_jpeg_quality"]
        jpeg_m = _row_metrics(frames, jq, "jpeg_intra")
        bpp_rel = abs(res_m["bpp"] - jpeg_m["bpp"]) / max(jpeg_m["bpp"], 1e-9)
        beats = (
            res_m["psnr"] > jpeg_m["psnr"]
            and bpp_rel <= 0.10
        )
        table_rows.append({**res_m, "beats_jpeg": beats, "matched_jpeg_bpp": jpeg_m["bpp"]})
        table_rows.append({**jpeg_m, "beats_jpeg": False, "matched_jpeg_bpp": jpeg_m["bpp"]})

    print(f"{'codec':<14} | {'bpp':>8} | {'psnr':>8} | {'ssim':>8} | beats_jpeg?")
    print("-" * 55)
    seen: set[tuple] = set()
    for row in sorted(table_rows, key=lambda r: (r["codec"], r["quality"])):
        key = (row["codec"], row["quality"])
        if key in seen:
            continue
        seen.add(key)
        bj = "yes" if row.get("beats_jpeg") else "no"
        print(
            f"{row['codec']:<14} | {row['bpp']:8.4f} | {row['psnr']:8.3f} | "
            f"{row['ssim']:8.4f} | {bj}"
        )

    proof_points = [
        r for r in table_rows
        if r["codec"] == "jpeg_residual" and r.get("beats_jpeg")
    ]
    success = len(proof_points) > 0

    out_dir = ROOT / "scripts" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "beat_jpeg_proof.json"
    best_margin = None
    if proof_points:
        # PSNR margin vs matched jpeg at same residual quality row
        margins = []
        for r in proof_points:
            jpeg_peers = [
                j for j in table_rows
                if j["codec"] == "jpeg_intra" and abs(j["bpp"] - r["bpp"]) / max(r["bpp"], 1e-9) <= 0.10
            ]
            if jpeg_peers:
                margins.append(r["psnr"] - max(j["psnr"] for j in jpeg_peers))
        best_margin = float(max(margins)) if margins else None

    payload = {
        "success": success,
        "clip_source": source,
        "best_psnr_margin_db": best_margin,
        "n_proof_points_within_10pct_bpp": len(proof_points),
        "match_rate_report": report,
        "table_sample": [
            r for r in table_rows if r["codec"] == "jpeg_residual"
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
