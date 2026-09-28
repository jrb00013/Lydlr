#!/usr/bin/env python3
# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# CLI rate–distortion harness over lydlr_ai.model.bench_baselines.

"""Matched-rate codec comparison on a clip (real NPZ or synthetic).

Usage:
  PYTHONPATH=ros2/src/lydlr_ai python scripts/bench_codecs.py --synthetic 8
  PYTHONPATH=ros2/src/lydlr_ai python scripts/bench_codecs.py \\
      --clip path/to/clip.npz --codecs jpeg,webp,h264,lydlr \\
      --target-bpps 0.05,0.1,0.2,0.5 --out scripts/results/rd_curve.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
_LYDLR_PKG = ROOT / "ros2" / "src" / "lydlr_ai"
if str(_LYDLR_PKG) not in sys.path:
    sys.path.insert(0, str(_LYDLR_PKG))

from lydlr_ai.model.bench_baselines import (  # noqa: E402
    Clip,
    LpipsProbe,
    build_registry,
    interp_rd,
    load_clip_npz,
    lydlr_rate_and_recon,
    match_rate,
    measure_point,
    sweep_codec,
    synthetic_markov_clip,
)

CODEC_ALIASES = {
    "jpeg": "jpeg420",
    "jpeg420": "jpeg420",
    "jpeg444": "jpeg444",
    "webp": "webp",
    "h264": "h264_intra",
    "h264_intra": "h264_intra",
    "lydlr": "lydlr",
}


def _parse_csv_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_codecs(raw: str) -> List[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def _resolve_registry_names(requested: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Return (baseline registry names, lydlr_requested)."""
    baseline: List[str] = []
    lydlr = False
    for c in requested:
        key = CODEC_ALIASES.get(c)
        if key is None:
            raise SystemExit(f"unknown codec {c!r}; known: {', '.join(sorted(CODEC_ALIASES))}")
        if key == "lydlr":
            lydlr = True
        else:
            baseline.append(key)
    return baseline, lydlr


def _lydlr_match_at_bpp(
    clip: Clip,
    target_bpp: float,
    *,
    tol: float,
    checkpoint: str,
    lpips: Optional[LpipsProbe],
) -> Dict[str, Any]:
    """Sweep target_quality and pick the operating point nearest target_bpp."""
    qualities = [round(q, 3) for q in _parse_csv_floats("0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95")]
    report = lydlr_rate_and_recon(
        clip.frames,
        side_inputs=clip.side_inputs,
        checkpoint=checkpoint,
        target_qualities=qualities,
        lpips_probe=lpips,
        latency=False,
    )
    row: Dict[str, Any] = {
        "codec": "lydlr",
        "target_bpp": target_bpp,
        "available": report.get("available", False),
        "reason": report.get("reason", ""),
        "trained": report.get("trained", False),
        "rate_source": report.get("rate_source"),
    }
    if not report.get("available"):
        row["matched"] = False
        return row

    points = report.get("points") or []
    if not points:
        row["matched"] = False
        row["reason"] = row["reason"] or "no lydlr points"
        return row

    def rel_err(bpp: float) -> float:
        return abs(bpp - target_bpp) / max(abs(target_bpp), 1e-9)

    best = min(points, key=lambda p: rel_err(float(p["bpp"])))
    rel = rel_err(float(best["bpp"]))
    row.update(
        {
            "matched": rel <= tol,
            "rel_err": rel,
            "bpp": float(best["bpp"]),
            "psnr_mean": float(best["psnr_mean"]),
            "ssim_mean": float(best["ssim_mean"]),
            "lpips_mean": best.get("lpips_mean"),
            "target_quality": float(best["target_quality"]),
            "bits_per_frame": float(best["bits"]),
            "rate_source": best.get("rate_source", report.get("rate_source")),
            "proxy_bits_per_frame": float(best.get("proxy_bits", 0.0)),
            "fixed_length_bits_per_frame": float(best.get("fixed_length_bits", 0.0)),
        }
    )
    if rel > tol:
        row["reason"] = f"nearest quality landed at rel_err={rel:.4f}"
    return row


def _baseline_at_bpp(
    codec_name: str,
    codec,
    clip: Clip,
    target_bpp: float,
    *,
    tol: float,
    lpips: Optional[LpipsProbe],
) -> Dict[str, Any]:
    m = match_rate(codec, clip.frames, target_bpp, tol=tol)
    row: Dict[str, Any] = {
        "codec": codec_name,
        "target_bpp": target_bpp,
        "matched": m.matched,
        "rel_err": m.rel_err,
        "bpp": m.bpp,
        "param": m.param,
        "param_label": codec.param_label,
        "bytes_total": m.bytes_total,
        "reason": m.reason,
    }
    if m.bytes_total <= 0:
        return row
    pt = measure_point(codec, m.param, clip.frames, lpips=lpips)
    row.update(
        {
            "psnr_mean": pt.psnr_mean,
            "ssim_mean": pt.ssim_mean,
            "lpips_mean": None if not math.isfinite(pt.lpips_mean) else pt.lpips_mean,
            "rate_measurement": pt.rate_measurement,
        }
    )
    return row


def run_benchmark(args: argparse.Namespace) -> dict:
    if args.clip:
        clip = load_clip_npz(args.clip, max_frames=args.max_frames, stride=args.stride)
    else:
        n = int(args.synthetic)
        clip = synthetic_markov_clip(
            frames=n,
            height=args.height,
            width=args.width,
            seed=args.seed,
        )

    baseline_names, want_lydlr = _resolve_registry_names(_parse_codecs(args.codecs))
    include_video = any(n.startswith("h264") or n.startswith("h265") for n in baseline_names)
    codecs, unavailable = build_registry(include=baseline_names or None, include_video=include_video)
    for name in baseline_names:
        if name not in codecs:
            unavailable[name] = unavailable.get(name, "not registered on this host")

    targets = _parse_csv_floats(args.target_bpps)
    lpips = LpipsProbe() if args.lpips else None

    rows: List[dict] = []
    for target in targets:
        for bname in baseline_names:
            if bname not in codecs:
                rows.append(
                    {
                        "codec": bname,
                        "target_bpp": target,
                        "matched": False,
                        "available": False,
                        "reason": unavailable.get(bname, "unavailable"),
                    }
                )
                continue
            rows.append(
                _baseline_at_bpp(
                    bname,
                    codecs[bname],
                    clip,
                    target,
                    tol=args.rate_tol,
                    lpips=lpips,
                )
            )
        if want_lydlr:
            rows.append(
                _lydlr_match_at_bpp(
                    clip,
                    target,
                    tol=args.rate_tol,
                    checkpoint=args.checkpoint or "",
                    lpips=lpips,
                )
            )

    curves: Dict[str, List[dict]] = {}
    if args.full_curves:
        for bname, codec in codecs.items():
            pts = sweep_codec(codec, clip.frames, max_points=args.curve_points, lpips=lpips)
            curves[bname] = [p.to_dict() for p in pts]
            for target in targets:
                hit = interp_rd(pts, target)
                if hit:
                    curves.setdefault(f"{bname}_interp", []).append(hit)

    report = {
        "clip": clip.describe(),
        "target_bpps": targets,
        "rate_tol": args.rate_tol,
        "unavailable": unavailable,
        "rows": rows,
    }
    if curves:
        report["curves"] = curves
    return report


def _markdown_table(rows: Sequence[dict]) -> str:
    headers = ("codec", "target_bpp", "matched", "bpp", "PSNR", "SSIM", "notes")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in rows:
        notes = r.get("reason") or r.get("rate_source") or r.get("rate_measurement") or ""
        if r.get("codec") == "lydlr" and r.get("trained") is False:
            notes = (notes + "; random weights").strip("; ")
        psnr = r.get("psnr_mean")
        ssim = r.get("ssim_mean")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r.get("codec", "")),
                    f"{float(r.get('target_bpp', 0)):.3f}",
                    str(r.get("matched", "")),
                    f"{float(r.get('bpp', float('nan'))):.4f}" if r.get("bpp") is not None else "",
                    f"{float(psnr):.2f}" if psnr is not None else "",
                    f"{float(ssim):.4f}" if ssim is not None else "",
                    str(notes).replace("|", "\\|")[:80],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Matched-rate RD benchmark (see BENCHMARK_PROTOCOL.md)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--clip", type=str, help="NPZ clip with object-array 'frames' (allow_pickle)")
    src.add_argument("--synthetic", type=int, metavar="N", help="Structured Markov synthetic clip length")
    p.add_argument(
        "--codecs",
        default="jpeg,webp",
        help="Comma-separated: jpeg, webp, h264, lydlr (default: jpeg,webp)",
    )
    p.add_argument(
        "--target-bpps",
        default="0.05,0.1,0.2,0.5",
        help="Comma-separated target bits-per-pixel values",
    )
    p.add_argument(
        "--out",
        default=str(ROOT / "scripts" / "results" / "rd_curve.json"),
        help="JSON output path",
    )
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rate-tol", type=float, default=0.10, help="Relative rate match tolerance")
    p.add_argument("--checkpoint", type=str, default="", help="Lydlr weights for honest codec claims")
    p.add_argument("--lpips", action="store_true", help="Enable LPIPS (requires torch)")
    p.add_argument("--full-curves", action="store_true", help="Also sweep full RD curves into JSON")
    p.add_argument("--curve-points", type=int, default=13)
    args = p.parse_args()

    report = run_benchmark(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(_markdown_table(report["rows"]))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
