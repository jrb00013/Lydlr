#!/usr/bin/env python3
"""
Benchmark link policy decisions across drone/IoT/high-cap uplinks.

Usage:
  python scripts/bench_link_policy.py                                  # stdout CSV
  python scripts/bench_link_policy.py --csv scripts/bench_link_policy_golden.csv

Output columns:
  scenario,vertical,budget_kbps,est_output_kbps,quality_score,
  latency_ms,target_level,vision_frame_skip,over_budget
"""
import argparse
import csv
import importlib.util
import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _load_link_policy():
    path = REPO / "ros2/src/lydlr_ai/lydlr_ai/communication/link_policy.py"
    spec = importlib.util.spec_from_file_location("link_policy", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["link_policy"] = mod
    spec.loader.exec_module(mod)
    return mod


SCENARIOS = [
    # (name, vertical, budget_kbps, est_output_kbps_range, quality_range, latency_ms_range, ingest_hz)
    ("drone_nominal", "drone", 512, (200, 400), (0.75, 0.95), (5, 30), 10),
    ("drone_burst", "drone", 512, (600, 900), (0.6, 0.85), (10, 40), 10),
    ("drone_low_quality", "drone", 512, (300, 500), (0.4, 0.7), (15, 50), 10),
    ("drone_high_latency", "drone", 512, (250, 450), (0.7, 0.9), (60, 120), 10),
    ("iot_nominal", "iot", 64, (20, 50), (0.65, 0.90), (10, 40), 2),
    ("iot_burst", "iot", 64, (70, 120), (0.5, 0.75), (15, 60), 2),
    ("iot_low_quality", "iot", 64, (30, 60), (0.35, 0.6), (20, 70), 2),
    ("highcap_nominal", "drone", 2048, (800, 1800), (0.8, 0.95), (3, 20), 30),
    ("highcap_burst", "drone", 2048, (2200, 3500), (0.7, 0.9), (5, 25), 30),
]


def run_scenario(lp, scenario, steps=20):
    name, vertical, budget_kbps, out_range, qual_range, lat_range, ingest_hz = scenario
    policy = lp.NodeLinkPolicy.from_dict(
        "bench_node",
        {
            "vertical": vertical,
            "uplink_budget_kbps": budget_kbps,
        },
    )
    rows = []
    for i in range(steps):
        t = i / max(steps - 1, 1)
        est_out = out_range[0] + (out_range[1] - out_range[0]) * t
        quality = qual_range[0] + (qual_range[1] - qual_range[0]) * (0.5 + 0.5 * (t - 0.5))
        latency = lat_range[0] + (lat_range[1] - lat_range[0]) * (0.3 + 0.7 * (t % 0.7))
        quality = max(0.01, min(0.99, quality))

        level = lp.target_compression_level(
            policy,
            estimated_output_kbps=est_out,
            quality_score=quality,
            latency_ms=latency,
        )
        skip = lp.vision_frame_skip(policy, ingest_hz)
        over_budget = "yes" if est_out > budget_kbps else "no"

        rows.append({
            "scenario": name,
            "vertical": vertical,
            "budget_kbps": budget_kbps,
            "est_output_kbps": round(est_out, 1),
            "quality_score": round(quality, 3),
            "latency_ms": round(latency, 1),
            "target_level": level,
            "vision_frame_skip": skip,
            "over_budget": over_budget,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Benchmark link policy decisions")
    ap.add_argument("--csv", type=str, default="", help="Write CSV to file")
    args = ap.parse_args()

    lp = _load_link_policy()
    all_rows = []
    for scenario in SCENARIOS:
        all_rows.extend(run_scenario(lp, scenario))

    fieldnames = [
        "scenario", "vertical", "budget_kbps", "est_output_kbps",
        "quality_score", "latency_ms", "target_level",
        "vision_frame_skip", "over_budget",
    ]
    out = Path(args.csv) if args.csv else sys.stdout
    if isinstance(out, Path):
        out.parent.mkdir(parents=True, exist_ok=True)
        fh = out.open("w", newline="")
    else:
        fh = out  # type: ignore

    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

    if isinstance(out, Path):
        fh.close()
        print(f"Wrote {len(all_rows)} rows to {out}", file=sys.stderr)

    # Brief analysis
    over_count = sum(1 for r in all_rows if r["over_budget"] == "yes")
    avg_level = sum(r["target_level"] for r in all_rows) / max(len(all_rows), 1)
    print(
        f"Scenarios: {len(SCENARIOS)} | Rows: {len(all_rows)} | "
        f"Over-budget: {over_count} | Avg target level: {avg_level:.3f}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
