#!/usr/bin/env python3
"""Compare heuristic vs RL compression controller across scenarios."""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "ros2" / "src" / "lydlr_ai"))

from lydlr_ai.model.rl_policy import RLCompressionController

SCENARIOS = [
    {"name": "nominal", "budget_ratio": 0.8, "quality": 0.85, "latency": 20},
    {"name": "burst", "budget_ratio": 1.5, "quality": 0.80, "latency": 25},
    {"name": "low_quality", "budget_ratio": 0.9, "quality": 0.50, "latency": 30},
    {"name": "high_latency", "budget_ratio": 1.0, "quality": 0.75, "latency": 80},
    {"name": "idle", "budget_ratio": 0.2, "quality": 0.95, "latency": 10},
    {"name": "congested", "budget_ratio": 2.0, "quality": 0.70, "latency": 60},
]


def run_bench(ctrl: RLCompressionController, label: str, steps: int = 50) -> list:
    rows = []
    for s in SCENARIOS:
        total_reward = 0.0
        adj_log = []
        for _ in range(steps):
            adj = ctrl.predict(
                budget_ratio=s["budget_ratio"],
                quality_score=s["quality"],
                latency_ms=s["latency"],
                quality_trend=s["quality"],
                cpu_load=0.5,
            )
            adj_log.append(adj)
            total_reward += ctrl.reward
        rows.append(
            {
                "controller": label,
                "scenario": s["name"],
                "budget_ratio": s["budget_ratio"],
                "avg_adjustment": round(float(np.mean(adj_log)), 4),
                "std_adjustment": round(float(np.std(adj_log)), 4),
                "total_reward": round(total_reward, 2),
            }
        )
    return rows


def main():
    rows = []

    heuristic = RLCompressionController(mode="heuristic")
    rows.extend(run_bench(heuristic, "heuristic"))

    try:
        rl = RLCompressionController(
            mode="ppo",
            model_path=_REPO / "ros2/src/lydlr_ai/models/rl_compression_policy.zip",
        )
        rows.extend(run_bench(rl, "ppo"))
    except (ImportError, FileNotFoundError) as e:
        print(f"RL model not available, skipping PPO bench: {e}", file=sys.stderr)

    dest = _REPO / "scripts/bench_rl_vs_heuristic_results.csv"
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {dest}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
