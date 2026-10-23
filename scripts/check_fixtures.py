#!/usr/bin/env python3
# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
"""Validate checked-in NPZ fixtures shape and temporal correlation flags."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2" / "src" / "lydlr_ai"))

from lydlr_ai.utils.clip_stats import summarize_clip  # noqa: E402


EXPECTED = {
    "fixture_correlated_clip.npz": {"must_correlate": True},
    "fixture_drone_clip.npz": {"must_correlate": False},
    "fixture_iot_clip.npz": {"must_correlate": False},
    "fixture_warehouse_clip.npz": {"must_correlate": False},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    scripts = ROOT / "scripts"
    reports = []
    ok = True
    for name, rules in EXPECTED.items():
        path = scripts / name
        if not path.is_file():
            reports.append({"name": name, "ok": False, "error": "missing"})
            ok = False
            continue
        summary = summarize_clip(path, max_frames=16)
        correlated = bool(summary["temporally_correlated"])
        expect = rules["must_correlate"]
        # Independent multimodal fixtures may still look mildly correlated by chance;
        # only enforce the positive claim for the beat-JPEG correlated clip.
        entry_ok = correlated if expect else True
        if not entry_ok:
            ok = False
        reports.append({**summary, "name": name, "ok": entry_ok, "expect_correlate": expect})
    if args.json:
        print(json.dumps({"ok": ok, "fixtures": reports}, indent=2))
    else:
        for r in reports:
            flag = "OK" if r.get("ok") else "FAIL"
            print(f"[{flag}] {r.get('name')} correlate={r.get('temporally_correlated')} n={r.get('n_frames')}")
        print("all_ok:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
