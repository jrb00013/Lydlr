#!/usr/bin/env python3
# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
"""Lightweight host probe: fixtures, imports, and optional API ping."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2" / "src" / "lydlr_ai"))


def check_fixtures() -> dict:
    scripts = ROOT / "scripts"
    names = [
        "fixture_correlated_clip.npz",
        "fixture_drone_clip.npz",
        "fixture_iot_clip.npz",
        "fixture_warehouse_clip.npz",
    ]
    out = {}
    for name in names:
        path = scripts / name
        out[name] = {"exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0}
    return out


def check_imports() -> dict:
    results = {}
    for mod in (
        "lydlr_ai.model.entropy_coder",
        "lydlr_ai.model.temporal_residual",
        "lydlr_ai.model.temporal_rans",
        "lydlr_ai.model.temporal_dct",
        "lydlr_ai.communication.wire",
        "lydlr_ai.communication.link_policy",
        "lydlr_ai.utils.bitrate_math",
        "lydlr_ai.utils.clip_stats",
    ):
        try:
            __import__(mod)
            results[mod] = "ok"
        except Exception as exc:  # noqa: BLE001 — probe must never crash
            results[mod] = f"fail: {exc}"
    return results


def check_api(base: str, timeout: float = 2.0) -> dict:
    url = base.rstrip("/") + "/api/health/"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read(256).decode("utf-8", errors="replace")
            return {"ok": True, "status": resp.status, "body": body[:120]}
    except urllib.error.URLError as exc:
        return {"ok": False, "error": str(exc.reason if hasattr(exc, "reason") else exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="", help="Optional API base URL to ping")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    report = {
        "fixtures": check_fixtures(),
        "imports": check_imports(),
    }
    if args.api:
        report["api"] = check_api(args.api)

    import_ok = all(v == "ok" for v in report["imports"].values())
    fixtures_ok = all(v["exists"] for v in report["fixtures"].values())
    api_ok = report.get("api", {}).get("ok", True)
    report["healthy"] = bool(import_ok and fixtures_ok and api_ok)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("fixtures:", "ok" if fixtures_ok else "MISSING")
        for name, meta in report["fixtures"].items():
            print(f"  {name}: {'yes' if meta['exists'] else 'no'} ({meta['bytes']} B)")
        print("imports:", "ok" if import_ok else "FAIL")
        for mod, status in report["imports"].items():
            print(f"  {mod}: {status}")
        if "api" in report:
            print("api:", report["api"])
        print("healthy:", report["healthy"])

    return 0 if report["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
