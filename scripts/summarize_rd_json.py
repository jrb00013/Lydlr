#!/usr/bin/env python3
# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
"""Pretty-print a bench_codecs / RD JSON result for PR comments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(data.get("points"), list):
        return data["points"]
    if isinstance(data.get("results"), list):
        return data["results"]
    if isinstance(data.get("codecs"), dict):
        rows = []
        for codec, payload in data["codecs"].items():
            if isinstance(payload, list):
                for p in payload:
                    rows.append({"codec": codec, **p})
            elif isinstance(payload, dict):
                rows.append({"codec": codec, **payload})
        return rows
    # flat list at top level
    if isinstance(data, list):
        return data
    return []


def render_markdown(data: Dict[str, Any]) -> str:
    rows = _rows(data)
    lines = ["| codec | bpp | PSNR | SSIM | matched |", "|---|---:|---:|---:|:---:|"]
    for row in rows:
        codec = row.get("codec") or row.get("name") or "?"
        bpp = row.get("bpp") or row.get("bits_per_pixel") or row.get("rate_bpp")
        psnr = row.get("psnr")
        ssim = row.get("ssim")
        matched = row.get("matched", "")
        lines.append(
            f"| {codec} | {_fmt(bpp)} | {_fmt(psnr)} | {_fmt(ssim)} | {matched} |"
        )
    if len(lines) == 2:
        lines.append("| _(no points)_ |  |  |  |  |")
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if v is None or v == "":
        return "—"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to RD JSON")
    parser.add_argument("--json", action="store_true", help="Echo normalized rows as JSON")
    args = parser.parse_args()
    data = json.loads(args.path.read_text())
    rows = _rows(data if isinstance(data, dict) else {"points": data})
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(render_markdown(data if isinstance(data, dict) else {"points": data}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
