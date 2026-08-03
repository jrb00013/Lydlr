#!/usr/bin/env python3
"""Smoke the realtime visualization control-plane path."""
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.getenv("LYDLR_API_URL", "http://127.0.0.1:8000").rstrip("/")


def get(path: str):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=5) as resp:
        return resp.status, resp.read()


def post(path: str, body: dict | None = None):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        return resp.status, resp.read()


def main() -> int:
    checks = []
    try:
        status, _ = get("/api/health/")
        checks.append(("health", status == 200))
    except Exception as exc:
        print(f"FAIL health: {exc}")
        return 1

    try:
        status, raw = post("/api/demo/pulse/")
        checks.append(("demo_pulse", status == 200))
        payload = json.loads(raw.decode())
        node = (payload.get("nodes") or ["node_0"])[0]
    except Exception as exc:
        print(f"FAIL demo pulse: {exc}")
        return 1

    try:
        status, body = get(f"/api/nodes/{node}/preview.jpg?side=reconstructed")
        checks.append(("preview_jpeg", status == 200 and body[:2] == b"\xff\xd8"))
    except Exception as exc:
        print(f"FAIL preview jpeg: {exc}")
        checks.append(("preview_jpeg", False))

    try:
        status, raw = get(f"/api/nodes/{node}/topics/")
        data = json.loads(raw.decode())
        checks.append(("topics", status == 200 and len(data.get("topics", [])) >= 3))
    except Exception as exc:
        print(f"FAIL topics: {exc}")
        checks.append(("topics", False))

    try:
        status, raw = get("/api/metrics/?limit=3")
        rows = json.loads(raw.decode())
        checks.append(("metrics", status == 200 and isinstance(rows, list) and len(rows) > 0))
    except Exception as exc:
        print(f"FAIL metrics: {exc}")
        checks.append(("metrics", False))

    ok = all(v for _, v in checks)
    for name, passed in checks:
        print(f"{'OK' if passed else 'FAIL':4} {name}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
