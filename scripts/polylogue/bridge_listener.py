#!/usr/bin/env python3
"""Persistent polylogue bridge listener — auto-resolves room + participant id."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

import websockets

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from resolve import resolve_bridge_context  # noqa: E402

DEFAULT_BRIDGE_HOST = os.environ.get("POLYLOGUE_BRIDGE_HOST", "127.0.0.1")
DEFAULT_BRIDGE_PORT = os.environ.get("POLYLOGUE_BRIDGE_PORT", "7850")


def _paths(participant_id: str) -> tuple[Path, Path]:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in participant_id)
    base = Path(os.environ.get("POLYLOGUE_BRIDGE_STATE_DIR", "/tmp"))
    return base / f"polylogue-bridge-{safe}.log", base / f"polylogue-bridge-{safe}-outbox.jsonl"


def _log(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
    print(line, flush=True)


async def _outbox_reader(
    ws: websockets.WebSocketClientProtocol,
    outbox: Path,
    log_path: Path,
) -> None:
    outbox.parent.mkdir(parents=True, exist_ok=True)
    if not outbox.exists():
        outbox.touch()
    offset = 0
    while True:
        await asyncio.sleep(0.25)
        try:
            raw = outbox.read_text(encoding="utf-8")
        except OSError:
            continue
        if len(raw) <= offset:
            continue
        for line in raw[offset:].splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                _log(log_path, f"outbox skip bad json: {line[:80]}")
                continue
            if payload.get("type") != "message":
                payload = {"type": "message", "text": str(payload.get("text", line))}
            await ws.send(json.dumps(payload))
            _log(log_path, f"sent: {json.dumps(payload, ensure_ascii=False)}")
        offset = len(raw)


async def _listen(ws: websockets.WebSocketClientProtocol, log_path: Path) -> None:
    async for raw in ws:
        _log(log_path, f"recv: {raw}")


async def run_session(room: str, participant_id: str, bridge_url: str, log_path: Path, outbox: Path) -> None:
    uri = f"{bridge_url.rstrip('/')}/ws?room={quote(room)}&id={quote(participant_id)}"
    async with websockets.connect(uri) as ws:
        _log(log_path, f"connected room={room} id={participant_id}")
        await asyncio.gather(_listen(ws, log_path), _outbox_reader(ws, outbox, log_path))


async def main_async(room: str, participant_id: str, bridge_url: str, log_path: Path, outbox: Path) -> None:
    while True:
        try:
            await run_session(room, participant_id, bridge_url, log_path, outbox)
        except (websockets.ConnectionClosed, OSError) as exc:
            _log(log_path, f"disconnected: {exc} — reconnecting in 2s")
            await asyncio.sleep(2)


def main() -> int:
    p = argparse.ArgumentParser(description="Polylogue bridge listener")
    p.add_argument("--cwd", type=Path, default=None)
    p.add_argument("--room", default=None)
    p.add_argument("--id", dest="participant_id", default=None)
    p.add_argument(
        "--url",
        default=os.environ.get("POLYLOGUE_BRIDGE_URL", f"ws://{DEFAULT_BRIDGE_HOST}:{DEFAULT_BRIDGE_PORT}"),
    )
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--outbox", type=Path, default=None)
    args = p.parse_args()

    ctx = resolve_bridge_context(
        args.cwd,
        participant_id=args.participant_id,
        room=args.room,
    )
    log_path, outbox = _paths(ctx.participant_id)
    if args.log:
        log_path = args.log
    if args.outbox:
        outbox = args.outbox

    _log(log_path, f"resolved provider={ctx.provider} room={ctx.room} id={ctx.participant_id} peers={ctx.peers}")

    try:
        asyncio.run(main_async(ctx.room, ctx.participant_id, args.url, log_path, outbox))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
