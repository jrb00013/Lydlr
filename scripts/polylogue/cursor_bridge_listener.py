#!/usr/bin/env python3
"""Persistent cursor-lydlr polylogue bridge — listen + send via outbox file."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import websockets

ROOM = "lydlr-multimodal"
PARTICIPANT = "cursor-lydlr"
URI = f"ws://127.0.0.1:7850/ws?room={ROOM}&id={PARTICIPANT}"
LOG = Path("/tmp/polylogue-cursor-bridge.log")
OUTBOX = Path("/tmp/polylogue-cursor-bridge-outbox.jsonl")


def _log(line: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
    print(line, flush=True)


async def _outbox_reader(ws: websockets.WebSocketClientProtocol) -> None:
    """Poll outbox file for messages to send on the live socket."""
    OUTBOX.parent.mkdir(parents=True, exist_ok=True)
    if not OUTBOX.exists():
        OUTBOX.touch()
    offset = 0
    while True:
        await asyncio.sleep(0.25)
        try:
            raw = OUTBOX.read_text(encoding="utf-8")
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
                _log(f"outbox skip bad json: {line[:80]}")
                continue
            if payload.get("type") != "message":
                payload = {"type": "message", "text": str(payload.get("text", line))}
            if "to" not in payload and payload.get("to") is None:
                payload.setdefault("to", "opencode-lydlr")
            await ws.send(json.dumps(payload))
            _log(f"sent: {json.dumps(payload, ensure_ascii=False)}")
        offset = len(raw)


async def _listen(ws: websockets.WebSocketClientProtocol) -> None:
    async for raw in ws:
        _log(f"recv: {raw}")


async def run_session() -> None:
    async with websockets.connect(URI) as ws:
        _log(f"connected as {PARTICIPANT} room={ROOM}")
        reader = asyncio.create_task(_listen(ws))
        writer = asyncio.create_task(_outbox_reader(ws))
        await asyncio.gather(reader, writer)


async def main() -> None:
    while True:
        try:
            await run_session()
        except (websockets.ConnectionClosed, OSError) as exc:
            _log(f"disconnected: {exc} — reconnecting in 2s")
            await asyncio.sleep(2)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
