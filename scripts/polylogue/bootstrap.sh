#!/usr/bin/env bash
# Bootstrap Deepiri Polylogue for lydlr — real-time chat bridge (v0.3+).
#
# Coordination is live over WebSocket (ws://127.0.0.1:7850/ws), not journal paste.
#
# Usage:
#   ./scripts/polylogue/bootstrap.sh                    # ensure service + session + participants
#   ./scripts/polylogue/bootstrap.sh connect cursor     # Cursor side: live bridge listener
#   ./scripts/polylogue/bootstrap.sh connect opencode   # OpenCode side: live bridge listener
#   ./scripts/polylogue/bootstrap.sh status

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEEPIRI_POLYLOGUE_REPO="${DEEPIRI_POLYLOGUE_REPO:-$HOME/Documents/Deepiri/deepiri-polylogue}"
ROOM="${POLYLOGUE_BRIDGE_ROOM:-lydlr-multimodal}"

resolve_polylogue_bin() {
  if [[ -n "${POLYLOGUE_BIN:-}" && -x "$POLYLOGUE_BIN" ]]; then
    echo "$POLYLOGUE_BIN"
    return 0
  fi
  if command -v deepiri-polylogue >/dev/null 2>&1; then
    command -v deepiri-polylogue
    return 0
  fi
  local venv_bin="$DEEPIRI_POLYLOGUE_REPO/.venv/bin/deepiri-polylogue"
  if [[ -x "$venv_bin" ]]; then
    echo "$venv_bin"
    return 0
  fi
  if [[ -d "$DEEPIRI_POLYLOGUE_REPO" ]]; then
    python3 -m venv "$DEEPIRI_POLYLOGUE_REPO/.venv" >/dev/null 2>&1 || true
    "$DEEPIRI_POLYLOGUE_REPO/.venv/bin/pip" install -e "$DEEPIRI_POLYLOGUE_REPO" -q
    echo "$DEEPIRI_POLYLOGUE_REPO/.venv/bin/deepiri-polylogue"
    return 0
  fi
  echo "ERROR: deepiri-polylogue not found." >&2
  exit 1
}

POLYLOGUE="$(resolve_polylogue_bin)"

ensure_service() {
  if ! curl -sf http://127.0.0.1:7849/health >/dev/null 2>&1; then
    echo "Starting polylogue service (HTTP + WebSocket bridge)..." >&2
    nohup "$POLYLOGUE" service start --foreground >/tmp/polylogue-service.log 2>&1 &
    sleep 1
  fi
}

participant_id() {
  case "$1" in
    cursor) echo "cursor-lydlr" ;;
    opencode) echo "opencode-lydlr" ;;
    *)
      echo "ERROR: unknown surface '$1' (use cursor or opencode)" >&2
      exit 1
      ;;
  esac
}

cmd_init() {
  ensure_service
  "$POLYLOGUE" --cwd "$REPO_ROOT" init --session "$ROOM"
  "$POLYLOGUE" --cwd "$REPO_ROOT" join --id cursor-lydlr \
    --label "Cursor — integration + UI" --provider cursor
  "$POLYLOGUE" --cwd "$REPO_ROOT" join --id opencode-lydlr \
    --label "OpenCode — phases 1-2-5" --provider opencode

  cat <<EOF
Polylogue real-time bridge ready.
Room: $ROOM
Bridge: ws://127.0.0.1:7850/ws?room=$ROOM&id=<participant-id>

In each chat session, run a live listener (duplex — type lines to send):

  # Cursor terminal
  ./scripts/polylogue/bootstrap.sh connect cursor

  # OpenCode terminal
  ./scripts/polylogue/bootstrap.sh connect opencode

One-shot send:
  $POLYLOGUE bridge send --room $ROOM --id cursor-lydlr --text "ping"

EOF
}

cmd_connect() {
  ensure_service
  local surface="${1:-}"
  if [[ -z "$surface" ]]; then
    echo "Usage: $0 connect {cursor|opencode}" >&2
    exit 1
  fi
  local pid
  pid="$(participant_id "$surface")"
  echo "Connecting to room=$ROOM as $pid (Ctrl+C to disconnect)..." >&2
  exec "$POLYLOGUE" bridge connect --room "$ROOM" --id "$pid" --stdin
}

cmd_status() {
  "$POLYLOGUE" service status 2>/dev/null || true
  echo ""
  "$POLYLOGUE" bridge status 2>/dev/null || true
  echo ""
  "$POLYLOGUE" --cwd "$REPO_ROOT" status 2>/dev/null || echo "(run init first)"
}

main() {
  case "${1:-init}" in
    init) cmd_init ;;
    connect) shift; cmd_connect "${1:-}" ;;
    status) cmd_status ;;
    *)
      echo "Usage: $0 {init|connect|status}" >&2
      exit 1
      ;;
  esac
}

main "$@"
