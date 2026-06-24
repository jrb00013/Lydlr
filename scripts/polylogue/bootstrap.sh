#!/usr/bin/env bash
# Polylogue bridge — auto-detects room + participant from repo registry and runtime.
#
# Usage:
#   ./scripts/polylogue/bootstrap.sh init
#   ./scripts/polylogue/bootstrap.sh listen      # from Cursor or OpenCode — no env vars
#   ./scripts/polylogue/bootstrap.sh send "ping"
#   ./scripts/polylogue/bootstrap.sh status

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEEPIRI_POLYLOGUE_REPO="${DEEPIRI_POLYLOGUE_REPO:-$HOME/Documents/Deepiri/deepiri-polylogue}"

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

resolve_python() {
  local venv_py="$DEEPIRI_POLYLOGUE_REPO/.venv/bin/python3"
  if [[ -x "$venv_py" ]]; then
    echo "$venv_py"
    return 0
  fi
  command -v python3
}

POLYLOGUE="$(resolve_polylogue_bin)"
PYTHON="$(resolve_python)"

resolve_json() {
  "$PYTHON" "$SCRIPT_DIR/resolve.py" --cwd "$REPO_ROOT" "$@" 2>/dev/null | tail -1
}

ensure_service() {
  if ! curl -sf http://127.0.0.1:7849/health >/dev/null 2>&1; then
    echo "Starting polylogue service..." >&2
    nohup "$POLYLOGUE" service start --foreground >/tmp/polylogue-service.log 2>&1 &
    sleep 1
  fi
}

cmd_init() {
  ensure_service
  local room="${1:-lydlr-multimodal}"
  "$POLYLOGUE" --cwd "$REPO_ROOT" init --session "$room"
  cat <<EOF
Session ready for $(basename "$REPO_ROOT").

Each agent in this repo just runs:
  ./scripts/polylogue/bootstrap.sh listen

Room + participant id are inferred from:
  - git repo → polylogue session registry
  - runtime (CURSOR_AGENT, opencode in process tree, etc.)
  - participants.json provider field

Send to the other agent:
  ./scripts/polylogue/bootstrap.sh send "your message"

EOF
}

cmd_listen() {
  ensure_service
  local ctx
  ctx="$(resolve_json)"
  local room id provider
  room="$(echo "$ctx" | "$PYTHON" -c "import sys,json; print(json.load(sys.stdin)['room'])")"
  id="$(echo "$ctx" | "$PYTHON" -c "import sys,json; print(json.load(sys.stdin)['participant_id'])")"
  provider="$(echo "$ctx" | "$PYTHON" -c "import sys,json; print(json.load(sys.stdin)['provider'])")"
  echo "Auto: provider=$provider room=$room id=$id" >&2
  exec "$PYTHON" "$SCRIPT_DIR/bridge_listener.py" --cwd "$REPO_ROOT"
}

cmd_send() {
  exec "$SCRIPT_DIR/bridge_send.sh" "$@"
}

cmd_whoami() {
  resolve_json --json | "$PYTHON" -m json.tool
}

cmd_status() {
  echo "=== resolved context ==="
  cmd_whoami
  echo ""
  "$POLYLOGUE" service status 2>/dev/null || true
  echo ""
  "$POLYLOGUE" bridge status 2>/dev/null || true
}

main() {
  case "${1:-}" in
    init) shift || true; cmd_init "${1:-lydlr-multimodal}" ;;
    listen|connect) cmd_listen ;;
    send|ping) shift; cmd_send "$@" ;;
    whoami) cmd_whoami ;;
    status) cmd_status ;;
    -h|--help|"")
      sed -n '2,9p' "$0"
      ;;
    *)
      echo "Usage: $0 {init|listen|send|whoami|status}" >&2
      exit 1
      ;;
  esac
}

main "$@"
