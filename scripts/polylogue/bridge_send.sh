#!/usr/bin/env bash
# Queue a bridge message — auto-resolves sender id and default recipient.
#
# Usage:
#   ./scripts/polylogue/bridge_send.sh "hello"
#   ./scripts/polylogue/bridge_send.sh --to opencode-lydlr "hello"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEEPIRI_POLYLOGUE_REPO="${DEEPIRI_POLYLOGUE_REPO:-$HOME/Documents/Deepiri/deepiri-polylogue}"
PYTHON="${DEEPIRI_POLYLOGUE_REPO}/.venv/bin/python3"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"

TO=""
TEXT=""
BROADCAST=0
PARTICIPANT_ID=""
ROOM=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --id) PARTICIPANT_ID="$2"; shift 2 ;;
    --room) ROOM="$2"; shift 2 ;;
    --to) TO="$2"; shift 2 ;;
    --broadcast) BROADCAST=1; shift ;;
    -h|--help)
      sed -n '2,6p' "$0"
      exit 0
      ;;
    *) TEXT="$1"; shift ;;
  esac
done

if [[ -z "$TEXT" ]]; then
  echo "Usage: $0 [--to ID] \"message\"" >&2
  exit 1
fi

RESOLVE_ARGS=(--cwd "$REPO_ROOT")
[[ -n "$PARTICIPANT_ID" ]] && RESOLVE_ARGS+=(--id "$PARTICIPANT_ID")
[[ -n "$ROOM" ]] && RESOLVE_ARGS+=(--room "$ROOM")

CTX="$("$PYTHON" "$SCRIPT_DIR/resolve.py" "${RESOLVE_ARGS[@]}" 2>/dev/null | tail -1)"
PARTICIPANT_ID="$(echo "$CTX" | "$PYTHON" -c "import sys,json; print(json.load(sys.stdin)['participant_id'])")"
ROOM="$(echo "$CTX" | "$PYTHON" -c "import sys,json; print(json.load(sys.stdin)['room'])")"

if [[ $BROADCAST -eq 0 && -z "$TO" ]]; then
  TO="$(echo "$CTX" | "$PYTHON" -c "
import sys, json
d = json.load(sys.stdin)
peers = d.get('peers') or []
print(peers[0] if len(peers) == 1 else '')
")"
fi

SAFE_ID="$(printf '%s' "$PARTICIPANT_ID" | tr -c '[:alnum:]-_' '_')"
STATE_DIR="${POLYLOGUE_BRIDGE_STATE_DIR:-/tmp}"
OUTBOX="$STATE_DIR/polylogue-bridge-${SAFE_ID}-outbox.jsonl"
LOG="$STATE_DIR/polylogue-bridge-${SAFE_ID}.log"

"$PYTHON" -c "
import json, sys
payload = {'type': 'message', 'text': sys.argv[1]}
mode = sys.argv[2]
if mode == 'broadcast':
    pass
elif mode:
    payload['to'] = mode
print(json.dumps(payload))
" "$TEXT" "$( [[ $BROADCAST -eq 1 ]] && echo broadcast || echo "$TO" )" >> "$OUTBOX"

if [[ $BROADCAST -eq 1 ]]; then
  echo "queued ($PARTICIPANT_ID @ $ROOM) broadcast: $TEXT" >&2
elif [[ -n "$TO" ]]; then
  echo "queued ($PARTICIPANT_ID @ $ROOM) → $TO: $TEXT" >&2
else
  echo "queued ($PARTICIPANT_ID @ $ROOM) room: $TEXT" >&2
fi
echo "log: $LOG" >&2
