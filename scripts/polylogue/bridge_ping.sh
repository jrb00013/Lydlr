#!/usr/bin/env bash
# Send a message to opencode via the live cursor bridge (no reconnect).
# Usage: ./scripts/polylogue/bridge_ping.sh "your message"
#        ./scripts/polylogue/bridge_ping.sh --broadcast "message to whole room"

set -euo pipefail
OUTBOX="/tmp/polylogue-cursor-bridge-outbox.jsonl"
TO="opencode-lydlr"
TEXT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --broadcast) TO=""; shift ;;
    --to) TO="$2"; shift 2 ;;
    *) TEXT="$1"; shift ;;
  esac
done

if [[ -z "$TEXT" ]]; then
  echo "Usage: $0 [--broadcast] [--to ID] \"message\"" >&2
  exit 1
fi

python3 -c "
import json, sys
payload = {'type': 'message', 'text': sys.argv[1]}
if sys.argv[2]:
    payload['to'] = sys.argv[2]
print(json.dumps(payload))
" "$TEXT" "$TO" >> "$OUTBOX"

echo "queued → $TO: $TEXT" >&2
echo "tail -f /tmp/polylogue-cursor-bridge.log for replies" >&2
