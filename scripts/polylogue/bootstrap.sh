#!/usr/bin/env bash
# Thin wrapper — install polylogue once with ./install.sh in deepiri-polylogue, then use
# deepiri-polylogue directly. This script only adds --cwd for the lydlr repo.
#
# Preferred (after install.sh):
#   deepiri-polylogue --cwd "$(git rev-parse --show-toplevel)" bridge listen
#   deepiri-polylogue --cwd "$(git rev-parse --show-toplevel)" bridge send --text "ping"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

if ! command -v deepiri-polylogue >/dev/null 2>&1; then
  cat >&2 <<EOF
deepiri-polylogue not on PATH.

Install once:
  cd ~/Documents/Deepiri/deepiri-polylogue && ./install.sh
  export PATH="\$HOME/.local/bin:\$PATH"

EOF
  exit 1
fi

CMD="${1:-}"
shift || true

case "$CMD" in
  init)
    exec deepiri-polylogue --cwd "$REPO_ROOT" init --session "${1:-lydlr-multimodal}"
    ;;
  listen|connect)
    exec deepiri-polylogue --cwd "$REPO_ROOT" bridge listen "$@"
    ;;
  send|ping)
    exec deepiri-polylogue --cwd "$REPO_ROOT" bridge send --text "$*"
    ;;
  whoami)
    exec deepiri-polylogue --cwd "$REPO_ROOT" bridge whoami "$@"
    ;;
  status)
    deepiri-polylogue --cwd "$REPO_ROOT" bridge whoami
    echo ""
    deepiri-polylogue service status
    echo ""
    deepiri-polylogue bridge status
    ;;
  "")
    sed -n '2,8p' "$0"
    ;;
  *)
    exec deepiri-polylogue --cwd "$REPO_ROOT" "$CMD" "$@"
    ;;
esac
