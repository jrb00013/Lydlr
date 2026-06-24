#!/usr/bin/env bash
# Detect host platform for polylogue service/bootstrap scripts.
# Outputs one of: linux | macos | wsl | windows | unknown

set -euo pipefail

detect_platform() {
  local uname_s
  uname_s="$(uname -s 2>/dev/null || echo unknown)"

  if [[ -f /proc/version ]] && grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
    echo wsl
    return 0
  fi

  case "$uname_s" in
    Linux) echo linux ;;
    Darwin) echo macos ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT) echo windows ;;
    *) echo unknown ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  detect_platform
fi
