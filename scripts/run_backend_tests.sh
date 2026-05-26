#!/usr/bin/env bash
# Run backend logic tests without live MongoDB/Redis/Docker.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q django djangorestframework django-cors-headers channels motor redis python-dotenv requests pytest pytest-django
fi

.venv/bin/python -m pytest tests/ -v "$@"
