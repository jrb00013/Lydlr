#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(pwd)/src:$(pwd)/.venv/lib/python3.10/site-packages"
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
source "$(pwd)/install/setup.bash"
