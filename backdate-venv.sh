#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

VENV_MTIME="2020-01-01 00:00:00 UTC"

ENGINE_DIR="${1:?usage: backdate-venv.sh <engine directory>}"
VENV="$ENGINE_DIR/.venv"

if [ ! -d "$VENV" ]; then
    echo "$VENV does not exist" >&2
    exit 1
fi

find "$VENV" -exec touch -h -d "$VENV_MTIME" -- {} +

echo "Every mtime under $VENV is now $VENV_MTIME"
