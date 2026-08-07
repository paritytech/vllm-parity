#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

COMPILE_CACHE_ROOT_DIR="${COMPILE_CACHE_ROOT_DIR:-/workspace/compile_cache}"
ARCHIVE="${1:-compile-cache.tgz}"

if [ ! -f "$ARCHIVE" ]; then
    echo "WARN: no compile cache baked into the image; startup will be slow!"
    exit 0
fi

mkdir -p "$COMPILE_CACHE_ROOT_DIR"
tar --extract --gzip --skip-old-files \
    --file="$ARCHIVE" \
    --directory="$(dirname -- "$COMPILE_CACHE_ROOT_DIR")"

echo "Restored cache!"
