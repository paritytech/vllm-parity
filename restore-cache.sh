#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [ -n "${1:-}" ]; then
    ENGINE="$1"
else
    source engine-environment.sh
fi

COMPILE_CACHE_ROOT_DIR="${COMPILE_CACHE_ROOT_DIR:-/workspace/compile_cache}"
ARCHIVE="engines/$ENGINE/compile-cache.tgz"

if [ ! -f "$ARCHIVE" ]; then
    echo "WARN: no compile cache baked into the image for $ENGINE; startup will be slow!"
    exit 0
fi

mkdir -p "$COMPILE_CACHE_ROOT_DIR/$ENGINE"
tar --extract --gzip --skip-old-files \
    --file="$ARCHIVE" \
    --directory="$COMPILE_CACHE_ROOT_DIR/$ENGINE"

echo "Restored $ARCHIVE into $COMPILE_CACHE_ROOT_DIR/$ENGINE"
