#!/bin/bash

set -euo pipefail

COMPILE_CACHE_ROOT_DIR="${COMPILE_CACHE_ROOT_DIR:-/workspace/compile_cache}"
ARCHIVE="${1:-/workspace/compile-cache.tgz}"

if [ ! -d "$COMPILE_CACHE_ROOT_DIR" ]; then
    echo "Error: '$COMPILE_CACHE_ROOT_DIR' does not exist; nothing to gather." >&2
    exit 1
fi

# Note: `--format=posix` is required to preserve sub-second mtimes.
tar --create --gzip --format=posix \
    --file="$ARCHIVE" \
    --directory="$(dirname -- "$COMPILE_CACHE_ROOT_DIR")" \
    -- "$(basename -- "$COMPILE_CACHE_ROOT_DIR")"

echo "Wrote $ARCHIVE ($(du -h -- "$ARCHIVE" | cut -f1))"
