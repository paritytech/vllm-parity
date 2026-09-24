#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [ -n "${1:-}" ]; then
    ENGINE="$1"
else
    source engine-environment.sh
fi

COMPILE_CACHE_ROOT_DIR="${COMPILE_CACHE_ROOT_DIR:-/workspace/compile_cache}"
ARCHIVE="${2:-/workspace/compile-cache-$ENGINE.tgz}"

if [ ! -d "$COMPILE_CACHE_ROOT_DIR/$ENGINE" ]; then
    echo "Error: '$COMPILE_CACHE_ROOT_DIR/$ENGINE' does not exist; nothing to gather." >&2
    exit 1
fi

# Note: `--format=posix` is required to preserve sub-second mtimes.
tar --create --gzip --format=posix \
    --file="$ARCHIVE" \
    --directory="$COMPILE_CACHE_ROOT_DIR/$ENGINE" \
    -- .

echo "Wrote $ARCHIVE ($(du -h -- "$ARCHIVE" | cut -f1))"
echo "Copy it to engines/$ENGINE/compile-cache.tgz in the repository and rebuild the image."
