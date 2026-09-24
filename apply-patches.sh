#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
shopt -s nullglob

ENGINE_DIR="${1:?usage: apply-patches.sh <engine directory>}"
matches=("$ENGINE_DIR"/.venv/lib/python3.*/site-packages)

if [ "${#matches[@]}" -ne 1 ]; then
    echo "$ENGINE_DIR/.venv/lib does not contain exactly one site-packages directory (${#matches[@]} matches)" >&2
    exit 1
fi

SITE_PACKAGES="${matches[0]}"

for patch_file in "$ENGINE_DIR"/patches/*.patch; do
    if ! patch -p1 -R -f -s --dry-run -d "$SITE_PACKAGES" <"$patch_file" >/dev/null 2>&1; then
        echo "Applying $patch_file..."
        patch -p1 -d "$SITE_PACKAGES" <"$patch_file"
    fi
done
