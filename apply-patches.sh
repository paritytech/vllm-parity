#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
shopt -s nullglob

SITE_PACKAGES=".venv/lib/python3.12/site-packages"

for patch_file in *.patch; do
    if ! patch -p1 -R -f -s --dry-run -d "$SITE_PACKAGES" <"$patch_file" >/dev/null 2>&1; then
        patch -p1 -d "$SITE_PACKAGES" <"$patch_file"
    fi
done
