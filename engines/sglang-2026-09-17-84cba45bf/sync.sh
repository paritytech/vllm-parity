#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export CARGO_HOME=/tmp/cargo RUSTUP_HOME=/tmp/rustup
export PATH="$CARGO_HOME/bin:$PATH"

curl -LsSf https://sh.rustup.rs -o /tmp/install-rustup.sh
bash /tmp/install-rustup.sh -y --profile minimal --default-toolchain 1.92 --no-modify-path

uv sync --frozen --no-cache

rm -rf /tmp/install-rustup.sh "$CARGO_HOME" "$RUSTUP_HOME"
