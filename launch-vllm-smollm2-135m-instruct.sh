#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Small, dumb model just to test out that everything works.
exec bash ./launch-template.sh \
    HuggingFaceTB/SmolLM2-135M-Instruct \
    --served-model-name "smollm2-135m-instruct" \
    "$@"
