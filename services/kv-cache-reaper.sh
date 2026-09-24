#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source services/launch-service.sh
source engine-environment.sh

rotate_logs "$SERVICE_LOG"

if [ "${ENGINE_FAMILY:-}" != vllm ]; then
    exit 0
fi

echo "Launching the KV cache reaper under screen..."
launch_service python3 reap-kv-cache.py
