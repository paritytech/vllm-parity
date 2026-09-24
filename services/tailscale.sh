#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source services/launch-service.sh

rotate_logs "$SERVICE_LOG"

if [ -z "${TS_AUTHKEY:-}" ]; then
    exit 0
fi

echo "Launching Tailscale under screen..."
launch_service bash launch-tailscale.sh
