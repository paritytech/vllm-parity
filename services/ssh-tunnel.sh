#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source services/launch-service.sh

rotate_logs "$SERVICE_LOG"

if [ -z "${SSH_TUNNEL_HOST:-}" ]; then
    exit 0
fi

echo "Launching the reverse SSH tunnel under screen..."
launch_service bash launch-ssh-tunnel.sh
