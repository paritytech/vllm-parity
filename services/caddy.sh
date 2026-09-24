#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source services/launch-service.sh

rotate_logs "$SERVICE_LOG"

if [ -z "${CADDY_API_KEY:-}" ]; then
    exit 0
fi

bash initialize-caddy-certificate.sh

echo "Launching Caddy under screen..."
launch_service caddy run
