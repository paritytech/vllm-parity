#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source services/launch-service.sh
source engine-environment.sh

rotate_logs "$SERVICE_LOG"

echo "Launching $ENGINE with model $MODEL under screen..."
launch_service python3 serve.py --model "$MODEL" --engine "$ENGINE"
