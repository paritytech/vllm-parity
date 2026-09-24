#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source services/launch-service.sh

METRICS=/workspace/metrics.jsonl

rotate_logs "$SERVICE_LOG" "$METRICS"

echo "Launching the metrics collector under screen..."
launch_service bash collect-metrics.sh "$METRICS"
