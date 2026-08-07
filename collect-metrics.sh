#!/bin/bash

# Deliberately no `set -e`: stats are auxiliary to serving. If the collector dies the
# container must keep serving, so this restarts it instead of propagating the failure.
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

OUTPUT="${1:-/workspace/metrics.jsonl}"

# Five seconds rather than one: bucket counts make a busy line ~1.5 kB, and nothing in
# the capacity analysis needs second-by-second resolution. ~26 MB a day at this rate.
INTERVAL="${METRICS_INTERVAL:-5}"

while true; do
    uv run python collect-metrics.py --output "$OUTPUT" --interval "$INTERVAL"
    echo "Metrics collector exited with status $?; restarting in 5s." >&2
    sleep 5
done
