#!/bin bash

set -euo pipefail

LOG_FILE="$1"
shift

exec "$@" 2>&1 | tee -a "$LOG_FILE"
