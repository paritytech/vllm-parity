#!/bin/bash

set -euo pipefail

if [[ -z "${TS_AUTHKEY:-}" ]]; then
    echo "Error: TS_AUTHKEY environment variable is required." >&2
    exit 1
fi

tailscaled --tun=userspace-networking --state=/workspace/tailscaled.state &
TAILSCALED_PID=$!

sleep 3

ARGS=(--authkey="$TS_AUTHKEY")

if [[ -n "${TS_HOSTNAME:-}" ]]; then
    ARGS+=(--hostname="$TS_HOSTNAME")
fi

tailscale up "${ARGS[@]}"

wait "$TAILSCALED_PID"
