#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

keep_container_alive() {
    [ -f "/tmp/keep-container-alive" ] || [ "${DEBUG_KEEP_CONTAINER_ALIVE:-0}" = "1" ]
}

finish() {
    local code="${1:-0}"

    if [ "$code" -ne 0 ]; then
        echo "Finished with status: $code" >&2
    fi

    if keep_container_alive; then
        echo "Keeping container alive forever!"
        while true; do
            sleep 60
        done
    fi

    exit "$code"
}

bash initialize-ssh-host-keys.sh
service ssh start

if [ -z "${MODEL:-}" ]; then
    echo "Error: MODEL environment variable is not set." >&2
    finish 1
fi

LAUNCH_SCRIPT="launch-vllm-${MODEL}.sh"

if [ ! -f "$LAUNCH_SCRIPT" ]; then
    echo "Error: Launch script '$LAUNCH_SCRIPT' does not exist." >&2
    finish 1
fi

bash restore-cache.sh

if [ -n "${CADDY_API_KEY:-}" ]; then
    bash initialize-caddy-certificate.sh
fi

uv sync --frozen --no-cache
bash apply-patches.sh

SESSION_VLLM=vllm
SESSION_CADDY=caddy
SESSION_METRICS=metrics
SESSION_TAILSCALE=tailscale
SESSION_SSH_TUNNEL=ssh-tunnel
LOG_VLLM=/workspace/vllm.log
LOG_CADDY=/workspace/caddy.log
LOG_METRICS=/workspace/metrics.log
LOG_TAILSCALE=/workspace/tailscale.log
LOG_SSH_TUNNEL=/workspace/ssh-tunnel.log
METRICS=/workspace/metrics.jsonl

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="/workspace/archive/$STAMP"
attempt=0
while [ -e "$ARCHIVE" ]; do
    attempt=$((attempt + 1))
    ARCHIVE="/workspace/archive/$STAMP-$attempt"
done

for file in "$LOG_VLLM" "$LOG_CADDY" "$LOG_METRICS" "$LOG_TAILSCALE" "$LOG_SSH_TUNNEL" "$METRICS"; do
    if [ -f "$file" ]; then
        mkdir -p "$ARCHIVE"
        mv -- "$file" "$ARCHIVE/"
    fi
    touch "$file"
done

if [ -d "$ARCHIVE" ]; then
    echo "Archived the previous run to $ARCHIVE"
fi

tail -n 0 -F "$LOG_VLLM" "$LOG_CADDY" "$LOG_METRICS" "$LOG_SSH_TUNNEL" &

echo "Launching vLLM under screen..."
screen -dmS "$SESSION_VLLM" bash with-logging.sh "$LOG_VLLM" bash "$LAUNCH_SCRIPT"

if [ -n "${CADDY_API_KEY:-}" ]; then
    echo "Launching Caddy under screen..."
    screen -dmS "$SESSION_CADDY" bash with-logging.sh "$LOG_CADDY" caddy run
fi

if [ -n "${TS_AUTHKEY:-}" ]; then
    echo "Launching Tailscale under screen..."
    screen -dmS "$SESSION_TAILSCALE" bash with-logging.sh "$LOG_TAILSCALE" bash launch-tailscale.sh
fi

if [ -n "${SSH_TUNNEL_HOST:-}" ]; then
    echo "Launching the reverse SSH tunnel under screen..."
    screen -dmS "$SESSION_SSH_TUNNEL" bash with-logging.sh "$LOG_SSH_TUNNEL" bash launch-ssh-tunnel.sh
fi

echo "Launching the metrics collector under screen..."
screen -dmS "$SESSION_METRICS" bash with-logging.sh "$LOG_METRICS" bash collect-metrics.sh "$METRICS"

echo "Waiting until vLLM quits..."

while true; do
    screen -wipe >/dev/null 2>&1 || true
    sessions="$(screen -list || true)"

    if ! grep -q "\.${SESSION_VLLM}[[:space:]]" <<<"$sessions"; then
        echo "vLLM screen session (${SESSION_VLLM}) quit" >&2
        break
    fi

    sleep 1
done

# Give the `tail` enough time to flush its output.
sleep 2

finish 1
