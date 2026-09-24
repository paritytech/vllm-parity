#!/bin/bash

set -euo pipefail

WORKSPACE=/workspace
mkdir -p "$WORKSPACE" "${MODELS_DIR:-$WORKSPACE/models}"

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

SESSION_ENGINE=engine

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

bash services/sshd.sh

if ! source engine-environment.sh; then
    finish 1
fi

bash restore-cache.sh
bash services/engine.sh
bash services/caddy.sh
bash services/tailscale.sh
bash services/ssh-tunnel.sh
bash services/metrics.sh
bash services/kv-cache-reaper.sh

tail -n +1 -F /workspace/engine.log /workspace/caddy.log /workspace/metrics.log /workspace/ssh-tunnel.log &

echo "Waiting until the engine quits..."

while true; do
    screen -wipe >/dev/null 2>&1 || true
    sessions="$(screen -list || true)"

    if ! grep -q "\.${SESSION_ENGINE}[[:space:]]" <<<"$sessions"; then
        echo "The engine screen session (${SESSION_ENGINE}) quit" >&2
        break
    fi

    sleep 1
done

# Give the `tail` enough time to flush its output.
sleep 2

finish 1
