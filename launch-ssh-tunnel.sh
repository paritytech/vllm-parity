#!/bin/bash

set -euo pipefail

for required in SSH_TUNNEL_HOST SSH_TUNNEL_REMOTE_VLLM_PORT SSH_TUNNEL_PRIVATE_KEY; do
    if [[ -z "${!required:-}" ]]; then
        echo "Error: $required environment variable is required." >&2
        exit 1
    fi
done

REMOTE_BIND_ADDRESS=127.0.0.1
REMOTE_PORT="$SSH_TUNNEL_REMOTE_VLLM_PORT"

if [[ "$REMOTE_PORT" == *:* ]]; then
    REMOTE_BIND_ADDRESS="${REMOTE_PORT%:*}"
    REMOTE_PORT="${REMOTE_PORT##*:}"
fi

if [[ ! "$REMOTE_PORT" =~ ^[0-9]+$ || ! "$REMOTE_BIND_ADDRESS" =~ ^([^:]+|\[[0-9a-fA-F:]+\])$ ]]; then
    echo "Error: SSH_TUNNEL_REMOTE_VLLM_PORT must be a port, optionally prefixed with the address" >&2
    exit 1
fi

KEY_FILE="${SSH_TUNNEL_PRIVATE_KEY_FILE:-/root/.ssh/tunnel_key}"
KNOWN_HOSTS_FILE="${SSH_TUNNEL_KNOWN_HOSTS_FILE:-/workspace/ssh_tunnel_known_hosts}"

umask 077

if ! mkdir -p -- "$(dirname -- "$KEY_FILE")" "$(dirname -- "$KNOWN_HOSTS_FILE")"; then
    echo "Error: cannot create the directory for $KEY_FILE or $KNOWN_HOSTS_FILE" >&2
    exit 1
fi

if [[ "$SSH_TUNNEL_PRIVATE_KEY" == *"PRIVATE KEY"* ]]; then
    printf '%s\n' "$SSH_TUNNEL_PRIVATE_KEY" > "$KEY_FILE"
elif ! base64 -d <<<"$SSH_TUNNEL_PRIVATE_KEY" > "$KEY_FILE"; then
    echo "Error: SSH_TUNNEL_PRIVATE_KEY is neither a private key nor base64-encoded" >&2
    exit 1
fi

if ! ssh-keygen -y -f "$KEY_FILE" </dev/null >/dev/null 2>&1; then
    echo "Error: SSH_TUNNEL_PRIVATE_KEY is not a usable private key" >&2
    exit 1
fi

if [[ -n "${SSH_TUNNEL_HOST_KEY:-}" ]]; then
    # Enforce that we're connecting to the target host.
    while read -r line; do
        case "$line" in
            ssh-*|ecdsa-*|sk-*) printf '* %s\n' "$line" ;;
            *)                  printf '%s\n' "$line" ;;
        esac
    done <<<"$SSH_TUNNEL_HOST_KEY" > "$KNOWN_HOSTS_FILE"

    if ! ssh-keygen -l -f "$KNOWN_HOSTS_FILE" >/dev/null 2>&1; then
        echo "Error: SSH_TUNNEL_HOST_KEY is not an SSH host key" >&2
        exit 1
    fi

    HOST_KEY_CHECKING=yes
else
    # Trust the host on first connection.
    HOST_KEY_CHECKING=accept-new
fi

echo "Tunnelling ${REMOTE_BIND_ADDRESS}:${REMOTE_PORT} on ${SSH_TUNNEL_HOST} to vLLM on 127.0.0.1:9001"

# Don't abort on failures from now on.
set +e

while true; do
    ssh -N \
        -i "$KEY_FILE" \
        -R "${REMOTE_BIND_ADDRESS}:${REMOTE_PORT}:127.0.0.1:9001" \
        -o BatchMode=yes \
        -o IdentitiesOnly=yes \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=15 \
        -o ServerAliveCountMax=3 \
        -o StrictHostKeyChecking="$HOST_KEY_CHECKING" \
        -o UserKnownHostsFile="$KNOWN_HOSTS_FILE" \
        "$SSH_TUNNEL_HOST"

    echo "SSH tunnel exited with status $?; reconnecting in 5s..." >&2
    sleep 5
done
