#!/bin/bash

set -euo pipefail

CERT_FILE="/workspace/caddy-cert.pem"
KEY_FILE="/workspace/caddy-key.pem"

if [[ ! -f "$CERT_FILE" || ! -f "$KEY_FILE" ]]; then
    echo "Generating self-signed TLS certificate..."

    openssl req -x509 \
        -newkey rsa:2048 \
        -nodes \
        -days 36500 \
        -keyout "$KEY_FILE" \
        -out "$CERT_FILE" \
        -subj "/CN=localhost" \
        -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:::1"

    chmod 600 "$KEY_FILE"
fi
