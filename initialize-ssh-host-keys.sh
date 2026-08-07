#!/bin/bash

set -euo pipefail

echo "Initializing SSH keys..."

KEY_DIR="/workspace/ssh_host_keys"

mkdir -p "$KEY_DIR"
cp -a "$KEY_DIR/." /etc/ssh/ 2>/dev/null || true
ssh-keygen -A

chown root:root /etc/ssh/ssh_host_*
chmod 600 /etc/ssh/ssh_host_*_key
chmod 644 /etc/ssh/ssh_host_*_key.pub

cp -a /etc/ssh/ssh_host_* "$KEY_DIR/"
