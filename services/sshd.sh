#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

bash initialize-ssh-host-keys.sh
service ssh start
