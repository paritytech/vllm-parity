#!/bin/sh

exec bash ./run-local.sh --entrypoint /app/entrypoint-interactive.sh "$@"
