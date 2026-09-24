#!/bin/bash

if ! engine_environment="$(python3 "$(dirname -- "${BASH_SOURCE[0]}")/serve.py" get-environment)"; then
    return 1
fi

eval "$engine_environment"
unset engine_environment
export MODEL ENGINE ENGINE_FAMILY
