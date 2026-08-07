#!/bin/sh

docker run --rm --tty --interactive --gpus all --ipc=host -p 10001:9001 -p 10002:9002 -p 10003:53267 "$@" vllm-parity
