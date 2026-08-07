#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

MODEL_ID="$1"
shift

# Disable buffering in case we're redirected to a log file.
export PYTHONUNBUFFERED="1"

export TORCHINDUCTOR_FX_GRAPH_CACHE="1"
export TORCHINDUCTOR_AUTOGRAD_CACHE="1"
export VLLM_DO_NOT_TRACK="1"

# Force the use of system's nvcc instead of using the torch's, otherwise we get a
# "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths" error.
# Without this DeepSeek-V4-Flash will fail to start, as it needs to JIT compile a bunch of stuff.
# (If this pops up again then check the version of `nvidia-cuda-nvcc` and `nvidia-cuda-runtime`.)
export CUDA_HOME=/usr/local/cuda

export HF_HUB_CACHE=/workspace/huggingface_cache
COMPILE_CACHE_ROOT_DIR="/workspace/compile_cache"

export TORCHINDUCTOR_CACHE_DIR="$COMPILE_CACHE_ROOT_DIR/torch"
export TRITON_CACHE_DIR="$COMPILE_CACHE_ROOT_DIR/triton"
export VLLM_CACHE_ROOT="$COMPILE_CACHE_ROOT_DIR/vllm"
export FLASHINFER_WORKSPACE_BASE="$COMPILE_CACHE_ROOT_DIR/flashinfer"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$COMPILE_CACHE_ROOT_DIR/flashinfer_autotune"
export CUDA_CACHE_PATH="$COMPILE_CACHE_ROOT_DIR/cuda"
export TILELANG_CACHE_DIR="$COMPILE_CACHE_ROOT_DIR/tilelang"
export CUTE_DSL_CACHE_DIR="$COMPILE_CACHE_ROOT_DIR/cutedsl"

if [[ "${DEBUG_EXPOSE_VLLM_PUBLICLY:-}" == "1" ]]; then
    HOST="0.0.0.0"
else
    HOST="127.0.0.1"
fi

uv run vllm serve \
    "$MODEL_ID" \
    --host="$HOST" \
    --port=9001 \
    --seed=234785107 \
    --download-dir=/workspace/models \
    --enable-prefix-caching \
    --scheduling-policy priority \
    --async-scheduling \
    --load-format fastsafetensors \
    --enable-mfu-metrics \
    --kv-cache-metrics \
    "$@"
