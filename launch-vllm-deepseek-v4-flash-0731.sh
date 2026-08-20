#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

GPU_COUNT=$(nvidia-smi -L | wc -l)
TOTAL_RAM=$(cat /sys/fs/cgroup/memory.max)
KV_CACHE_CPU_OFFLOAD_SIZE=$((TOTAL_RAM / 10 * 5)) # 50% is KV cache
DATA_PARALLEL_COUNT=$(( $GPU_COUNT < 2 ? 1 : $GPU_COUNT / 2 ))

BLACKWELL_GPU_COUNT=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -cE '^(10|12)\.' || true)
FP4_INDEXER_CACHE_ARGS=()
if (( BLACKWELL_GPU_COUNT == GPU_COUNT )); then
    FP4_INDEXER_CACHE_ARGS=(--attention_config.use_fp4_indexer_cache=True)
    FP4_INDEXER_CACHE_DECISION="enabled"
else
    FP4_INDEXER_CACHE_DECISION="disabled"
fi

echo "Blackwell GPUs detected: $BLACKWELL_GPU_COUNT of $GPU_COUNT; FP4 indexer cache: $FP4_INDEXER_CACHE_DECISION"

mkdir -p /workspace/kv_cache

exec bash ./launch-template.sh \
    deepseek-ai/DeepSeek-V4-Flash-0731 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --served-model-name "deepseek-v4-flash-0731" \
    --default-chat-template-kwargs '{"thinking": true, "reasoning_effort": "max"}' \
    --override-generation-config '{"top_p":0.95}' \
    --kv-transfer-config "{
        \"kv_connector\": \"OffloadingConnector\",
        \"kv_role\": \"kv_both\",
        \"kv_connector_extra_config\":{
            \"cpu_bytes_to_use\": $KV_CACHE_CPU_OFFLOAD_SIZE,
            \"blocks_per_chunk\": 4,
            \"eviction_policy\": \"lru\",
            \"secondary_tiers\": [{
                \"type\": \"fs\",
                \"root_dir\": \"/workspace/kv_cache\",
                \"n_read_threads\": 32,
                \"n_write_threads\": 16
            }]
        }
    }" \
    "${FP4_INDEXER_CACHE_ARGS[@]}" \
    --enable-expert-parallel \
    --data-parallel-size $DATA_PARALLEL_COUNT \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --max-num-batched-tokens 8192 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE", "custom_ops":["all"]}' \
    "$@"
