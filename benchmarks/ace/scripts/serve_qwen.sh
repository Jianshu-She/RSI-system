#!/bin/bash
# Usage: serve_qwen.sh <gpu> <port>   -- one Qwen3-32B vLLM replica (TP=1)
# Qwen3-32B is natively 32k; YaRN x4 -> 131072 (Qwen's recommended long-context setting) so ACE playbooks + AppWorld histories fit.
# --override-generation-config caps max_new_tokens at 8192 server-side (stops runaway repetition; legit outputs <= 2.6k).
source "$(dirname "$(readlink -f "$0")")/../env.sh"
CUDA_VISIBLE_DEVICES=$1 exec $VLLM_BIN serve $MODEL_DIR --served-model-name qwen3-32b \
  --port $2 --host $LOCAL_LLM_HOST --max-model-len 131072 \
  --gpu-memory-utilization 0.90 --enable-prefix-caching --max-num-seqs 64 \
  --override-generation-config '{"max_new_tokens": 8192}'
