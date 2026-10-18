#!/bin/bash
# Serve the two helper models every evaluation needs:
#   - Qwen3-Embedding-8B (memory retrieval for ReasoningBank)            GPU ${EMBED_GPUS:-2}, port 8001
#   - Qwen3.5-397B-A17B-FP8 (official grader for fuzzy-match answers,     GPUs ${JUDGE_GPUS:-4,5,6,7}, port 8002
#     also the cross-check judge / memory writer for reasoningbank_cross)
source "$(dirname "$0")/../env.sh" > /dev/null
source $WA_VLLM_VENV/bin/activate
# NO_EMBED=1 skips the embedding server (only ReasoningBank needs it)
[ -z "$NO_EMBED" ] && CUDA_VISIBLE_DEVICES=${EMBED_GPUS:-2} setsid nohup vllm serve Qwen/Qwen3-Embedding-8B --served-model-name qwen3-embedding-8b \
  --runner pooling --host 127.0.0.1 --port 8001 > "$WA_LOGS/vllm_embed.log" 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=${JUDGE_GPUS:-4,5,6,7} setsid nohup vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --served-model-name qwen3.5-397b \
  --tensor-parallel-size 4 --max-model-len 131072 --reasoning-parser qwen3 --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enable-prefix-caching --host 127.0.0.1 --port 8002 > "$WA_LOGS/vllm_judge.log" 2>&1 < /dev/null &
echo "waiting for embedding (8001) and grader (8002); the 397B model takes ~10-15 min to load"
until { [ -n "$NO_EMBED" ] || curl -s -o /dev/null --max-time 5 http://127.0.0.1:8001/v1/models; } && curl -s -o /dev/null --max-time 5 http://127.0.0.1:8002/v1/models; do sleep 20; done
echo "aux models up"
