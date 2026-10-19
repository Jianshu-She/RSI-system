#!/bin/bash
# One ACE Finance test-split eval with RSI-system's exact protocol (their job_lora_eval_finance.sbatch):
# stock vLLM 0.31, Qwen3-32B YaRN x4 config, TP=1, temp 0 (eval_split.py), EMPTY playbook context.
# Usage: eval-rsi/eval_ace.sh <gpu> <port> <formula|finer> <tag> [adapter_dir]
set -u
GPU=$1; PORT=$2; TASK=$3; TAG=$4; ADAPTER=${5:-}
ACE=/mnt/weka/home/jianshu.she/RSI-system/benchmark/ACE
VLLM=/mnt/weka/home/jianshu.she/RSI-system/benchmark/reasoningbank/envs/vllm/bin/vllm
MODEL=$ACE/models/Qwen3-32B-yarn131k
OUT=/mnt/weka/home/jianshu.she/deltaserve/DeltaServe-vLLM/eval-rsi/eval; mkdir -p $OUT/logs
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
LORA=(); NAME=qwen3-32b
if [ -n "$ADAPTER" ]; then LORA=(--enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules "$TAG=$ADAPTER"); NAME=$TAG; fi
CUDA_VISIBLE_DEVICES=$GPU setsid $VLLM serve $MODEL --served-model-name qwen3-32b --port $PORT --host 127.0.0.1 \
  --max-model-len 131072 --gpu-memory-utilization 0.90 --enable-prefix-caching --max-num-seqs 64 \
  --override-generation-config '{"max_new_tokens": 8192}' "${LORA[@]}" > $OUT/logs/vllm_$TAG.log 2>&1 &
S=$!
until curl -s 127.0.0.1:$PORT/v1/models | grep -q "\"$NAME\""; do sleep 10; kill -0 $S 2>/dev/null || { echo "vllm died ($TAG)"; exit 1; }; done
cd $ACE/ace
LOCAL_LLM_URL=http://127.0.0.1:$PORT/v1 EVAL_MODEL=$NAME .venv/bin/python ../baselines/eval_split.py $TASK test \
  $ACE/sft/empty_context.txt $OUT/eval_$TAG.json > $OUT/logs/eval_$TAG.log 2>&1
echo "$(date '+%T') $(grep RESULT $OUT/logs/eval_$TAG.log) [$TAG]"
kill -INT -$S 2>/dev/null; sleep 15; kill -KILL -$S 2>/dev/null
