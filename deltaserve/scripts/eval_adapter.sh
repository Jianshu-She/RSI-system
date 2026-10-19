#!/bin/bash
# Score one model/adapter on an ACE Finance TEST split with the ACE benchmark's protocol (= results/RESULTS.md of
# benchmarks/ace): stock vLLM 0.31 (VLLM_BIN), MODEL_DIR (YaRN x4), TP=1, temperature 0, EMPTY playbook.
# Usage: deltaserve/scripts/eval_adapter.sh <gpu> <port> <formula|finer> <tag> <out_dir> [adapter_dir]
#   writes <out_dir>/eval_<tag>_<task>.json (+ logs/); no adapter = the base model
set -u
source "$(dirname "$(readlink -f "$0")")/../env.sh"
GPU=$1; PORT=$2; TASK=$3; TAG=$4; OUT=$5; ADAPTER=${6:-}
mkdir -p $OUT/logs
NAME=qwen3-32b; LORA=()
if [ -n "$ADAPTER" ]; then NAME=$TAG; LORA=(--enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules "$TAG=$ADAPTER"); fi
CUDA_VISIBLE_DEVICES=$GPU setsid $VLLM_BIN serve $MODEL_DIR --served-model-name qwen3-32b --port $PORT --host 127.0.0.1 \
  --max-model-len 131072 --gpu-memory-utilization 0.90 --enable-prefix-caching --max-num-seqs 64 \
  --override-generation-config '{"max_new_tokens": 8192}' "${LORA[@]}" > $OUT/logs/vllm_${TAG}_$TASK.log 2>&1 &
S=$!
until curl -s 127.0.0.1:$PORT/v1/models | grep -q "\"$NAME\""; do
  sleep 10; kill -0 $S 2>/dev/null || { echo "vLLM died ($TAG), see $OUT/logs/vllm_${TAG}_$TASK.log"; exit 1; }
done
cd $ACE_UPSTREAM_DIR
LOCAL_LLM_URL=http://127.0.0.1:$PORT/v1 EVAL_MODEL=$NAME $ACE_PY $ACE_ROOT/eval/eval_split.py $TASK test \
  $DS_HOME/configs/empty_context.txt $OUT/eval_${TAG}_$TASK.json > $OUT/logs/eval_${TAG}_$TASK.log 2>&1
echo "$(date '+%T') $(grep RESULT $OUT/logs/eval_${TAG}_$TASK.log) [$TAG]"
kill -INT -$S 2>/dev/null; sleep 15; kill -KILL -$S 2>/dev/null; true
