#!/bin/bash
# One stock vLLM 0.31 server (TP=1, the ACE protocol's settings), several evals in a row, each with/without skills.
#   bench: formula | finer   -> ACE eval_split.py, test split, playbook in ACE's Generator prompt ("skills")
#          gsm8k | mmlu      -> eval_general.py, playbook in the system message
#   ctx:   empty | skills (skills = $PLAYBOOK)
# Writes <out>/eval_<tag>_<ctx>_<bench>.json (+ metrics_<tag>_<ctx>_<bench>.jsonl: KV usage etc. during the eval)
# Usage: eval_suite.sh <gpu> <port> <tag> <out_dir> <adapter_dir|-> bench:ctx [bench:ctx ...]
set -u
source "$(dirname "$(readlink -f "$0")")/../env.sh"; ds_activate
GPU=$1; PORT=$2; TAG=$3; OUT=$4; ADAPTER=$5; shift 5
mkdir -p $OUT/logs
NAME=qwen3-32b; LORA=()
if [ "$ADAPTER" != "-" ]; then NAME=$TAG; LORA=(--enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules "$TAG=$ADAPTER"); fi
CUDA_VISIBLE_DEVICES=$GPU setsid $VLLM_BIN serve $MODEL_DIR --served-model-name qwen3-32b --port $PORT --host 127.0.0.1 \
  --max-model-len 131072 --gpu-memory-utilization 0.90 --enable-prefix-caching --max-num-seqs 64 \
  --override-generation-config '{"max_new_tokens": 8192}' "${LORA[@]}" > $OUT/logs/vllm_${TAG}.log 2>&1 &
S=$!
until curl -s 127.0.0.1:$PORT/v1/models | grep -q "\"$NAME\""; do
  sleep 10; kill -0 $S 2>/dev/null || { echo "vLLM died ($TAG), see $OUT/logs/vllm_${TAG}.log"; exit 1; }
done
for job in "$@"; do
  B=${job%%:*}; C=${job##*:}; ID=${TAG}_${C}_$B
  CTX=$DS_HOME/configs/empty_context.txt; SK=""
  [ "$C" = skills ] && CTX=$PLAYBOOK && SK=$PLAYBOOK
  STOP=$OUT/logs/.stop_$ID; rm -f $STOP
  python3 $DS_HOME/scripts/scrape_metrics.py http://127.0.0.1:$PORT $OUT/metrics_$ID.jsonl --stop-file $STOP &
  t0=$(date +%s)
  case $B in
    formula|finer) (cd $ACE_UPSTREAM_DIR && LOCAL_LLM_URL=http://127.0.0.1:$PORT/v1 EVAL_MODEL=$NAME \
                     $ACE_PY $ACE_ROOT/eval/eval_split.py $B test $CTX $OUT/eval_$ID.json > $OUT/logs/eval_$ID.log 2>&1) ;;
    gsm8k|mmlu)    $DS_ENV/bin/python $DS_HOME/scripts/eval_general.py --url http://127.0.0.1:$PORT/v1 --model $NAME \
                     --bench $B ${SK:+--skills $SK} --out $OUT/eval_$ID.json > $OUT/logs/eval_$ID.log 2>&1 ;;
  esac
  touch $STOP; wait $! 2>/dev/null
  echo "$(date '+%T') $(grep RESULT $OUT/logs/eval_$ID.log | tail -1) [$ID, $(( $(date +%s) - t0 )) s]"
done
kill -INT -$S 2>/dev/null; sleep 15; kill -KILL -$S 2>/dev/null; true
