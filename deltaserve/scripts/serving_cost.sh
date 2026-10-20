#!/bin/bash
# Serving cost of "skills in context" vs "skills in a LoRA", same traffic, same GPUs, stock vLLM 0.31 (TP = #GPUs):
#   P  plain server, base model:  formula test traffic with the playbook in the prompt  (skills)   and with "(empty)" (base)
#   L  LoRA server,  RSI adapter: formula test traffic with "(empty)"                   (lora)
# Each condition at every rate in RATES (Poisson, one pass over the 200 test items). Per request: TTFT, TBT, E2E,
# prompt/completion tokens (traffic_<cond>_r<rate>.jsonl); per second: KV-cache usage, queue, prefix-cache hits
# (metrics_<server>.jsonl). Then the prompt-extraction probe on each deployment (leak_<cond>.json).
# Usage: serving_cost.sh <gpus> <port> <out_dir> <adapter_dir>
#   env: RATES ("1 4")  SPLIT (test; use train = 500 items for a fresh pass)  SERVERS ("P L")  LEAK (1)
#        NOPC=1: prefix caching OFF (= every user brings different skills: nothing is shared)
#        SERVERS containing M: also a plain server on MERGED_DIR (the LoRA merged into the weights): merged, merged_skills
set -u
source "$(dirname "$(readlink -f "$0")")/../env.sh"; ds_activate
GPUS=$1; PORT=$2; OUT=$3; ADAPTER=$4; RATES=${RATES:-"1 4"}; SPLIT=${SPLIT:-test}; SERVERS=${SERVERS:-"P L"}
LEAK=${LEAK:-1}; PC=--enable-prefix-caching; [ -n "${NOPC:-}" ] && PC=--no-enable-prefix-caching
TP=$(echo $GPUS | tr ',' '\n' | wc -l); URL=http://127.0.0.1:$PORT
mkdir -p $OUT/logs
T() { echo "=== $(date '+%F %T') $*" | tee -a $OUT/driver.log; }
traffic() { (cd $ACE_UPSTREAM_DIR && $ACE_PY $DS_HOME/scripts/rsi_traffic.py --url $URL/v1 "$@"); }
leak() { (cd $ACE_UPSTREAM_DIR && $ACE_PY $DS_HOME/scripts/leak_probe.py --url $URL/v1 --playbook $PLAYBOOK "$@"); }
serve() {  # serve <name> [lora args]   (SERVE_MODEL overrides the weights dir)
  local N=$1; shift
  CUDA_VISIBLE_DEVICES=$GPUS setsid $VLLM_BIN serve ${SERVE_MODEL:-$MODEL_DIR} --served-model-name qwen3-32b --port $PORT --host 127.0.0.1 \
    -tp $TP --max-model-len 131072 --gpu-memory-utilization 0.90 $PC --max-num-seqs 64 \
    --override-generation-config '{"max_new_tokens": 8192}' "$@" > $OUT/logs/vllm_$N.log 2>&1 &
  SV=$!
  until curl -s $URL/v1/models | grep -q qwen3-32b; do sleep 10; kill -0 $SV 2>/dev/null || { T "vLLM $N died"; exit 1; }; done
  rm -f $OUT/logs/.stop_$N
  python3 $DS_HOME/scripts/scrape_metrics.py $URL $OUT/metrics_$N.jsonl --stop-file $OUT/logs/.stop_$N & SC=$!
  T "server $N up: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $OUT/logs/vllm_$N.log | head -1)"
}
stop() { touch $OUT/logs/.stop_$1; wait $SC 2>/dev/null; kill -INT -$SV 2>/dev/null; sleep 20; kill -KILL -$SV 2>/dev/null; sleep 5; }
run() {  # run <cond> <model> [--context F]
  local C=$1 M=$2; shift 2
  for r in $RATES; do
    T "$C @ $r rps"
    traffic --model $M --task formula --split $SPLIT --rps $r "$@" --out $OUT/traffic_${C}_r$r.jsonl | tail -1 | tee -a $OUT/driver.log
  done
}

if [[ " $SERVERS " == *" P "* ]]; then
  T "P: plain server (base) on GPUs $GPUS, TP=$TP ($PC)"
  serve P
  run skills qwen3-32b --context $PLAYBOOK
  run base qwen3-32b
  if [ "$LEAK" = 1 ]; then
    leak --model qwen3-32b --in-context --out $OUT/leak_skills.json | tee -a $OUT/driver.log
    leak --model qwen3-32b --out $OUT/leak_base.json | tee -a $OUT/driver.log
  fi
  stop P
fi
if [[ " $SERVERS " == *" L "* ]]; then
  T "L: LoRA server (rsi adapter, $PC)"
  serve L --enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules rsi=$ADAPTER
  run lora rsi
  run lora_skills rsi --context $PLAYBOOK
  [ "$LEAK" = 1 ] && leak --model rsi --out $OUT/leak_lora.json | tee -a $OUT/driver.log
  stop L
fi
if [[ " $SERVERS " == *" M "* ]]; then
  T "M: plain server on the merged weights $MERGED_DIR ($PC)"
  SERVE_MODEL=$MERGED_DIR serve M
  run merged qwen3-32b
  run merged_skills qwen3-32b --context $PLAYBOOK
  stop M
fi
T "serving cost done"
