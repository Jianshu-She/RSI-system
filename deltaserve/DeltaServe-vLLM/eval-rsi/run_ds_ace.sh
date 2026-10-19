#!/bin/bash
# Train an RSI-system ACE Finance adapter with DeltaServe (TP=4, training only) and stop once
# eval-rsi/adapters/<task>/final is exported.   Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 eval-rsi/run_ds_ace.sh <formula|finer>
set -u
ROOT=/mnt/weka/home/jianshu.she/deltaserve/DeltaServe-vLLM; cd $ROOT
ENV=/mnt/weka/home/jianshu.she/miniconda3/envs/dserve-vllm
export FLASHINFER_WORKSPACE_BASE=$HOME/.cache/flashinfer-dserve CUDA_HOME=/usr/local/cuda-13.2 \
       PATH=/usr/local/cuda-13.2/bin:$ENV/bin:$PATH HF_HUB_OFFLINE=1 no_proxy=localhost,127.0.0.1
T=$1; PORT=${PORT:-8000}
OUT=$ROOT/eval-rsi/adapters/$T; LOG=$ROOT/eval-rsi/logs; mkdir -p $LOG; rm -rf $OUT
CFG=$ROOT/configs/.gen_rsi_ace_$T.yaml
sed -e "s#sft/data_v2/formula.jsonl#sft/data_v2/$T.jsonl#" -e "s#eval-rsi/adapters/formula#eval-rsi/adapters/$T#" \
    -e "s#^  adapter_save_every: 20#  adapter_save_every: 20\n  bwd_log_path: $LOG/bwd_$T.csv#" \
    configs/serving_config_finetuning_qwen3_32b_tp4_rsi_ace.yaml > $CFG
rm -f $LOG/bwd_$T.csv
echo "=== $(date '+%F %T') $(hostname) DS train $T (GPUs $CUDA_VISIBLE_DEVICES)"
setsid $ENV/bin/python eval-tp/launch_deltaserve.py --config $CFG --port $PORT --start-finetuning \
    --startup-timeout 2400 > $LOG/ds_$T.log 2>&1 &
L=$!
until curl -s -o /dev/null 127.0.0.1:$PORT/health; do sleep 10; kill -0 $L 2>/dev/null || { echo "launcher died"; exit 1; }; done
sleep 20   # let /start_finetuning land, then kick the idle engine loop once
curl -s 127.0.0.1:$PORT/v1/completions -H 'Content-Type: application/json' \
     -d '{"model":"llama3","prompt":"Hello","max_tokens":2}' > /dev/null
echo "=== $(date '+%T') serving + training"
for i in $(seq 720); do
  [ -f $OUT/final/adapter_model.safetensors ] && break
  kill -0 $L 2>/dev/null || { echo "launcher died"; break; }
  sleep 10
done
ls $OUT/final 2>/dev/null && echo "=== $(date '+%T') final adapter exported"
# The server runs in its own session (not the launcher's process group): find it by its
# per-task bwd log path, SIGINT its group, then SIGKILL whatever is left. Return only when
# port $PORT is free so the next task cannot hit the old server's /health.
SPID=$(pgrep -f "dserve-vllm serve.*bwd_$T.csv" | head -1)
kill -INT -$L ${SPID:+-$SPID} 2>/dev/null; sleep 30
kill -KILL -$L ${SPID:+-$SPID} 2>/dev/null
for i in $(seq 30); do curl -s -o /dev/null 127.0.0.1:$PORT/health || break; sleep 5; done
echo "=== $(date '+%F %T') done $T"
