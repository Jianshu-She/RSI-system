#!/bin/bash
# DeltaServe as a plain trainer: train an RSI-system SFT file (chat JSONL) to final/ with no inference traffic.
# Usage: bash deltaserve/scripts/run_train.sh <sft.jsonl> [run_name]     e.g. $ACE_SFT_DIR/data_v2/formula.jsonl
# Then:  bash deltaserve/scripts/eval_adapter.sh $EVAL_GPU 9300 formula ds <run>/eval <run>/adapters/final
set -u
source "$(dirname "$(readlink -f "$0")")/../env.sh"; ds_activate
DATA=$(readlink -f $1); RUN=$DS_WORK/runs/${2:-train_$(basename $DATA .jsonl)_$(date +%Y%m%d_%H%M%S)}
mkdir -p $RUN; rm -rf $RUN/adapters; URL=http://127.0.0.1:$DS_PORT; TP=$(echo $DS_GPUS | tr ',' '\n' | wc -l)
sed -e "s#__INIT_ADAPTER__#$DS_INIT_ADAPTER#g" -e "s#__SAVE_DIR__#$RUN/adapters#" -e "s#__BWD_LOG__#$RUN/bwd.csv#" \
    -e "s#__MODEL_DIR__#$MODEL_DIR#" -e "s#__TP__#$TP#" -e "s#__DATA__#$DATA#" -e "s#^  port: 8000#  port: $DS_PORT#" \
    $DS_HOME/configs/train_only.yaml > $RUN/config.yaml
echo "=== $(date '+%T') train $DATA on GPUs $DS_GPUS -> $RUN"
cd $DS_ROOT
CUDA_VISIBLE_DEVICES=$DS_GPUS setsid $DS_ENV/bin/python eval-tp/launch_deltaserve.py --config $RUN/config.yaml \
    --port $DS_PORT --start-finetuning --startup-timeout 3000 > $RUN/server.log 2>&1 &
L=$!
until curl -s -o /dev/null $URL/health; do sleep 10; kill -0 $L 2>/dev/null || { echo "server died: see $RUN/server.log"; exit 1; }; done
sleep 20
curl -s $URL/v1/completions -H 'Content-Type: application/json' -d "{\"model\":\"$DS_MODEL_NAME\",\"prompt\":\"hi\",\"max_tokens\":2}" > /dev/null   # wake the idle engine loop
for i in $(seq 720); do [ -f $RUN/adapters/final/adapter_model.safetensors ] && break; kill -0 $L 2>/dev/null || break; sleep 10; done
SPID=$(pgrep -f "dserve-vllm serve.*$RUN/bwd.csv" | head -1)
kill -INT -$L ${SPID:+-$SPID} 2>/dev/null; sleep 30; kill -KILL -$L ${SPID:+-$SPID} 2>/dev/null
ls $RUN/adapters/final && echo "=== $(date '+%T') final adapter: $RUN/adapters/final"
