#!/bin/bash
# Sequential (time-disaggregated) vs co-serving on the same GPUs, agent workload from TraceLab (agent_trace.py).
#   inf : inference only — the agent workload on a plain serving deployment: stock vLLM ($VLLM_BIN, the same server
#         the paper-1 serving-cost runs use), gpu_memory_utilization 0.90, prefix caching, same max_model_len /
#         step budget / chunking as the DeltaServe config. (The DeltaServe fork with finetuning off hit a flash-attn
#         illegal memory access on this workload; agent-1 notes.)
#   ft  : training only  — the FT job (DATA x EPOCHS) with no inference traffic
#   co  : co-serving     — the agent workload AND the FT job on one DeltaServe server, both started at t=0
# Sequential completion time = inf makespan + ft time (or ft first, then inf: every request waits for ft).
# Usage: agent_run.sh <inf|ft|co> <gpus> <port> <out_dir>     env: WORKLOAD DATA EPOCHS (4)
set -u
source "$(dirname "$(readlink -f "$0")")/../env.sh"; ds_activate
MODE=$1; GPUS=$2; PORT=$3; OUT=$4; EPOCHS=${EPOCHS:-4}
TP=$(echo $GPUS | tr ',' '\n' | wc -l); URL=http://127.0.0.1:$PORT
mkdir -p $OUT; rm -rf $OUT/adapters
T() { echo "=== $(date '+%F %T') $*" | tee -a $OUT/driver.log; }
SLO_PROFILE=false; [ $MODE = co ] && SLO_PROFILE=true
sed -e "s#__INIT_ADAPTER__#$DS_INIT_ADAPTER#g" -e "s#__SAVE_DIR__#$OUT/adapters#" -e "s#__BWD_LOG__#$OUT/bwd.csv#" \
    -e "s#__MODEL_DIR__#$MODEL_DIR#" -e "s#__TP__#$TP#" -e "s#__DATA__#$DATA#" -e "s#^  port: 8000#  port: $PORT#" \
    -e "s#^  num_epochs: .*#  num_epochs: $EPOCHS#" -e "s#^  profile_on_launch: .*#  profile_on_launch: $SLO_PROFILE#" \
    $DS_HOME/configs/train_only.yaml > $OUT/config.yaml
cd $DS_ROOT
if [ $MODE = inf ]; then  # a plain serving deployment: no FT buffers, so vLLM gets the memory for KV
  T "$MODE: stock vLLM on GPUs $GPUS (TP=$TP)"
  CUDA_VISIBLE_DEVICES=$GPUS setsid $VLLM_BIN serve $MODEL_DIR --served-model-name $DS_MODEL_NAME --port $PORT \
      --host 127.0.0.1 -tp $TP --max-model-len 40960 --gpu-memory-utilization 0.90 --enable-prefix-caching \
      --max-num-batched-tokens 16384 --long-prefill-token-threshold 2048 --seed 0 > $OUT/server.log 2>&1 &
else
  T "$MODE: DeltaServe on GPUs $GPUS (TP=$TP), epochs $EPOCHS, data $DATA"
  CUDA_VISIBLE_DEVICES=$GPUS setsid $DS_ENV/bin/python eval-tp/launch_deltaserve.py --config $OUT/config.yaml \
      --port $PORT --startup-timeout 3000 > $OUT/server.log 2>&1 &
fi
L=$!
until curl -s -o /dev/null $URL/health; do sleep 10; kill -0 $L 2>/dev/null || { T "server died: see $OUT/server.log"; exit 1; }; done
T "server up: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $OUT/server.log | head -1)"
setsid bash $DS_HOME/scripts/gpu_monitor.sh $OUT/gpu.txt & GM=$!

replay() { $ACE_PY $DS_HOME/scripts/agent_replay.py --url $URL/v1 --model $DS_MODEL_NAME --workload $WORKLOAD \
           --out $OUT/rounds.jsonl | tee -a $OUT/driver.log; }
ft_done() { [ -f $OUT/adapters/final/adapter_model.safetensors ]; }
wait_ft() { for i in $(seq 2160); do ft_done && break; kill -0 $L 2>/dev/null || break; sleep 5; done; }
if [ $MODE != inf ]; then
  T "FT start"; date +%s.%N > $OUT/t_ft_start
  curl -s -X POST $URL/start_finetuning | tee -a $OUT/driver.log; echo
  curl -s $URL/v1/completions -H 'Content-Type: application/json' \
       -d "{\"model\":\"$DS_MODEL_NAME\",\"prompt\":\"hi\",\"max_tokens\":2}" > /dev/null   # wake the idle engine loop
fi
case $MODE in
  inf) T "replay start"; date +%s.%N > $OUT/t_replay_start; replay; date +%s.%N > $OUT/t_replay_end; T "replay done" ;;
  ft)  wait_ft ;;
  co)  T "replay start"; date +%s.%N > $OUT/t_replay_start
       replay; date +%s.%N > $OUT/t_replay_end; T "replay done"; wait_ft ;;
esac
if [ $MODE != inf ]; then
  ft_done && { date +%s.%N > $OUT/t_ft_end; T "FT final adapter exported"; } \
          || T "WARNING: FT did not finish"
fi
kill -- -$GM 2>/dev/null
SPID=$(pgrep -f "dserve-vllm serve.*$OUT/bwd.csv" | head -1)
kill -INT -$L ${SPID:+-$SPID} 2>/dev/null; sleep 30; kill -KILL -$L ${SPID:+-$SPID} 2>/dev/null
T "$MODE done"
