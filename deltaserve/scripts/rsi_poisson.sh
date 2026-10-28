#!/bin/bash
# RSI traffic with Poisson arrivals + the RSI FT job on ONE DeltaServe server: sequential vs co-serving.
#   traffic : ACE formula requests (SPLIT, empty playbook) to the base model, Poisson at RATE rps, one pass
#   FT job  : DATA (the loop's SFT rows) x EPOCHS, starting at TRAIN_AT s after the first arrival
#   co  : FT starts and the server keeps serving
#   seq : the server stops serving for the FT job: arrivals are held, in-flight requests drain, FT runs alone,
#         then the held requests are released (time-disaggregated on the same GPUs)
# Both modes use the same server config (train_only.yaml + the SLO profiling pass), so the engine is identical.
# Usage: rsi_poisson.sh <co|seq> <gpus> <port> <out_dir>
#   env: RATE (0.5) DATA EPOCHS (2) TRAIN_AT (120) SPLIT (train) DSERVE_BACKWARD_NCCL_PORT (one per server on a node)
#   WORKLOAD: an agent_trace.py JSON -> TraceLab session timing instead of per-request Poisson at RATE
#   FT_ADMIT: idle (default: train_only.yaml as is: FT only in fully idle steps + the 0.1 s RPS throttle — the 16K
#             long-context settings) | ride (DeltaServe's standard admission: FT samples ride inference steps as far
#             as the SLO estimator allows, ft_tokens_admission_constrain_factor -1, no RPS throttle)
set -u
source "$(dirname "$(readlink -f "$0")")/../env.sh"; ds_activate
MODE=$1; GPUS=$2; PORT=$3; OUT=$4; RATE=${RATE:-0.5}; EPOCHS=${EPOCHS:-2}; TRAIN_AT=${TRAIN_AT:-120}
SPLIT=${SPLIT:-train}; TP=$(echo $GPUS | tr ',' '\n' | wc -l); URL=http://127.0.0.1:$PORT
mkdir -p $OUT; rm -rf $OUT/adapters $OUT/traffic.jsonl*
T() { echo "=== $(date '+%F %T') $*" | tee -a $OUT/driver.log; }
sed -e "s#__INIT_ADAPTER__#$DS_INIT_ADAPTER#g" -e "s#__SAVE_DIR__#$OUT/adapters#" -e "s#__BWD_LOG__#$OUT/bwd.csv#" \
    -e "s#__MODEL_DIR__#$MODEL_DIR#" -e "s#__TP__#$TP#" -e "s#__DATA__#$DATA#" -e "s#^  port: 8000#  port: $PORT#" \
    -e "s#^  num_epochs: .*#  num_epochs: $EPOCHS#" -e "s#^  profile_on_launch: .*#  profile_on_launch: true#" \
    $DS_HOME/configs/train_only.yaml > $OUT/config.yaml
if [ "${FT_ADMIT:-idle}" = ride ]; then
  sed -i -e "s#^  ft_tokens_admission_constrain_factor: .*#  ft_tokens_admission_constrain_factor: -1#" \
         -e "s#^  rps_throttle_enable: .*#  rps_throttle_enable: false#" $OUT/config.yaml
fi
cd $DS_ROOT
T "$MODE @ $RATE rps (FT admission ${FT_ADMIT:-idle}): DeltaServe on GPUs $GPUS (TP=$TP), FT $DATA x $EPOCHS at t=${TRAIN_AT}s"
CUDA_VISIBLE_DEVICES=$GPUS setsid $DS_ENV/bin/python eval-tp/launch_deltaserve.py --config $OUT/config.yaml \
    --port $PORT --startup-timeout 3000 > $OUT/server.log 2>&1 &
L=$!
until curl -s -o /dev/null $URL/health; do sleep 10; kill -0 $L 2>/dev/null || { T "server died: see $OUT/server.log"; exit 1; }; done
T "server up: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $OUT/server.log | head -1)"
setsid bash $DS_HOME/scripts/gpu_monitor.sh $OUT/gpu.txt & GM=$!
(cd $ACE_UPSTREAM_DIR && $ACE_PY $DS_HOME/scripts/rsi_traffic.py --url $URL/v1 --model $DS_MODEL_NAME --task formula \
    --split $SPLIT --rps $RATE ${WORKLOAD:+--workload $WORKLOAD} --train-at $TRAIN_AT --train-mode $MODE \
    --ft-final $OUT/adapters/final/adapter_model.safetensors --out $OUT/traffic.jsonl) | tail -1 | tee -a $OUT/driver.log
kill -- -$GM 2>/dev/null
SPID=$(pgrep -f "dserve-vllm serve.*$OUT/bwd.csv" | head -1)
kill -INT -$L ${SPID:+-$SPID} 2>/dev/null; sleep 30; kill -KILL -$L ${SPID:+-$SPID} 2>/dev/null
T "$MODE @ $RATE done"
