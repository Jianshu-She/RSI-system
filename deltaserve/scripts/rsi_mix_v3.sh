#!/bin/bash
# Mixed RSI traffic (ACE formula + finer, TraceLab session timing) + several FT rounds, three ways on the same GPUs:
#   seq : DeltaServe; at each round the server stops serving (arrivals held, in-flight drained), trains the round
#         alone, then releases the held requests (time-disaggregated)
#   co  : DeltaServe co-serving; each round is ingested and trains while the server keeps serving
#   tt  : stock vLLM 0.21 serves + torchtune 0.6.1 (FSDP, its own copy of the weights) trains next to it, the GPUs
#         shared through NVIDIA MPS (vLLM VLLM_MPS %, torchtune TT_MPS %): DeltaServe-Baselines' vLLM+torchtune
# Rounds: <ROUNDS>/r<k>.jsonl from rsi_mix_prep.py, started at ROUNDS_AT (s after the first arrival).
# Usage: rsi_mix.sh <seq|co|tt> <gpus> <port> <out_dir>
#   env: WORKLOAD (agent_trace.py JSON) ROUNDS ROUNDS_AT (120,270,420,570) DURATION (750) MIX (1,2 = formula:finer)
#        DeltaServe: TTFT_SLO (0.3) MAX_TBT_SLO (0.05) PHASE (prefill|both) DSERVE_BACKWARD_NCCL_PORT
#        DECODE_MARGIN (slo.decode_only_ft_safety_margin, 0.7) PAUSE_MIN_PREFILL (finetune.pause_min_prefill_tokens, 0)
#        STALL_BUDGET (slo.ft_decode_stall_budget, 0 = off) STALL_WINDOW (slo.ft_stall_window_s, 2)
#        BWD_TF32=1 (finetune.backward_tf32) BWD_SDPA=1 (finetune.backward_attn_sdpa): the fast backward
#        STEP_TRACE=1 (DeltaServe per-step admission trace -> <out>/step_trace.csv)
#        tt: TT_SHARE (time | mps; default time) VLLM_MPS (100) TT_MPS (20, mps only) TT_GPU_UTIL (0.45) TT_MASTER_PORT (29517) BASELINES (DeltaServe-Baselines)
set -u
source "$(dirname "$(readlink -f "$0")")/../env.sh"; ds_activate
MODE=$1; GPUS=$2; PORT=$3; OUT=$4; TP=$(echo $GPUS | tr ',' '\n' | wc -l); URL=http://127.0.0.1:$PORT
ROUNDS_AT=${ROUNDS_AT:-120,270,420,570}; DURATION=${DURATION:-750}; MIX=${MIX:-1,2}
NR=$(echo $ROUNDS_AT | tr ',' '\n' | wc -l)
BASELINES=${BASELINES:-/mnt/weka/home/jianshu.she/deltaserve/DeltaServe-Baselines}
mkdir -p $OUT; rm -rf $OUT/adapters $OUT/traffic.jsonl* $OUT/rounds $OUT/progress.txt
T() { echo "=== $(date '+%F %T') $*" | tee -a $OUT/driver.log; }
cp -r $ROUNDS $OUT/rounds; rm -f $OUT/rounds/go* $OUT/rounds/waiting
PIDS=()
servers() { pgrep -f "serve .*--port $PORT( |$)"; }   # the launcher starts the server in its own session
targets() { { pgrep -f "$OUT/"; servers; } | sort -u | grep -vx "$$"; }   # this cell's processes only
signal() {  # signal each target's own process group, never the group this script shares with its batch
  local mypg=$(ps -o pgid= $$ | tr -d ' ') p pg
  for p in $(targets); do pg=$(ps -o pgid= $p 2>/dev/null | tr -d ' ')
    if [ -n "$pg" ] && [ "$pg" != "$mypg" ]; then kill -$1 -- -$pg 2>/dev/null; else kill -$1 $p 2>/dev/null; fi; done
}
cleanup() {
  signal INT
  for i in $(seq 30); do [ -z "$(targets)" ] && ! curl -s -o /dev/null $URL/health && break; sleep 2; done
  signal KILL
  [ -n "${MPS_PIPE:-}" ] && { echo quit | CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE nvidia-cuda-mps-control; rm -rf $MPS_PIPE; }
  sleep 5
}
# a server still answering on this port belongs to someone else (e.g. the previous cell's teardown): wait, then refuse
for i in $(seq 60); do curl -s -o /dev/null $URL/health || break; sleep 2; done
curl -s -o /dev/null $URL/health && { T "port $PORT is busy: refusing to start"; exit 1; }
up() { curl -s -o /dev/null $URL/health && grep -q "GPU KV cache size" $OUT/server.log; }
trap cleanup EXIT
rm -f $OUT/server.log

if [ $MODE = tt ]; then
  M=${TT_MODEL:-/mnt/weka/home/jianshu.she/scratch/Qwen3-32B}   # plain Qwen3-32B: lms_tune has no YaRN rope scaling
  # time (default): two plain CUDA contexts per GPU, time-sliced by the driver, like DeltaServe's backward child.
  # mps: NVIDIA MPS with per-client SM caps — on torch 2.11+cu129 a vLLM cap of 60-85 % (and 70 % on the full model)
  # makes cuBLAS bf16 GEMMs return garbage, and 100 % + 20 % deadlocked vLLM's TP all-reduce once torchtune trained.
  if [ "${TT_SHARE:-time}" = mps ]; then
    MPS_PIPE=$(mktemp -d /tmp/mps-XXXX); mkdir -p $OUT/mps-log
    export CUDA_VISIBLE_DEVICES=$GPUS CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE CUDA_MPS_LOG_DIRECTORY=$OUT/mps-log
    nvidia-cuda-mps-control -d || { T "MPS daemon failed"; exit 1; }
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TP - 1)))   # MPS clients index the daemon's visible set: $GPUS -> 0..TP-1
    SHARE="MPS: vLLM ${VLLM_MPS:-100}%, torchtune ${TT_MPS:-20}%"
  else
    export CUDA_VISIBLE_DEVICES=$GPUS; SHARE="time-sliced, no MPS"
  fi
  T "tt: vLLM + torchtune ($SHARE) on GPUs $GPUS (TP=$TP), rounds at $ROUNDS_AT"
  MPSV=(); MPST=(); [ -n "${MPS_PIPE:-}" ] && { MPSV=(CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${VLLM_MPS:-100}); MPST=(CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${TT_MPS:-20}); }
  env "${MPSV[@]}" VLLM_USE_FLASHINFER_SAMPLER=0 setsid \
    $BASELINES/.venvs/baseline-vllm/bin/vllm serve $M --served-model-name $DS_MODEL_NAME --tensor-parallel-size $TP \
    --port $PORT --max-model-len 16384 --gpu-memory-utilization ${TT_GPU_UTIL:-0.45} --enable-lora --max-lora-rank 16 \
    --max-num-batched-tokens 16384 --long-prefill-token-threshold 2048 > $OUT/server.log 2>&1 &
  PIDS+=($!)
  until up; do sleep 10; kill -0 ${PIDS[0]} 2>/dev/null || { T "vLLM died"; exit 1; }; done
  T "vLLM up: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $OUT/server.log | head -1)"
  FILES=$(python3 -c "import json;print(','.join(sorted(set(json.load(open('$M/model.safetensors.index.json'))['weight_map'].values()))))")
  (cd $OUT && env "${MPST[@]}" OMP_NUM_THREADS=4 \
    PYTHONPATH=$BASELINES/vllm-torchtune:$DS_HOME/baselines setsid $BASELINES/.venvs/baseline-tune/bin/tune run \
    --nnodes 1 --nproc_per_node $TP --master_port ${TT_MASTER_PORT:-29517} lora_finetune_distributed \
    --config $DS_HOME/baselines/tt_rsi/qwen3_32b_rsi_lora.yaml model_dir=$M output_dir=$OUT rounds_dir=$OUT/rounds \
    rounds=$NR "checkpointer.checkpoint_files=[$FILES]" > $OUT/torchtune.log 2>&1) &
  PIDS+=($!)
  until [ -f $OUT/rounds/waiting ]; do sleep 5; kill -0 ${PIDS[1]} 2>/dev/null || { T "torchtune died"; exit 1; }; done
  T "torchtune loaded, waiting for round 0"
  FT="--ft-backend tt --tt-progress $OUT/progress.txt"
else
  sed -e "s#__INIT_ADAPTER__#$DS_INIT_ADAPTER#g" -e "s#__SAVE_DIR__#$OUT/adapters#" -e "s#__BWD_LOG__#$OUT/bwd.csv#" \
      -e "s#__MODEL_DIR__#$MODEL_DIR#" -e "s#__TP__#$TP#" -e "s#__DATA__#$OUT/rounds/r0.jsonl#" \
      -e "s#^  port: 8000#  port: $PORT#" -e "s#^  num_epochs: .*#  num_epochs: 1#" \
      -e "s#^  profile_on_launch: .*#  profile_on_launch: true#" \
      -e "s#^  ft_tokens_admission_constrain_factor: .*#  ft_tokens_admission_constrain_factor: -1#" \
      -e "s#^  rps_throttle_enable: .*#  rps_throttle_enable: false#" \
      -e "s#^  ttft_slo: .*#  ttft_slo: ${TTFT_SLO:-0.3}#" -e "s#^  max_tbt_slo: .*#  max_tbt_slo: ${MAX_TBT_SLO:-0.05}#" \
      $DS_HOME/configs/train_only.yaml > $OUT/config.yaml
  sed -i "s#^slo:#slo:\n  coserving_admission_phase: ${PHASE:-prefill}#" $OUT/config.yaml
  TF32=false; [ -n "${BWD_TF32:-}" ] && TF32=true; SDPA=false; [ -n "${BWD_SDPA:-}" ] && SDPA=true
  sed -i -e "s#^  decode_only_ft_safety_margin: .*#  decode_only_ft_safety_margin: ${DECODE_MARGIN:-0.7}#" \
         -e "s#^slo:#slo:\n  ft_decode_stall_budget: ${STALL_BUDGET:-0.0}\n  ft_stall_window_s: ${STALL_WINDOW:-2.0}#" \
         -e "s#^finetune:#finetune:\n  pause_min_prefill_tokens: ${PAUSE_MIN_PREFILL:-0}\n  backward_tf32: $TF32\n  backward_attn_sdpa: $SDPA#" $OUT/config.yaml
  [ -n "${STEP_TRACE:-}" ] && sed -i "s#^finetune:#finetune:\n  step_trace_path: $OUT/step_trace.csv#" $OUT/config.yaml
  rm -f $OUT/bwd.csv
  T "$MODE: DeltaServe on GPUs $GPUS (TP=$TP), SLO ttft ${TTFT_SLO:-0.3} max_tbt ${MAX_TBT_SLO:-0.05} phase ${PHASE:-prefill}, decode margin ${DECODE_MARGIN:-0.7}, pause>=${PAUSE_MIN_PREFILL:-0} tok, stall budget ${STALL_BUDGET:-0}, bwd tf32 ${BWD_TF32:-0} sdpa ${BWD_SDPA:-0}, rounds at $ROUNDS_AT"
  (cd $DS_ROOT && CUDA_VISIBLE_DEVICES=$GPUS setsid $DS_ENV/bin/python eval-tp/launch_deltaserve.py \
    --config $OUT/config.yaml --port $PORT --startup-timeout 3000 > $OUT/server.log 2>&1) &
  PIDS+=($!)
  until up; do sleep 10; kill -0 ${PIDS[0]} 2>/dev/null || { T "server died"; exit 1; }; done
  T "server up: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $OUT/server.log | head -1)"
  FT="--ft-backend ds --bwd-csv $OUT/bwd.csv"
fi
setsid bash $DS_HOME/scripts/gpu_monitor.sh $OUT/gpu.txt & PIDS+=($!)
TM=$MODE; [ $MODE = tt ] && TM=co
(cd $ACE_UPSTREAM_DIR && $ACE_PY $DS_HOME/scripts/rsi_traffic.py --url $URL/v1 --model $DS_MODEL_NAME --task formula,finer \
    --mix $MIX --split train --workload $WORKLOAD --duration $DURATION --train-mode $TM --rounds-at $ROUNDS_AT \
    --rounds-dir $OUT/rounds $FT --out $OUT/traffic.jsonl) | tail -1 | tee -a $OUT/driver.log
T "$MODE done"
