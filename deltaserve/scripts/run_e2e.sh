#!/bin/bash
# End-to-end RSI on ONE DeltaServe server (ACE Formula, Qwen3-32B YaRN x4, TP = #DS_GPUS):
#   A  inference only : formula TEST traffic, empty playbook, Poisson RPS        -> latency baseline + base accuracy
#   B  user traffic   : formula TRAIN requests WITH the learned playbook          -> traces (prompt, answer, correct)
#   I  traces -> SFT  : correct only, playbook -> "(empty)", bullet_ids -> []     -> POST /ft/ingest + /start_finetuning
#   C  co-serving     : the same traffic as A while the LoRA trains in the gaps   -> latency under training; final/ adapter
#   D  eval           : the exported adapter with the ACE protocol (stock vLLM 0.31 on EVAL_GPU) + paired test vs base
# Usage (on an 8-GPU node, after setup.sh):   bash deltaserve/scripts/run_e2e.sh [run_name]
# Knobs (env): RPS (0.5)  B_CONCURRENCY (16)  DS_GPUS (0,1,2,3)  EVAL_GPU (4)  PLAYBOOK  SKIP_BASE_EVAL=1  SKIP_EVAL=1 (no phase D)
set -u
source "$(dirname "$(readlink -f "$0")")/../env.sh"; ds_activate
RUN=$DS_WORK/runs/${1:-e2e_$(date +%Y%m%d_%H%M%S)}; mkdir -p $RUN; rm -rf $RUN/adapters
RPS=${RPS:-0.5}; BC=${B_CONCURRENCY:-16}; URL=http://127.0.0.1:$DS_PORT
TP=$(echo $DS_GPUS | tr ',' '\n' | wc -l)
T() { echo "=== $(date '+%F %T') $*" | tee -a $RUN/driver.log; }
[ -f "$PLAYBOOK" ] || { echo "PLAYBOOK not found ($PLAYBOOK): run the ACE formula offline+GT adaptation first"; exit 1; }
[ -f "$DS_INIT_ADAPTER/adapter_model.safetensors" ] || { echo "no $DS_INIT_ADAPTER: run deltaserve/scripts/setup.sh"; exit 1; }
sed -e "s#__INIT_ADAPTER__#$DS_INIT_ADAPTER#g" -e "s#__SAVE_DIR__#$RUN/adapters#" -e "s#__BWD_LOG__#$RUN/bwd.csv#" \
    -e "s#__MODEL_DIR__#$MODEL_DIR#" -e "s#__TP__#$TP#" -e "s#^  port: 8000#  port: $DS_PORT#" \
    $DS_HOME/configs/coserve_e2e.yaml > $RUN/config.yaml
traffic() { (cd $ACE_UPSTREAM_DIR && $ACE_PY $DS_HOME/scripts/rsi_traffic.py --url $URL/v1 --model $DS_MODEL_NAME "$@"); }

T "start DeltaServe on GPUs $DS_GPUS (TP=$TP); first launch compiles for 10+ min"
cd $DS_ROOT
CUDA_VISIBLE_DEVICES=$DS_GPUS setsid $DS_ENV/bin/python eval-tp/launch_deltaserve.py --config $RUN/config.yaml \
    --port $DS_PORT --startup-timeout 3000 > $RUN/server.log 2>&1 &
L=$!
until curl -s -o /dev/null $URL/health; do sleep 10; kill -0 $L 2>/dev/null || { T "server died: see $RUN/server.log"; exit 1; }; done
T "server up"

T "A: inference only (formula test, ${RPS} rps)"
traffic --task formula --split test --rps $RPS --out $RUN/A.jsonl | tail -1 | tee -a $RUN/driver.log

T "B: user traffic with the learned playbook (formula train, concurrency $BC)"
traffic --task formula --split train --context $PLAYBOOK --concurrency $BC --out $RUN/B.jsonl | tail -1 | tee -a $RUN/driver.log

T "I: traces -> SFT rows -> /ft/ingest -> /start_finetuning"
python3 $DS_HOME/scripts/traces_to_sft.py $RUN/B.jsonl $RUN/sft_from_traces.jsonl --ingest $URL | tee -a $RUN/driver.log
curl -s -X POST $URL/start_finetuning | tee -a $RUN/driver.log; echo

T "C: co-serving — the same traffic as A while training, until final/ is exported"
traffic --task formula --split test --rps $RPS --loops 50 --stop-file $RUN/adapters/final/adapter_model.safetensors \
    --out $RUN/C.jsonl | tail -1 | tee -a $RUN/driver.log
for i in $(seq 360); do [ -f $RUN/adapters/final/adapter_model.safetensors ] && break; sleep 10; done
[ -f $RUN/adapters/final/adapter_model.safetensors ] && T "final adapter exported" || T "WARNING: no final/ adapter"

SPID=$(pgrep -f "dserve-vllm serve.*$RUN/bwd.csv" | head -1)
kill -INT -$L ${SPID:+-$SPID} 2>/dev/null; sleep 30; kill -KILL -$L ${SPID:+-$SPID} 2>/dev/null
T "server stopped"

[ -n "${SKIP_EVAL:-}" ] && { python3 $DS_HOME/scripts/summarize.py $RUN | tee $RUN/summary.txt; T "done (SKIP_EVAL): $RUN"; exit 0; }
T "D: evaluate (ACE protocol, GPU $EVAL_GPU)"
bash $DS_HOME/scripts/eval_adapter.sh $EVAL_GPU 9334 formula rsi $RUN/eval $RUN/adapters/final | tee -a $RUN/driver.log
[ -z "${SKIP_BASE_EVAL:-}" ] && bash $DS_HOME/scripts/eval_adapter.sh $EVAL_GPU 9335 formula base $RUN/eval | tee -a $RUN/driver.log
python3 $DS_HOME/scripts/compare.py $RUN/eval formula base rsi | tee -a $RUN/driver.log
python3 $DS_HOME/scripts/summarize.py $RUN | tee $RUN/summary.txt
T "done: $RUN"
