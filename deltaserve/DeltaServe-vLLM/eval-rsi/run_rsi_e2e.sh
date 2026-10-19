#!/bin/bash
# End-to-end RSI on ONE DeltaServe server (ACE Formula, Qwen3-32B YaRN, TP=4 on GPUs 0-3):
#   A  inference only : test-split traffic (empty playbook), Poisson RPS_A      -> latency baseline + base acc
#   B  user traffic   : train-split requests WITH the learned playbook          -> traces (prompt, answer, correct)
#   I  traces -> SFT  : correct only, playbook -> "(empty)", bullet_ids -> []   -> POST /ft/ingest, /start_finetuning
#   C  co-serving     : the same test-split traffic as A while the LoRA trains in the gaps, until final/ is exported
#   D  eval           : the exported adapter with RSI-system's protocol (stock vLLM 0.31, GPU 4)
# Usage (inside the allocation):  eval-rsi/run_rsi_e2e.sh
set -u
ROOT=/mnt/weka/home/jianshu.she/deltaserve/DeltaServe-vLLM; cd $ROOT
ENV=/mnt/weka/home/jianshu.she/miniconda3/envs/dserve-vllm
ACE=/mnt/weka/home/jianshu.she/RSI-system/benchmark/ACE
PLAYBOOK=$(ls $ACE/results/finance/formula_offline_gt/*/best_playbook.txt | head -1)
export FLASHINFER_WORKSPACE_BASE=$HOME/.cache/flashinfer-dserve CUDA_HOME=/usr/local/cuda-13.2 \
       PATH=/usr/local/cuda-13.2/bin:$ENV/bin:$PATH HF_HUB_OFFLINE=1 no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
E=$ROOT/eval-rsi/e2e; mkdir -p $E; rm -rf $E/adapters; PORT=8000; URL=http://127.0.0.1:$PORT
RPS_A=${RPS_A:-0.5}
CFG=$ROOT/configs/.gen_rsi_e2e.yaml
sed -e "s#^  adapter_save_every: 10#  adapter_save_every: 10\n  bwd_log_path: $E/bwd.csv#" \
    configs/serving_config_finetuning_qwen3_32b_tp4_rsi_e2e.yaml > $CFG
rm -f $E/bwd.csv $E/A.jsonl $E/B.jsonl $E/C.jsonl
T() { echo "=== $(date '+%F %T') $*"; }
traffic() { (cd $ACE/ace && .venv/bin/python $ROOT/eval-rsi/rsi_traffic.py --url $URL/v1 --model llama3 "$@"); }

T "start DeltaServe (GPUs $CUDA_VISIBLE_DEVICES)"
setsid $ENV/bin/python eval-tp/launch_deltaserve.py --config $CFG --port $PORT --startup-timeout 3000 > $E/server.log 2>&1 &
L=$!
until curl -s -o /dev/null $URL/health; do sleep 10; kill -0 $L 2>/dev/null || { T "launcher died"; exit 1; }; done
T "server up"

T "A: inference only, test split, ${RPS_A} rps"
traffic --task formula --split test --rps $RPS_A --out $E/A.jsonl | tail -1

T "B: user traffic with the learned playbook ($PLAYBOOK), train split"
traffic --task formula --split train --context $PLAYBOOK --concurrency 16 --out $E/B.jsonl | tail -1

T "I: traces -> SFT rows -> /ft/ingest, then /start_finetuning"
python3 eval-rsi/traces_to_sft.py $E/B.jsonl $E/sft_from_traces.jsonl --ingest $URL
curl -s -X POST $URL/start_finetuning; echo

T "C: co-serving — same test traffic while training, until final/ is exported"
traffic --task formula --split test --rps $RPS_A --loops 20 --stop-file $E/adapters/final/adapter_model.safetensors \
    --out $E/C.jsonl | tail -1
for i in $(seq 360); do [ -f $E/adapters/final/adapter_model.safetensors ] && break; sleep 10; done
ls $E/adapters/final && T "final adapter exported"

SPID=$(pgrep -f "dserve-vllm serve.*$E/bwd.csv" | head -1)
kill -INT -$L ${SPID:+-$SPID} 2>/dev/null; sleep 30; kill -KILL -$L ${SPID:+-$SPID} 2>/dev/null
T "server stopped"

T "D: eval the RSI adapter (RSI-system protocol, GPU 4)"
cp -r $E/adapters/final $E/adapter_rsi_final
eval-rsi/eval_ace.sh 4 9334 formula e2e_rsi_formula $E/adapter_rsi_final
T "done"
