#!/bin/bash
# Everything the paper needs from one 8-GPU node, in one run directory:
#   GPUs 0-3  run_e2e.sh phases A/B/I/C on DeltaServe (co-serving: latency, SLO, training, GPU utilization)
#             then serving_cost.sh on stock vLLM TP=4 (skills-in-context vs LoRA: TTFT/TBT/KV/prompt tokens + leakage)
#   GPUs 4,5  base model evals (ACE formula/finer, GSM8K, MMLU) x {empty, skills}   — while DeltaServe runs
#   GPUs 6,7  RSI adapter evals, same grid                                          — once the adapter is exported
#   all GPUs  DCGM utilization at 1 Hz (gpu.txt)
# then paper_metrics.py -> paper_metrics.json + paper_metrics.md
# Usage: bash deltaserve/scripts/paper_run.sh [run_name]
set -u
source "$(dirname "$(readlink -f "$0")")/../env.sh"; ds_activate
NAME=${1:-paper_$(date +%Y%m%d_%H%M%S)}; RUN=$DS_WORK/runs/$NAME; mkdir -p $RUN/eval/logs
SC=$DS_HOME/scripts
T() { echo "=== $(date '+%F %T') $*" | tee -a $RUN/paper.log; }
ACE_JOBS="formula:empty formula:skills finer:empty finer:skills"
GEN_JOBS="gsm8k:empty gsm8k:skills mmlu:empty mmlu:skills"

bash $SC/gpu_monitor.sh $RUN/gpu.txt & MON=$!
T "base evals on GPUs 4,5 (background)"
bash $SC/eval_suite.sh 4 9400 base $RUN/eval - $ACE_JOBS > $RUN/eval/logs/suite_base_ace.txt 2>&1 & B1=$!
bash $SC/eval_suite.sh 5 9401 base $RUN/eval - $GEN_JOBS > $RUN/eval/logs/suite_base_gen.txt 2>&1 & B2=$!

T "DeltaServe e2e (A/B/I/C) on GPUs $DS_GPUS"
SKIP_EVAL=1 bash $SC/run_e2e.sh $NAME
AD=$RUN/adapters/final
[ -f $AD/adapter_model.safetensors ] || { T "no final adapter: stop"; kill $MON; exit 1; }

T "adapter evals on GPUs 6,7 (background) + serving cost on GPUs $DS_GPUS"
bash $SC/eval_suite.sh 6 9402 rsi $RUN/eval $AD $ACE_JOBS > $RUN/eval/logs/suite_rsi_ace.txt 2>&1 & R1=$!
bash $SC/eval_suite.sh 7 9403 rsi $RUN/eval $AD $GEN_JOBS > $RUN/eval/logs/suite_rsi_gen.txt 2>&1 & R2=$!
bash $SC/serving_cost.sh $DS_GPUS 9404 $RUN/cost $AD
wait $B1 $B2 $R1 $R2
cat $RUN/eval/logs/suite_*.txt | tee -a $RUN/paper.log
kill $MON 2>/dev/null; pkill -f "dcgmi dmon" 2>/dev/null
python3 $SC/paper_metrics.py $RUN | tee $RUN/paper_metrics.md
T "done: $RUN"
