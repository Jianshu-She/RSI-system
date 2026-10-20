#!/bin/bash
#SBATCH --job-name=ds-pool
#SBATCH --account=CHANGE_ME          # source cluster: deploy  (or: sbatch --account=... --qos=...)
#SBATCH --qos=CHANGE_ME              # source cluster: deploy
#SBATCH --partition=main             # source cluster: main
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --output=slurm-ds-pool-%j.out  # relative to the submit dir
# Data selection step 1+2: scored candidate pools for Formula and FiNER (TRAIN only) -> $DS_DIR/pool_<task>.jsonl
# GPU0-3: Qwen3-32B replicas (teacher = +playbook, base probe, NLL scoring); GPU4-7: Qwen3.5-397B judge (port 8002).
# Needs the ACE offline+GT playbooks in $ACE_RESULTS_DIR/finance. Resumable (pool files are rewritten after each phase).
# Usage (cwd benchmarks/ace): sbatch finetune/job_ds_pool.sbatch      then: $ACE_PY datagen/ds_select.py <task>
set -u
source "${ACE_ROOT:-${SLURM_SUBMIT_DIR:-.}}/env.sh" || { echo "submit from benchmarks/ace or export ACE_ROOT"; exit 1; }
export DS_DIR=${DS_DIR:-$ACE_ROOT/data/selected}; LOG=$DS_DIR/logs; mkdir -p $LOG
WA=$ACE_ROOT/../webarena_reasoningbank
echo "=== $(date '+%F %T') $(hostname) job $SLURM_JOB_ID"
P=(9400 9411 9402 9403)
for i in 0 1 2 3; do
  $ACE_ROOT/scripts/serve_qwen.sh $i ${P[$i]} > $LOG/vllm_$i.log 2>&1 &
  [ $i = 0 ] && { until curl -s 127.0.0.1:${P[0]}/v1/models | grep -q qwen3-32b; do sleep 10; done; }   # warm compile cache first
done
NO_EMBED=1 JUDGE_GPUS=4,5,6,7 bash $WA/scripts/serve_aux.sh > $LOG/vllm_judge.log 2>&1 &   # :8002 (needs webarena env.sh venvs)
U=(); for i in 0 1 2 3; do for k in $(seq 90); do curl -s 127.0.0.1:${P[$i]}/v1/models | grep -q qwen3-32b && break; sleep 10; done
  curl -s 127.0.0.1:${P[$i]}/v1/models | grep -q qwen3-32b && U+=("http://127.0.0.1:${P[$i]}/v1"); done
until curl -s 127.0.0.1:8002/v1/models | grep -q qwen3.5-397b; do sleep 20; done
URLS=$(IFS=,; echo "${U[*]}"); echo "=== $(date '+%T') serving: $URLS + judge"
for t in formula finer; do
  GEN_URLS=$URLS JUDGE_URL=http://127.0.0.1:8002/v1 K=4 $ACE_PY $ACE_ROOT/datagen/ds_pool.py $t > $LOG/pool_$t.log 2>&1 &
done
wait $(jobs -p | tail -2)
for t in formula finer; do $ACE_PY $ACE_ROOT/datagen/ds_select.py $t > $LOG/select_$t.log 2>&1; done
tail -12 $LOG/pool_formula.log $LOG/pool_finer.log
echo "=== $(date '+%F %T') DONE"
