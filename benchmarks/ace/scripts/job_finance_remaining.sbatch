#!/bin/bash
#SBATCH --job-name=ace
#SBATCH --account=CHANGE_ME          # source cluster: deploy  (or: sbatch --account=... --qos=...)
#SBATCH --qos=CHANGE_ME              # source cluster: deploy
#SBATCH --partition=main             # source cluster: main
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --output=slurm-ace-%j.out      # relative to the submit dir
# Remaining ACE finance work as one batch job (survives interactive-session loss). RE-SUBMITTABLE: every run resumes from
# the latest saved state, so after a failure just `sbatch` this file again.
#   FiNER offline ±GT : resume from the latest playbook snapshot (eval/finance_snapshots.py) to 1000 train steps, then
#                       final test with the best-val playbook (original 100-step schedule) -> results/finance/finer_offline_<lab>_finaltest
#   DC-CU FiNER ±GT   : resumes from steps.jsonl automatically
set -u
source "${ACE_ROOT:-${SLURM_SUBMIT_DIR:-.}}/env.sh" || { echo "submit from benchmarks/ace or export ACE_ROOT"; exit 1; }
PY=$TRAIN_PY
VLLM=$VLLM_BIN
MODEL=$MODEL_DIR
LOG=$ACE_LOGS_DIR/fin_remaining_${SLURM_JOB_ID}; mkdir -p $LOG
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
echo "=== $(date '+%F %T') node $(hostname) job $SLURM_JOB_ID"
PORTS=$($PY -c "
import socket
out=[]
for p in range(9500,9700):
    s=socket.socket()
    try: s.bind(('127.0.0.1',p)); out.append(p)
    except OSError: pass
    finally: s.close()
    if len(out)==4: break
print(' '.join(map(str,out)))")
read -a P <<< "$PORTS"; echo "ports: ${P[*]}"
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i $VLLM serve $MODEL --served-model-name qwen3-32b --port ${P[$i]} --host 127.0.0.1 \
    --max-model-len 131072 --gpu-memory-utilization 0.90 --enable-prefix-caching --max-num-seqs 64 \
    --override-generation-config '{"max_new_tokens": 8192}' > $LOG/vllm_$i.log 2>&1 &
  if [ $i = 0 ]; then until curl -s 127.0.0.1:${P[0]}/v1/models | grep -q qwen3-32b; do sleep 10; done; fi
done
for i in 0 1 2 3; do until curl -s 127.0.0.1:${P[$i]}/v1/models | grep -q qwen3-32b; do sleep 10; done; done
echo "=== $(date '+%F %T') serving"
cd $ACE_UPSTREAM_DIR
B=$ACE_RESULTS_DIR/finance

offline() {  # $1 label (gt|nogt)  $2 port
  local lab=$1 port=$2
  read G SNAP <<< "$(python3 $ACE_ROOT/eval/finance_snapshots.py latest finer $lab)"
  echo "$(date '+%T') finer offline $lab: latest snapshot at global step $G"
  if [ "$G" -lt 1000 ]; then
    # eval every 100 global steps on the original schedule: local eval_steps 50 when G is not a multiple of 100
    ES=$([ $((G % 100)) = 0 ] && echo 100 || echo 50)
    LOCAL_LLM_URL=http://127.0.0.1:$port/v1 $ACE_PY -u $ACE_ROOT/eval/resume_finance.py finer offline $lab $G $SNAP \
      $B/finer_offline_${lab}_resume$G $ES > $ACE_LOGS_DIR/fin_finer_offline_${lab}_resume$G.log 2>&1
    echo "$(date '+%T') finer offline $lab: training segment from $G finished (exit $?)"
  fi
  read BS BV BSNAP <<< "$(python3 $ACE_ROOT/eval/finance_snapshots.py best finer $lab)"
  echo "$(date '+%T') finer offline $lab: best val $BV at step $BS -> final test with $BSNAP"
  LOCAL_LLM_URL=http://127.0.0.1:$port/v1 $ACE_PY -u -m eval.finance.run --task_name finer --mode eval_only \
    --api_provider local --generator_model qwen3-32b --reflector_model qwen3-32b --curator_model qwen3-32b \
    --initial_playbook_path $BSNAP --save_path $B/finer_offline_${lab}_finaltest > $LOG/finaltest_$lab.log 2>&1
  echo "$(date '+%T') finer offline $lab FINAL: $(grep -E 'Test Accuracy' $LOG/finaltest_$lab.log | tail -1) (best-val step $BS, val $BV)"
}
dc() {  # $1 label  $2 port
  LOCAL_LLM_URL=http://127.0.0.1:$2/v1 $ACE_PY -u $ACE_ROOT/baselines/run_dc_finance.py finer $1 $ACE_RESULTS_DIR/finance_dc/finer_$1 \
    >> $ACE_LOGS_DIR/dc_finer_$1.log 2>&1
  echo "$(date '+%T') $(grep RESULT $ACE_LOGS_DIR/dc_finer_$1.log | tail -1)"
}
offline gt ${P[0]} & W="$!"
offline nogt ${P[1]} & W="$W $!"
dc gt ${P[2]} & W="$W $!"
dc nogt ${P[3]} & W="$W $!"
wait $W   # not a bare wait: the vLLM servers are background jobs too
echo "=== $(date '+%F %T') DONE"
