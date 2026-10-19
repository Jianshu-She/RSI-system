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
# Finance LoRA: train formula + finer adapters (GPU0/1), then serve (free ports) and evaluate LoRA on test (base not re-run).
# Usage (cwd benchmarks/ace): sbatch finetune/job_lora_fin.sbatch $ACE_SFT_DIR/data_v2 lora_r2_fin   (<data_dir> <run_name>)
set -u
source "${ACE_ROOT:-${SLURM_SUBMIT_DIR:-.}}/env.sh" || { echo "submit from benchmarks/ace or export ACE_ROOT"; exit 1; }
DATA=$1
RUN=$ACE_SFT_DIR/runs/$2_${SLURM_JOB_ID}
PY=$TRAIN_PY
VLLM=$VLLM_BIN
MODEL=$MODEL_DIR
mkdir -p $RUN/logs
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
echo "=== $(date '+%F %T') node $(hostname) job $SLURM_JOB_ID data $DATA"
for spec in "0 formula" "1 finer"; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=$1 $PY -u $ACE_ROOT/finetune/train_lora_ace.py --data $DATA/$2.jsonl --out $RUN/adapter_$2 \
      --epochs 2 --lr 1e-4 --rank 16 --alpha 32 --dropout 0.05 --tokens_per_step 32768 --seed 42 > $RUN/logs/train_$2.log 2>&1 &
done
wait
echo "=== $(date '+%F %T') training done"; grep -h SAVED $RUN/logs/train_*.log
for t in formula finer; do [ -f $RUN/adapter_$t/adapter_config.json ] || { echo "MISSING adapter_$t"; exit 1; }; done
PORTS=$($PY -c "
import socket
out=[]
for p in range(9400,9500):
    s=socket.socket()
    try: s.bind(('127.0.0.1',p)); out.append(p)
    except OSError: pass
    finally: s.close()
    if len(out)==3: break
print(' '.join(map(str,out)))")
read -a P <<< "$PORTS"; echo "ports: ${P[*]}"
LORAS="formula=$RUN/adapter_formula finer=$RUN/adapter_finer"
for i in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$i $VLLM serve $MODEL --served-model-name qwen3-32b --port ${P[$i]} --host 127.0.0.1 \
    --max-model-len 131072 --gpu-memory-utilization 0.90 --enable-prefix-caching --max-num-seqs 64 \
    --override-generation-config '{"max_new_tokens": 8192}' \
    --enable-lora --max-lora-rank 16 --max-loras 2 --lora-modules $LORAS > $RUN/logs/vllm_$i.log 2>&1 &
  if [ $i = 0 ]; then until curl -s 127.0.0.1:${P[0]}/v1/models | grep -q finer; do sleep 10; done; fi
done
for i in 0 1 2; do for k in $(seq 120); do curl -s 127.0.0.1:${P[$i]}/v1/models | grep -q finer && break; sleep 10; done; done
echo "=== $(date '+%F %T') serving"
cd $ACE_UPSTREAM_DIR
EMPTY=$ACE_ROOT/../../deltaserve/configs/empty_context.txt
BEST_FORMULA=$(ls $ACE_RESULTS_DIR/finance/formula_offline_gt/*/best_playbook.txt)
fin_eval() {  # $1 port  $2 model  $3 task  $4 context  $5 tag
  LOCAL_LLM_URL=http://127.0.0.1:$1/v1 EVAL_MODEL=$2 EVAL_LOG_DIR=$RUN/raw_$5 $ACE_PY $ACE_ROOT/eval/eval_split.py $3 test $4 $RUN/eval_$5.json > $RUN/logs/eval_$5.log 2>&1
  echo "$(date '+%T') $(grep RESULT $RUN/logs/eval_$5.log) [$5]"
}
# Base (non-LoRA) is NOT re-run: compare against the existing same-prompt base results (job 2668076, eval_*_base_empty.json)
fin_eval ${P[0]} formula formula $EMPTY formula_lora_empty & X="$!"
fin_eval ${P[1]} formula formula $BEST_FORMULA formula_lora_aceplaybook & X="$X $!"
fin_eval ${P[2]} finer finer $EMPTY finer_lora_empty & X="$X $!"
wait $X
echo "=== $(date '+%F %T') DONE"
