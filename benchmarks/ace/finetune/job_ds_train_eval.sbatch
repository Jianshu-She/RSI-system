#!/bin/bash
#SBATCH --job-name=ds-train-eval
#SBATCH --account=CHANGE_ME          # source cluster: deploy  (or: sbatch --account=... --qos=...)
#SBATCH --qos=CHANGE_ME              # source cluster: deploy
#SBATCH --partition=main             # source cluster: main
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --output=slurm-ds-train-eval-%j.out  # relative to the submit dir
# Data selection step 4: one LoRA per selected set (1 GPU each, rsi/finetune/train_lora.py), then test evals with an
# EMPTY context (temp 0), plus same-node base control. ~1h50 on 8xH100 (finer_rand_hardup is the long pole, ~90 min).
# Usage (cwd benchmarks/ace): sbatch finetune/job_ds_train_eval.sbatch [set ...]   (default: all 10 sets of ds_r1)
set -u
source "${ACE_ROOT:-${SLURM_SUBMIT_DIR:-.}}/env.sh" || { echo "submit from benchmarks/ace or export ACE_ROOT"; exit 1; }
DS_DIR=${DS_DIR:-$ACE_ROOT/data/selected}; RUN=$ACE_SFT_DIR/runs/ds_${SLURM_JOB_ID}; mkdir -p $RUN/logs
TRAINER=$ACE_ROOT/../../rsi/finetune/train_lora.py
SETS=(finer_rand_hardup finer_rand finer_quality finer_hard finer_hard_ctrl finer_clean formula_rand_hardup formula_rand formula_hard formula_hard_ctrl)
[ $# -gt 0 ] && SETS=("$@")
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
echo "=== $(date '+%F %T') $(hostname) job $SLURM_JOB_ID sets: ${SETS[*]}"
train() {  # $1 gpu $2 set
  CUDA_VISIBLE_DEVICES=$1 $TRAIN_PY $TRAINER --base $MODEL_DIR --data $DS_DIR/$2.jsonl --out $RUN/lora_$2 --thinking off --loss_on last \
    --epochs 2 --lr 1e-4 --rank 16 --alpha 32 --batch 16 --max_len 8192 --saves 0 > $RUN/logs/train_$2.log 2>&1
  echo "$(date '+%T') trained $2: $(tail -n 2 $RUN/logs/train_$2.log | head -1 | cut -c1-120)"
}
for ((k=0; k<${#SETS[@]}; k+=8)); do
  for ((i=0; i<8 && k+i<${#SETS[@]}; i++)); do train $i ${SETS[$((k+i))]} & done; wait
done
LORAS=""; for s in "${SETS[@]}"; do
  [ -f $RUN/lora_$s/final/adapter_config.json ] && LORAS="$LORAS $s=$RUN/lora_$s/final" || echo "MISSING adapter $s"; done
echo "=== $(date '+%F %T') training done"

P=(9500 9511 9502 9503 9504 9505 9506 9507); LAST=${SETS[-1]}
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i $VLLM_BIN serve $MODEL_DIR --served-model-name qwen3-32b --port ${P[$i]} --host 127.0.0.1 --max-model-len 32768 \
    --gpu-memory-utilization 0.90 --enable-prefix-caching --max-num-seqs 64 --override-generation-config '{"max_new_tokens": 8192}' \
    --enable-lora --max-lora-rank 16 --max-loras ${#SETS[@]} --lora-modules $LORAS > $RUN/logs/vllm_$i.log 2>&1 &
  [ $i = 0 ] && { until curl -s 127.0.0.1:${P[0]}/v1/models | grep -q $LAST; do sleep 10; done; }
done
U=(); for i in 0 1 2 3 4 5 6 7; do for k in $(seq 90); do curl -s 127.0.0.1:${P[$i]}/v1/models | grep -q $LAST && break; sleep 10; done
  curl -s 127.0.0.1:${P[$i]}/v1/models | grep -q $LAST && U+=("http://127.0.0.1:${P[$i]}/v1"); done
echo "=== $(date '+%T') serving on ${#U[@]} replicas"
cd $ACE_UPSTREAM_DIR; EMPTY=$ACE_ROOT/../../deltaserve/configs/empty_context.txt  # literal "(empty)"
EVALS=(); for t in $(printf '%s\n' "${SETS[@]%%_*}" | sort -u); do EVALS+=("qwen3-32b $t"); done
for s in "${SETS[@]}"; do EVALS+=("$s ${s%%_*}"); done
PIDS=""; j=0
for e in "${EVALS[@]}"; do
  read -r model task <<< "$e"; url=${U[$((j % ${#U[@]}))]}; j=$((j+1)); tag=${model}__$task
  ( LOCAL_LLM_URL=$url EVAL_MODEL=$model $ACE_PY $ACE_ROOT/eval/eval_split.py $task test $EMPTY $RUN/eval_$tag.json \
      > $RUN/logs/eval_$tag.log 2>&1; echo "$(date '+%T') $(grep RESULT $RUN/logs/eval_$tag.log | sed 's/ctx=[^:]*//') [$tag]" ) &
  PIDS="$PIDS $!"
done
wait $PIDS
echo "=== $(date '+%F %T') DONE"
