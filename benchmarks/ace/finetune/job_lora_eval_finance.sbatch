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
# Finance-only evaluation of finished LoRA adapters (same procedure as phase 2-3 of job_lora.sbatch).
# Usage (cwd benchmarks/ace): sbatch finetune/job_lora_eval_finance.sbatch <run_dir with adapter_formula/ adapter_finer/>
set -u
source "${ACE_ROOT:-${SLURM_SUBMIT_DIR:-.}}/env.sh" || { echo "submit from benchmarks/ace or export ACE_ROOT"; exit 1; }
RUN=$1
OUT=$RUN/evalfin_${SLURM_JOB_ID}
VLLM=$VLLM_BIN
MODEL=$MODEL_DIR
mkdir -p $OUT/logs
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
echo "=== $(date '+%F %T') node $(hostname) job $SLURM_JOB_ID run $RUN"
for t in formula finer; do [ -f $RUN/adapter_$t/adapter_config.json ] || { echo "MISSING adapter_$t"; exit 1; }; done
LORAS="formula=$RUN/adapter_formula finer=$RUN/adapter_finer"
for i in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$i $VLLM serve $MODEL --served-model-name qwen3-32b --port 920$i --host 127.0.0.1 \
    --max-model-len 131072 --gpu-memory-utilization 0.90 --enable-prefix-caching --max-num-seqs 64 \
    --override-generation-config '{"max_new_tokens": 8192}' \
    --enable-lora --max-lora-rank 16 --max-loras 2 --lora-modules $LORAS > $OUT/logs/vllm_$i.log 2>&1 &
  [ $i = 0 ] && { until curl -s 127.0.0.1:9200/v1/models | grep -q qwen3-32b; do sleep 10; done; }
done
for p in 9200 9201 9202 9203 9204; do until curl -s 127.0.0.1:$p/v1/models | grep -q finer; do sleep 10; done; done
echo "=== $(date '+%F %T') serving"
cd $ACE_UPSTREAM_DIR
EMPTY=$ACE_ROOT/eval/empty_context.txt
BEST_FORMULA=$(ls $ACE_RESULTS_DIR/finance/formula_offline_gt/*/best_playbook.txt)
fin_eval() {  # $1 port  $2 model  $3 task  $4 context  $5 tag
  LOCAL_LLM_URL=http://127.0.0.1:$1/v1 EVAL_MODEL=$2 $ACE_PY $ACE_ROOT/eval/eval_split.py $3 test $4 $OUT/eval_$5.json > $OUT/logs/eval_$5.log 2>&1
  echo "$(date '+%T') $(grep RESULT $OUT/logs/eval_$5.log) [$5]"
}
fin_eval 9200 qwen3-32b formula $EMPTY formula_base_empty & P="$!"
fin_eval 9201 formula formula $EMPTY formula_lora_empty & P="$P $!"
fin_eval 9202 formula formula $BEST_FORMULA formula_lora_aceplaybook & P="$P $!"
fin_eval 9203 qwen3-32b finer $EMPTY finer_base_empty & P="$P $!"
fin_eval 9204 finer finer $EMPTY finer_lora_empty & P="$P $!"
wait $P
echo "=== $(date '+%F %T') DONE"
