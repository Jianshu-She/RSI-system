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
# SFT baselines for the ACE-distilled LoRA (finance), same LoRA recipe + same eval as lora_r2_fin.
#   A) base model (no playbook, temp 0) answers every TRAIN item -> raw generations (for SFT-self)
#   B) build data (datagen/build_sft_baselines.py): {formula,finer} x {gold,self} x {all,matched}; train 8 adapters (1 per GPU)
#   C) serve base + 8 adapters, evaluate each on its task's TEST set with the "(empty)" prompt
set -u
source "${ACE_ROOT:-${SLURM_SUBMIT_DIR:-.}}/env.sh" || { echo "submit from benchmarks/ace or export ACE_ROOT"; exit 1; }
RUN=$ACE_SFT_DIR/runs/sft_baselines_${SLURM_JOB_ID}
PY=$TRAIN_PY
VLLM=$VLLM_BIN
MODEL=$MODEL_DIR
EMPTY=$ACE_ROOT/eval/empty_context.txt
mkdir -p $RUN/logs
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
echo "=== $(date '+%F %T') node $(hostname) job $SLURM_JOB_ID"
free_ports() { $PY -c "
import socket
out=[]
for p in range($1,$1+200):
    s=socket.socket()
    try: s.bind(('127.0.0.1',p)); out.append(p)
    except OSError: pass
    finally: s.close()
    if len(out)==$2: break
print(' '.join(map(str,out)))"; }
serve() {  # $1 n_replicas  $2 log prefix  $3.. extra vllm args ; sets P (ports) and SPIDS
  local n=$1 tag=$2; shift 2
  read -a P <<< "$(free_ports 9700 $n)"; SPIDS=""
  for i in $(seq 0 $((n-1))); do
    CUDA_VISIBLE_DEVICES=$i $VLLM serve $MODEL --served-model-name qwen3-32b --port ${P[$i]} --host 127.0.0.1 \
      --max-model-len 131072 --gpu-memory-utilization 0.90 --enable-prefix-caching --max-num-seqs 64 \
      --override-generation-config '{"max_new_tokens": 8192}' "$@" > $RUN/logs/vllm_${tag}_$i.log 2>&1 &
    SPIDS="$SPIDS $!"
    if [ $i = 0 ]; then until curl -s 127.0.0.1:${P[0]}/v1/models | grep -q qwen3-32b; do sleep 10; done; fi
  done
  for i in $(seq 0 $((n-1))); do until curl -s 127.0.0.1:${P[$i]}/v1/models | grep -q qwen3-32b; do sleep 10; done; done
  echo "=== $(date '+%F %T') serving ($tag) on ${P[*]}"
}

# ---------- A) base generations on TRAIN ----------
serve 2 base
cd $ACE_UPSTREAM_DIR
LOCAL_LLM_URL=http://127.0.0.1:${P[0]}/v1 EVAL_LOG_DIR=$RUN/raw_base_train_formula $ACE_PY $ACE_ROOT/eval/eval_split.py \
  formula train $EMPTY $RUN/base_train_formula.json > $RUN/logs/base_train_formula.log 2>&1 & A="$!"
LOCAL_LLM_URL=http://127.0.0.1:${P[1]}/v1 EVAL_LOG_DIR=$RUN/raw_base_train_finer $ACE_PY $ACE_ROOT/eval/eval_split.py \
  finer train $EMPTY $RUN/base_train_finer.json > $RUN/logs/base_train_finer.log 2>&1 & A="$A $!"
wait $A; grep -h RESULT $RUN/logs/base_train_*.log
kill $SPIDS; sleep 30; pkill -u $(id -u) -f "[v]llm serve $MODEL" ; sleep 20

# ---------- B) data + training ----------
$ACE_PY $ACE_ROOT/datagen/build_sft_baselines.py $RUN/data $RUN/raw_base_train_formula $RUN/raw_base_train_finer > $RUN/logs/build.log 2>&1
cat $RUN/data/summary.json
NAMES="formula_gold_all formula_gold_matched formula_self_all formula_self_matched finer_gold_all finer_gold_matched finer_self_all finer_self_matched"
g=0; T=""
for n in $NAMES; do
  CUDA_VISIBLE_DEVICES=$g $PY -u $ACE_ROOT/finetune/train_lora_ace.py --data $RUN/data/$n.jsonl --out $RUN/adapter_$n \
    --epochs 2 --lr 1e-4 --rank 16 --alpha 32 --dropout 0.05 --tokens_per_step 32768 --seed 42 > $RUN/logs/train_$n.log 2>&1 &
  T="$T $!"; g=$((g+1))
done
wait $T
echo "=== $(date '+%F %T') training done"; grep -h SAVED $RUN/logs/train_*.log | wc -l

# ---------- C) evaluate adapters on TEST ----------
LORAS=""; for n in $NAMES; do LORAS="$LORAS $n=$RUN/adapter_$n"; done
serve 4 lora --enable-lora --max-lora-rank 16 --max-loras 8 --lora-modules $LORAS
for n in $NAMES; do  # smoke-test every adapter name before evaluating (lesson from the 404 model-name bug)
  code=$(curl -s -o /dev/null -w "%{http_code}" 127.0.0.1:${P[0]}/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$n\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":2}")
  echo "adapter $n -> HTTP $code"
done
E=""; k=0
for n in $NAMES; do
  task=${n%%_*}; port=${P[$((k % 4))]}
  ( LOCAL_LLM_URL=http://127.0.0.1:$port/v1 EVAL_MODEL=$n EVAL_LOG_DIR=$RUN/raw_test_$n $ACE_PY $ACE_ROOT/eval/eval_split.py \
      $task test $EMPTY $RUN/eval_$n.json > $RUN/logs/eval_$n.log 2>&1; echo "$(date '+%T') $(grep RESULT $RUN/logs/eval_$n.log) [$n]" ) &
  E="$E $!"; k=$((k+1))
done
wait $E   # not a bare wait: the vLLM servers are background jobs too
echo "=== $(date '+%F %T') DONE"
