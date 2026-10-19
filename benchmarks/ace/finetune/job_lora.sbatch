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
# LoRA SFT of Qwen3-32B on ACE train-split traces, then evaluation of the adapters (one node, 8 GPUs).
#   Phase 1  train 3 adapters in parallel (GPU0 formula, GPU1 finer, GPU2 appworld)
#   Phase 2  serve 8 vLLM replicas of the same base model (YaRN config) with all adapters (ports 9100-9107)
#   Phase 3  finance evals (empty context "(empty)" = SFT input format) + AppWorld ReAct eval with the appworld adapter
#   Phase 4  score AppWorld, write summary.json
set -u
source "${ACE_ROOT:-${SLURM_SUBMIT_DIR:-.}}/env.sh" || { echo "submit from benchmarks/ace or export ACE_ROOT"; exit 1; }
RUN=$ACE_SFT_DIR/runs/lora_r1_${SLURM_JOB_ID}
DATA=$ACE_SFT_DIR/data_v1
PY=$TRAIN_PY
VLLM=$VLLM_BIN
MODEL=$MODEL_DIR
mkdir -p $RUN/logs
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
echo "=== $(date '+%F %T') node $(hostname) job $SLURM_JOB_ID"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# ---------------- Phase 1: train ----------------
for spec in "0 formula 2 1e-4" "1 finer 2 1e-4" "2 appworld 2 1e-4"; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=$1 $PY -u $ACE_ROOT/finetune/train_lora_ace.py --data $DATA/$2.jsonl --out $RUN/adapter_$2 \
      --epochs $3 --lr $4 --rank 16 --alpha 32 --dropout 0.05 --tokens_per_step 32768 --seed 42 > $RUN/logs/train_$2.log 2>&1 &
done
wait
echo "=== $(date '+%F %T') training done"; grep -h SAVED $RUN/logs/train_*.log
for t in formula finer appworld; do [ -f $RUN/adapter_$t/adapter_config.json ] || { echo "MISSING adapter_$t"; exit 1; }; done

# ---------------- Phase 2: serve base + adapters ----------------
LORAS="formula=$RUN/adapter_formula finer=$RUN/adapter_finer appworld=$RUN/adapter_appworld"
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i $VLLM serve $MODEL --served-model-name qwen3-32b --port 910$i --host 127.0.0.1 \
    --max-model-len 131072 --gpu-memory-utilization 0.90 --enable-prefix-caching --max-num-seqs 64 \
    --override-generation-config '{"max_new_tokens": 8192}' \
    --enable-lora --max-lora-rank 16 --max-loras 3 --lora-modules $LORAS > $RUN/logs/vllm_$i.log 2>&1 &
  [ $i = 0 ] && { until curl -s 127.0.0.1:9100/v1/models | grep -q qwen3-32b; do sleep 10; done; }  # warm compile cache first
done
for p in 9100 9101 9102 9103 9104 9105 9106 9107; do until curl -s 127.0.0.1:$p/v1/models | grep -q appworld; do sleep 10; done; done
echo "=== $(date '+%F %T') serving"; curl -s 127.0.0.1:9100/v1/models | $PY -c "import json,sys;print([m['id'] for m in json.load(sys.stdin)['data']])"

# ---------------- Phase 3: evaluate ----------------
cd $ACE_UPSTREAM_DIR
EMPTY=$ACE_ROOT/../../deltaserve/configs/empty_context.txt
BEST_FORMULA=$(ls $ACE_RESULTS_DIR/finance/formula_offline_gt/*/best_playbook.txt)
fin_eval() {  # $1 port  $2 model  $3 task  $4 context  $5 tag
  LOCAL_LLM_URL=http://127.0.0.1:$1/v1 EVAL_MODEL=$2 $ACE_PY $ACE_ROOT/eval/eval_split.py $3 test $4 $RUN/eval_$5.json > $RUN/logs/eval_$5.log 2>&1
  grep RESULT $RUN/logs/eval_$5.log
}
( fin_eval 9100 qwen3-32b formula $EMPTY formula_base_empty; fin_eval 9100 formula formula $EMPTY formula_lora_empty
  fin_eval 9100 formula formula $BEST_FORMULA formula_lora_aceplaybook ) & EVAL_PIDS="$!"
( fin_eval 9101 qwen3-32b finer $EMPTY finer_base_empty; fin_eval 9101 finer finer $EMPTY finer_lora_empty ) & EVAL_PIDS="$EVAL_PIDS $!"
cd $APPWORLD_DIR; export APPWORLD_PROJECT_PATH=$(pwd)
for split in test_normal test_challenge; do
  URLS=$( [ $split = test_normal ] && echo http://127.0.0.1:9102/v1,http://127.0.0.1:9103/v1,http://127.0.0.1:9104/v1 \
                                   || echo http://127.0.0.1:9105/v1,http://127.0.0.1:9106/v1,http://127.0.0.1:9107/v1 )
  LOCAL_LLM_URL=$URLS $APPWORLD_BIN run qwen_lora_react_eval --num-processes 8 \
      --override "{\"config\":{\"dataset\":\"$split\"}}" > $RUN/logs/aw_$split.log 2>&1 &
  EVAL_PIDS="$EVAL_PIDS $!"
done
wait $EVAL_PIDS   # (not a bare wait: the vLLM servers are background jobs too)
echo "=== $(date '+%F %T') evals done"

# ---------------- Phase 4: integrity + score ----------------
for split in test_normal test_challenge; do
  MISSING=$($PY -c "
import os,json; ids=open('data/datasets/$split.txt').read().split()
print(json.dumps([t for t in ids if not os.path.exists(f'experiments/outputs/qwen_lora_react_eval/tasks/{t}/dbs')]))")
  echo "$split missing: $MISSING"
  if [ "$MISSING" != "[]" ]; then
    LOCAL_LLM_URL=http://127.0.0.1:9102/v1,http://127.0.0.1:9103/v1,http://127.0.0.1:9104/v1,http://127.0.0.1:9105/v1 \
      $APPWORLD_BIN run qwen_lora_react_eval --num-processes 8 --override "{\"config\":{\"dataset\":\"$split\",\"task_ids\":$MISSING}}" >> $RUN/logs/aw_$split.log 2>&1
  fi
  $APPWORLD_BIN evaluate qwen_lora_react_eval $split > $RUN/logs/score_$split.log 2>&1
done
$PY - <<EOF
import json,glob,os
run="$RUN"; out={}
for f in sorted(glob.glob(run+"/eval_*.json")):
    d=json.load(open(f)); out[os.path.basename(f)[5:-5]]=round(100*d["results"]["accuracy"],2)
for s in ["test_normal","test_challenge"]:
    p="$APPWORLD_DIR/experiments/outputs/qwen_lora_react_eval/evaluations/"+s+".json"
    if os.path.exists(p): out["appworld_lora_"+s]=json.load(open(p))["aggregate"]
json.dump(out,open(run+"/summary.json","w"),indent=1); print(json.dumps(out,indent=1))
EOF
echo "=== $(date '+%F %T') DONE"
