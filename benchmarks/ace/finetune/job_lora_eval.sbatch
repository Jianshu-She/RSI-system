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
# Evaluate trained LoRA adapters (vLLM --enable-lora on the base Qwen3-32B, same serving flags as all other runs).
#   1) Formula LoRA on test with "(empty)" context, saving every raw generation (diagnostic)
#   2) AppWorld ReAct with the appworld adapter on test_normal + test_challenge; rerun missing ids; score
# Ports are picked free at runtime (a fixed port collided with another user's process on a shared node).
# Usage (cwd benchmarks/ace): [SKIP_FORMULA=1] sbatch finetune/job_lora_eval.sbatch <run_dir with adapter_formula/ adapter_finer/ adapter_appworld/>
set -u
source "${ACE_ROOT:-${SLURM_SUBMIT_DIR:-.}}/env.sh" || { echo "submit from benchmarks/ace or export ACE_ROOT"; exit 1; }
RUN=$1
OUT=$RUN/eval_${SLURM_JOB_ID}
PY=$TRAIN_PY
VLLM=$VLLM_BIN
MODEL=$MODEL_DIR
mkdir -p $OUT/logs
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
echo "=== $(date '+%F %T') node $(hostname) job $SLURM_JOB_ID run $RUN"
PORTS=$($PY -c "
import socket
out=[]
for p in range(9300,9400):
    s=socket.socket()
    try: s.bind(('127.0.0.1',p)); out.append(p)
    except OSError: pass
    finally: s.close()
    if len(out)==8: break
print(' '.join(map(str,out)))")
read -a P <<< "$PORTS"; echo "ports: ${P[*]}"
LORAS="formula=$RUN/adapter_formula finer=$RUN/adapter_finer appworld=$RUN/adapter_appworld"
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i $VLLM serve $MODEL --served-model-name qwen3-32b --port ${P[$i]} --host 127.0.0.1 \
    --max-model-len 131072 --gpu-memory-utilization 0.90 --enable-prefix-caching --max-num-seqs 64 \
    --override-generation-config '{"max_new_tokens": 8192}' \
    --enable-lora --max-lora-rank 16 --max-loras 3 --lora-modules $LORAS > $OUT/logs/vllm_$i.log 2>&1 &
  if [ $i = 0 ]; then until curl -s 127.0.0.1:${P[0]}/v1/models | grep -q appworld; do sleep 10; done; fi
done
for i in 0 1 2 3 4 5 6 7; do
  for k in $(seq 120); do curl -s 127.0.0.1:${P[$i]}/v1/models | grep -q appworld && break; sleep 10; done
  curl -s 127.0.0.1:${P[$i]}/v1/models | grep -q appworld || { echo "replica $i failed:"; tail -5 $OUT/logs/vllm_$i.log; }
done
echo "=== $(date '+%F %T') serving"
U=(); for i in 0 1 2 3 4 5 6 7; do curl -s 127.0.0.1:${P[$i]}/v1/models | grep -q appworld && U+=("http://127.0.0.1:${P[$i]}/v1"); done
echo "live replicas: ${#U[@]}"

# 1) Formula LoRA diagnostic (raw generations saved)
cd $ACE_UPSTREAM_DIR
PIDS=""
[ "${SKIP_FORMULA:-0}" = 1 ] || ( LOCAL_LLM_URL=${U[0]} EVAL_MODEL=formula EVAL_LOG_DIR=$OUT/raw_formula_lora_empty $ACE_PY $ACE_ROOT/eval/eval_split.py \
    formula test $ACE_ROOT/../../deltaserve/configs/empty_context.txt $OUT/eval_formula_lora_empty.json > $OUT/logs/eval_formula_lora_empty.log 2>&1
  echo "$(date '+%T') $(grep RESULT $OUT/logs/eval_formula_lora_empty.log)" ) & [ "${SKIP_FORMULA:-0}" = 1 ] || PIDS="$!"

# 2) AppWorld LoRA (ReAct agent, shipped initial playbook, generator = appworld adapter)
cd $APPWORLD_DIR; export APPWORLD_PROJECT_PATH=$(pwd)
N=${#U[@]}; H=$(( (N - 1) / 2 + 1 ))
URLS_N=$(IFS=,; echo "${U[*]:1:$((H-1))}"); URLS_C=$(IFS=,; echo "${U[*]:$H}")
[ -z "$URLS_N" ] && URLS_N=${U[0]}; [ -z "$URLS_C" ] && URLS_C=${U[0]}
echo "test_normal -> $URLS_N"; echo "test_challenge -> $URLS_C"
for split in test_normal test_challenge; do
  URLS=$([ $split = test_normal ] && echo $URLS_N || echo $URLS_C)
  LOCAL_LLM_URL=$URLS $APPWORLD_BIN run qwen_lora_react_eval --num-processes 8 \
      --override "{\"config\":{\"dataset\":\"$split\"}}" > $OUT/logs/aw_$split.log 2>&1 &
  PIDS="$PIDS $!"
done
wait $PIDS   # not a bare wait: the vLLM servers are background jobs too
echo "=== $(date '+%F %T') evals done"
ALL=$(IFS=,; echo "${U[*]}")
for split in test_normal test_challenge; do
  MISSING=$($PY -c "
import os,json; ids=open('data/datasets/$split.txt').read().split()
print(json.dumps([t for t in ids if not os.path.exists(f'experiments/outputs/qwen_lora_react_eval/tasks/{t}/dbs')]))")
  echo "$split missing: $MISSING"
  if [ "$MISSING" != "[]" ]; then
    LOCAL_LLM_URL=$ALL $APPWORLD_BIN run qwen_lora_react_eval --num-processes 8 \
      --override "{\"config\":{\"dataset\":\"$split\",\"task_ids\":$MISSING}}" >> $OUT/logs/aw_$split.log 2>&1
  fi
  $APPWORLD_BIN evaluate qwen_lora_react_eval $split > $OUT/logs/score_$split.log 2>&1
  $PY -c "import json;print('$split',json.load(open('experiments/outputs/qwen_lora_react_eval/evaluations/$split.json'))['aggregate'])"
done
echo "=== $(date '+%F %T') DONE"
