#!/bin/bash
#SBATCH --job-name=wa-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=%x_%j.out
# Evaluate one agent on the TEST split (86 unseen-template tasks), sharded over 3 fresh sites. Resumable.
# Usage: sbatch ... jobs/eval.sbatch <run_name> [method=no_memory]
#   base model:     sbatch jobs/eval.sbatch base_q32
#   LoRA adapter:   ADAPTER=$WA_WORK/ckpt/lora_v3/final sbatch jobs/eval.sbatch lora_v3
#   ReasoningBank:  sbatch jobs/eval.sbatch rb_q32 reasoningbank          (memory is built online over the test stream)
# Layout: GPU0-1 agent (Qwen3-32B [+LoRA]) :8000 | GPU2 embedding :8001 | GPU4-7 Qwen3.5-397B grader :8002
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"; source env.sh > /dev/null
NAME=$1; METHOD=${2:-no_memory}; SPLIT=${SPLIT:-test}; MAXTOK=${MAX_NEW_TOKENS:-8192}
bash scripts/serve_aux.sh &  AUX=$!
bash scripts/serve_agent.sh Qwen/Qwen3-32B "$NAME" 0,1 8000 $ADAPTER       # adapter (if any) is served under $NAME
MODEL=$NAME; [ -z "$ADAPTER" ] && MODEL=$NAME
wait $AUX
if [ "$METHOD" = no_memory ]; then SHARDS="0 1 2"; else SHARDS="all"; fi    # ReasoningBank is sequential (streaming memory)
for k in $SHARDS; do
  off=$([ "$k" = all ] && echo 1 || echo $((1+k))); split=$([ "$k" = all ] && echo $SPLIT || echo ${SPLIT}_part$k)
  bash scripts/start_site.sh "site_${NAME}_$k" $off
  R=""; [ -d "$WA_RUNS/$NAME" ] && R=--resume
  bash scripts/run_eval.sh $R --name "$NAME" --method "$METHOD" --model "$MODEL" --tokenizer Qwen/Qwen3-32B \
    --site $((7770+off)) --split "$split" --agent_url http://127.0.0.1:8000/v1 --max_new_tokens "$MAXTOK"
  sleep 5
done
sleep 60; while pgrep -u $(id -u) -f "[p]ipeline_memory.py .*runs/$NAME( |$)" > /dev/null; do sleep 120; done
python tools/report.py --split $SPLIT "$WA_RUNS/$NAME"
