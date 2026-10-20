#!/bin/bash
# Follow-up to paper_run.sh's serving cost (same run dir, adapter from it), fixing two gaps:
#   cost_fresh/  every condition at 4 rps on the formula TRAIN split (500 items never sent to this server), so no
#                request is a full-prompt prefix-cache hit (the 4-rps pass of serving_cost.sh repeated the test items)
#   merged/      the adapter merged into the weights (merge_lora.py): the "customized model", served with no LoRA
#                kernels; measured in cost_fresh/ next to the unmerged LoRA
#   cost_nopc/   prefix caching OFF at 1 rps: skills that differ per user cannot be shared, every request prefills
#                and holds its own 30K-token playbook in KV
# Usage: bash deltaserve/scripts/paper_run_cost2.sh <run_name>
set -u
source "$(dirname "$(readlink -f "$0")")/../env.sh"
RUN=$DS_WORK/runs/$1; AD=$RUN/adapters/final; SC=$DS_HOME/scripts
bash $SC/gpu_monitor.sh $RUN/gpu_cost2.txt & MON=$!
grep -q "^merged" $RUN/merge.log 2>/dev/null || $DS_ENV/bin/python $SC/merge_lora.py $MODEL_DIR $AD $RUN/merged > $RUN/merge.log 2>&1
tail -1 $RUN/merge.log
MERGED_DIR=$RUN/merged RATES=4 SPLIT=train LEAK=0 SERVERS="P L M" bash $SC/serving_cost.sh $DS_GPUS 9404 $RUN/cost_fresh $AD
MERGED_DIR=$RUN/merged RATES=1 SERVERS=M LEAK=0 bash $SC/serving_cost.sh $DS_GPUS 9404 $RUN/cost $AD
RATES=1 LEAK=0 SERVERS=P NOPC=1 bash $SC/serving_cost.sh $DS_GPUS 9404 $RUN/cost_nopc $AD
kill $MON; pkill -f "dcgmi dmon"
python3 $SC/paper_metrics.py $RUN > $RUN/paper_metrics.md
echo "=== done cost2"
