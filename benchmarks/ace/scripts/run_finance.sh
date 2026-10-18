#!/bin/bash
# Usage: run_finance.sh <finer|formula> <offline|online> <gt|nogt> <port>
#   ACE on a finance task; the run first tests the EMPTY playbook (= base LLM), then adapts, then tests.
#   Log convention used by eval/report.py: > $ACE_LOGS_DIR/fin_<task>_<mode>_<gt|nogt>.log
source "$(dirname "$(readlink -f "$0")")/../env.sh"
cd $ACE_UPSTREAM_DIR
EXTRA=""; [ "$3" = "nogt" ] && EXTRA="--no_ground_truth"
export LOCAL_LLM_URL=$(ace_url $4)
exec $ACE_PY -u -m eval.finance.run --task_name $1 --mode $2 $EXTRA --api_provider local \
  --generator_model qwen3-32b --reflector_model qwen3-32b --curator_model qwen3-32b \
  --save_path $ACE_RESULTS_DIR/finance/$1_$2_$3
