#!/bin/bash
# Usage: run_icl.sh <finer|formula> <port>  -- ICL baseline = eval_only with demonstrations in the context slot
#   contexts: $ACE_ICL_DIR/<task>_icl_context.txt (shipped in baselines/icl/, rebuild with baselines/build_icl_context.py)
source "$(dirname "$(readlink -f "$0")")/../env.sh"
cd $ACE_UPSTREAM_DIR
export LOCAL_LLM_URL=$(ace_url $2)
exec $ACE_PY -u -m eval.finance.run --task_name $1 --mode eval_only --api_provider local \
  --generator_model qwen3-32b --reflector_model qwen3-32b --curator_model qwen3-32b \
  --initial_playbook_path $ACE_ICL_DIR/$1_icl_context.txt --save_path $ACE_RESULTS_DIR/finance/$1_icl
