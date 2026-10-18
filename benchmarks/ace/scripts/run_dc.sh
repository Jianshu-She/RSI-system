#!/bin/bash
# Usage: run_dc.sh <finer|formula> <gt|nogt> <port>  -- Dynamic Cheatsheet (DC-CU) online baseline on a finance task
#   Re-running the same command resumes from steps.jsonl.  Log: > $ACE_LOGS_DIR/dc_<task>_<gt|nogt>.log
source "$(dirname "$(readlink -f "$0")")/../env.sh"
cd $ACE_UPSTREAM_DIR
export LOCAL_LLM_URL=$(ace_url $3)
exec $ACE_PY -u $ACE_ROOT/baselines/run_dc_finance.py $1 $2 $ACE_RESULTS_DIR/finance_dc/$1_$2
