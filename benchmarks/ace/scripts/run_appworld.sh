#!/bin/bash
# Usage: run_appworld.sh <port[,port,...]> <step> [<step> ...]
#   step = adapt:<config> | eval:<config>:<dataset> | online:<config> | score:<config>:<dataset>
#   configs = configs/qwen_*.jsonnet (installed into $APPWORLD_DIR/experiments/configs by setup/install.sh)
#   A comma-separated port list spreads the eval processes over several replicas (patched lite_llm_generator).
source "$(dirname "$(readlink -f "$0")")/../env.sh"
cd $APPWORLD_DIR
URLS=$(for p in ${1//,/ }; do printf '%s,' "$(ace_url $p)"; done); URLS=${URLS%,}
export APPWORLD_PROJECT_PATH=$(pwd) LOCAL_LLM_URL=$URLS
mkdir -p $ACE_RESULTS_DIR/appworld/playbooks   # trained playbooks are written here (configs read ACE_RESULTS_DIR)
shift
for step in "$@"; do
  IFS=: read kind cfg ds <<< "$step"
  echo "=== $(date '+%F %T') START $step"
  case $kind in
    adapt) $APPWORLD_BIN run $cfg ;;
    eval)  $APPWORLD_BIN run $cfg --num-processes 8 --override "{\"config\":{\"dataset\":\"$ds\"}}" ;;
    online) $APPWORLD_BIN run $cfg ;;
    score) $APPWORLD_BIN evaluate $cfg $ds ;;
  esac
  echo "=== $(date '+%F %T') END $step exit=$?"
done
