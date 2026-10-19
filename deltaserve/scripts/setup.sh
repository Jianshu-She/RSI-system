#!/bin/bash
# One-time setup for the DeltaServe RSI loop (idempotent). Needs the ACE benchmark already set up
# (benchmarks/ace/setup/install.sh: ACE clone + venv, YaRN model dir, stock vLLM 0.31) and conda + uv.
#   1. create the conda env DS_ENV with the dserve-vllm package from deltaserve/DeltaServe-vLLM
#      (editable, precompiled kernels; torch cu130)
#   2. write the zero-B q,k,v,o LoRA (r16, alpha 32) the server trains: DS_INIT_ADAPTER
# Usage: bash deltaserve/scripts/setup.sh [env|adapter]   (no argument = both)
set -eu
source "$(dirname "$(readlink -f "$0")")/../env.sh"
STEPS=${1:-"env adapter"}
mkdir -p $DS_WORK
if [[ " $STEPS " == *" env "* ]]; then
  [ -x $DS_ENV/bin/python ] || conda create -y -p $DS_ENV python=3.12
  cd $DS_ROOT/dserve-vllm
  VLLM_VERSION_OVERRIDE=0.21.1rc1.dev123+g117afeea4.precompiled \
  VLLM_PRECOMPILED_WHEEL_COMMIT=117afeea4665367a3066c1df58d4082d07fcc946 VLLM_USE_PRECOMPILED=1 \
    uv pip install --python $DS_ENV/bin/python --editable . --torch-backend=auto
  uv pip install --python $DS_ENV/bin/python pyarrow
fi
if [[ " $STEPS " == *" adapter "* ]]; then
  [ -f $DS_INIT_ADAPTER/adapter_model.safetensors ] || \
    $DS_ENV/bin/python $DS_HOME/scripts/init_lora_adapter.py --model $MODEL_DIR --out $DS_INIT_ADAPTER \
      --rank 16 --alpha 32 --targets q,k,v,o
fi
echo "setup done: DS_ROOT=$DS_ROOT DS_ENV=$DS_ENV DS_INIT_ADAPTER=$DS_INIT_ADAPTER"
