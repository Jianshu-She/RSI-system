#!/bin/bash
# One-time setup of the ACE reproduction (Qwen3-32B). Idempotent: re-running skips finished steps.
#   setup/install.sh [step ...]      steps (default: all, in this order):
#     upstream      clone ace @82709de (+ submodule ace-appworld @9f3e921), apply patches/, install configs/qwen_*.jsonnet
#     ace-venv      $ACE_UPSTREAM_DIR/.venv  : `uv sync --frozen` of the upstream uv.lock, Python 3.12
#     appworld-venv $APPWORLD_DIR/.venv     : Python 3.11, deps pinned to the exact versions of the source run
#                                             (setup/requirements-appworld.lock.txt = uv --exclude-newer 2025-10-20 + click<8.2)
#     appworld-data `appworld install --repo` + `appworld download data`
#     dc            clone dynamic-cheatsheet @5cfe3c3 (unmodified; only needed for the DC-CU baseline)
#     vllm          $ACE_WORK_DIR/envs/vllm : vllm==0.31.0 (skipped if $VLLM_BIN already exists)
#     model         $MODEL_DIR : symlinked Qwen3-32B + YaRN x4 config (setup/make_yarn_model.sh)
#     train-env     (NOT in "all") $ACE_WORK_DIR/envs/train : torch 2.8.0 / transformers 4.57.3 / peft 0.18.0 / flash-attn 2.8.1
# Requirements: git, git-lfs (AppWorld ships LFS bundles), uv (https://docs.astral.sh/uv/), CUDA 12.x driver for vLLM/training.
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
source "$HERE/../env.sh"
ACE_COMMIT=82709de050e1db6e6ef2f07bcb0393560b94992a
APPWORLD_COMMIT=9f3e92155345a9159f3a8b25abc334eeca05b545
DC_COMMIT=5cfe3c37e8e52b1d858d0f3df46e7f17c50991b9
STEPS=${*:-upstream ace-venv appworld-venv appworld-data dc vllm model}
mkdir -p "$ACE_WORK_DIR" "$ACE_RESULTS_DIR"/{finance,finance_dc,appworld/playbooks,diag} "$ACE_LOGS_DIR" "$ACE_SFT_DIR/runs"

apply_patch() {  # $1 repo dir  $2 patch file
  if git -C "$1" apply --reverse --check "$2" 2>/dev/null; then echo "  already applied: $(basename "$2")"
  else git -C "$1" apply "$2" && echo "  applied: $(basename "$2")"; fi
}

for step in $STEPS; do
  echo "=== [$step]"
  case $step in
  upstream)
    git lfs install --skip-repo >/dev/null
    if [ ! -d "$ACE_UPSTREAM_DIR/.git" ]; then
      git clone https://github.com/ace-agent/ace.git "$ACE_UPSTREAM_DIR"
    fi
    git -C "$ACE_UPSTREAM_DIR" checkout -q $ACE_COMMIT
    git -C "$ACE_UPSTREAM_DIR" submodule update --init ace-appworld     # pinned by the superproject to 9f3e921
    [ "$(git -C "$APPWORLD_DIR" rev-parse HEAD)" = $APPWORLD_COMMIT ] || git -C "$APPWORLD_DIR" checkout -q $APPWORLD_COMMIT
    git -C "$APPWORLD_DIR" lfs pull
    apply_patch "$ACE_UPSTREAM_DIR" "$ACE_ROOT/patches/ace.patch"
    apply_patch "$APPWORLD_DIR" "$ACE_ROOT/patches/ace-appworld.patch"
    cp "$ACE_ROOT"/configs/qwen_*.jsonnet "$APPWORLD_DIR/experiments/configs/"
    echo "  configs installed: $(ls "$ACE_ROOT"/configs | tr '\n' ' ')"
    ;;
  ace-venv)
    (cd "$ACE_UPSTREAM_DIR" && uv sync --frozen --python 3.12)
    ;;
  appworld-venv)
    [ -x "$APPWORLD_DIR/.venv/bin/python" ] || uv venv --python 3.11 "$APPWORLD_DIR/.venv"
    uv pip install --python "$APPWORLD_DIR/.venv/bin/python" -r "$HERE/requirements-appworld.lock.txt"
    uv pip install --python "$APPWORLD_DIR/.venv/bin/python" --no-deps -e "$APPWORLD_DIR" -e "$APPWORLD_DIR/experiments"
    # Equivalent original route (resolves to the same versions):
    #   uv pip install --exclude-newer 2025-10-20 -e . -e experiments "click<8.2" joblib
    ;;
  appworld-data)
    cd "$APPWORLD_DIR"; export APPWORLD_PROJECT_PATH="$APPWORLD_DIR"
    "$APPWORLD_BIN" install --repo
    [ -d data/tasks ] || "$APPWORLD_BIN" download data
    cd - >/dev/null
    ;;
  dc)
    [ -d "$DC_DIR/.git" ] || git clone https://github.com/suzgunmirac/dynamic-cheatsheet.git "$DC_DIR"
    git -C "$DC_DIR" checkout -q $DC_COMMIT
    ;;
  vllm)
    if [ -x "$VLLM_BIN" ]; then echo "  exists: $VLLM_BIN"; else
      uv venv --python 3.12 "$ACE_WORK_DIR/envs/vllm"
      uv pip install --python "$ACE_WORK_DIR/envs/vllm/bin/python" "vllm==0.31.0"
    fi
    ;;
  model)
    if [ -f "$MODEL_DIR/config.json" ]; then echo "  exists: $MODEL_DIR"; else "$HERE/make_yarn_model.sh"; fi
    ;;
  train-env)
    E="$ACE_WORK_DIR/envs/train"
    [ -x "$E/bin/python" ] || uv venv --python 3.12 "$E"
    uv pip install --python "$E/bin/python" "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python "$E/bin/python" "transformers==4.57.3" "peft==0.18.0" accelerate
    uv pip install --python "$E/bin/python" --no-build-isolation "flash-attn==2.8.1"
    echo "  export TRAIN_PY=$E/bin/python   (e.g. in env.local.sh)"
    ;;
  *) echo "unknown step $step"; exit 1 ;;
  esac
done
echo "=== done. Next: scripts/serve_qwen.sh 0 8100  (see README.md)"
