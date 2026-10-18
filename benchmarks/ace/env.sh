# ACE benchmark — single source of paths/ports. Source it: `source benchmarks/ace/env.sh`.
# Every variable can be overridden by exporting it before sourcing, or in an (untracked) env.local.sh next to this file.
# Python scripts read the same variables (with the same defaults) from the environment.

_ace_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$_ace_here/env.local.sh" ] && source "$_ace_here/env.local.sh"

export ACE_ROOT="${ACE_ROOT:-$_ace_here}"                          # this directory (scripts, configs, patches)
export ACE_WORK_DIR="${ACE_WORK_DIR:-$ACE_ROOT/work}"              # everything generated: clones, venvs, model dir, results
export ACE_UPSTREAM_DIR="${ACE_UPSTREAM_DIR:-$ACE_WORK_DIR/ace}"    # clone of github.com/ace-agent/ace @ 82709de (+patch)
export APPWORLD_DIR="${APPWORLD_DIR:-$ACE_UPSTREAM_DIR/ace-appworld}"  # submodule ace-appworld @ 9f3e921 (+patch)
export DC_DIR="${DC_DIR:-$ACE_WORK_DIR/dynamic-cheatsheet}"         # clone of github.com/suzgunmirac/dynamic-cheatsheet @ 5cfe3c3
export ACE_RESULTS_DIR="${ACE_RESULTS_DIR:-$ACE_WORK_DIR/results}"  # finance/, finance_dc/, appworld/playbooks/, diag/
export ACE_LOGS_DIR="${ACE_LOGS_DIR:-$ACE_WORK_DIR/logs}"
export ACE_SFT_DIR="${ACE_SFT_DIR:-$ACE_WORK_DIR/sft}"              # data_v1/, data_v2/, runs/<job>/ (adapters, evals)
export ACE_ICL_DIR="${ACE_ICL_DIR:-$ACE_ROOT/baselines/icl}"        # ICL contexts (<task>_icl_context.txt)

# Model: local dir with symlinked Qwen3-32B weights + config.json patched for YaRN x4 (131072 ctx); see setup/make_yarn_model.sh
export QWEN3_32B_SRC="${QWEN3_32B_SRC:-}"                            # HF snapshot dir of Qwen/Qwen3-32B (empty -> download)
export MODEL_DIR="${MODEL_DIR:-$ACE_WORK_DIR/models/Qwen3-32B-yarn131k}"

# Interpreters / binaries
export ACE_PY="${ACE_PY:-$ACE_UPSTREAM_DIR/.venv/bin/python}"       # ACE finance, DC, eval_split (uv sync of ace/uv.lock, py3.12)
export APPWORLD_BIN="${APPWORLD_BIN:-$APPWORLD_DIR/.venv/bin/appworld}"  # AppWorld venv (py3.11, deps pinned to Oct 2025)
export VLLM_BIN="${VLLM_BIN:-$ACE_WORK_DIR/envs/vllm/bin/vllm}"     # vLLM 0.31.0 venv (also has transformers -> tokenizer tools)
export VLLM_PY="${VLLM_PY:-$(dirname "$VLLM_BIN")/python}"
export TRAIN_PY="${TRAIN_PY:-python}"                               # LoRA env: torch 2.8, transformers 4.57.3, peft 0.18, flash-attn 2.8

# Serving
export LOCAL_LLM_HOST="${LOCAL_LLM_HOST:-127.0.0.1}"
# served model name is fixed to "qwen3-32b" (hard-coded in scripts and configs/qwen_*.jsonnet)
# one replica per GPU; source runs used GPU0 8100, GPU1 8110 (8101 was taken), GPU2-7 8102-8107
export ACE_PORTS="${ACE_PORTS:-8100 8110 8102 8103 8104 8105 8106 8107}"
export no_proxy="localhost,127.0.0.1${no_proxy:+,$no_proxy}" NO_PROXY="localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}"

ace_url() { echo "http://$LOCAL_LLM_HOST:$1/v1"; }   # ace_url <port>
unset _ace_here
