# DeltaServe co-serving RSI loop — single source of paths. `source deltaserve/env.sh` before any script.
# Reuses the ACE benchmark's env (ACE upstream clone + venv, YaRN model dir, stock vLLM 0.31, results) and adds the
# DeltaServe pieces. Override anything by exporting it first or in an untracked deltaserve/env.local.sh.
_ds_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$_ds_here/env.local.sh" ] && source "$_ds_here/env.local.sh"
source "$_ds_here/../benchmarks/ace/env.sh"          # ACE_ROOT ACE_WORK_DIR ACE_UPSTREAM_DIR ACE_PY MODEL_DIR VLLM_BIN ACE_RESULTS_DIR

export DS_HOME="$_ds_here"                                   # this directory
export DS_WORK="${DS_WORK:-$DS_HOME/work}"                   # outputs: runs/, adapters/, logs (git-ignored)
# DeltaServe-vLLM source, vendored in this folder (= github.com/852866031/DeltaServe-vLLM @ eea2a4a
# + patches/deltaserve-vllm.patch). Point DS_ROOT elsewhere only to run another checkout.
export DS_ROOT="${DS_ROOT:-$DS_HOME/DeltaServe-vLLM}"
export DS_ENV="${DS_ENV:-$HOME/miniconda3/envs/dserve-vllm}"  # env with the dserve-vllm package (editable from DS_ROOT)
export DS_CUDA_HOME="${DS_CUDA_HOME:-/usr/local/cuda-13.2}"   # nvcc >= 12.9 for FlashInfer's JIT
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$HOME/.cache/flashinfer-dserve}"  # env-private JIT cache

# Run layout (one 8-GPU node): DeltaServe server on DS_GPUS (TP = #GPUs), adapter eval with stock vLLM on EVAL_GPU
export DS_GPUS="${DS_GPUS:-0,1,2,3}"
export EVAL_GPU="${EVAL_GPU:-4}"
export DS_PORT="${DS_PORT:-8000}"
export DS_MODEL_NAME="${DS_MODEL_NAME:-llama3}"              # served name (the DeltaServe launcher's default preset tag; the model is Qwen3-32B)
export DS_INIT_ADAPTER="${DS_INIT_ADAPTER:-$DS_WORK/adapters/qwen3-32b-qkvo-r16-initial}"  # zero-B LoRA the server trains
export PLAYBOOK="${PLAYBOOK:-$(ls $ACE_RESULTS_DIR/finance/formula_offline_gt/*/best_playbook.txt 2>/dev/null | head -1)}"

ds_activate() {   # environment for any DeltaServe process
  export CUDA_HOME=$DS_CUDA_HOME PATH=$DS_CUDA_HOME/bin:$DS_ENV/bin:$PATH HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} \
         no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
}
unset _ds_here
