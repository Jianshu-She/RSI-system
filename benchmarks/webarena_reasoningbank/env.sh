# WebArena-Shopping (ReasoningBank) evaluation environment. `source env.sh` before any script.
# Every path is overridable: export the variable before sourcing (e.g. WA_WORK on a shared filesystem for SLURM).

export WA_HOME=${WA_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}   # this directory (scripts, tools, splits)
export WA_WORK=${WA_WORK:-$WA_HOME/work}             # large artifacts: upstream clone, venvs, site image, runs, logs
export WA_UPSTREAM=${WA_UPSTREAM:-$WA_WORK/reasoning-bank}   # google-research/reasoning-bank @ ed80611 + patches/
export WA_CODE=$WA_UPSTREAM/WebArena                 # agent / ReasoningBank code (run from here)
export WA_RUNS=${WA_RUNS:-$WA_WORK/runs}             # one dir per evaluation run
export WA_LOGS=${WA_LOGS:-$WA_WORK/logs}
export WA_SITE_IMAGE=${WA_SITE_IMAGE:-$WA_WORK/images/shopping.sqsh}   # enroot image of the shopping site
export WA_AGENT_VENV=${WA_AGENT_VENV:-$WA_WORK/envs/agent}   # browsergym / playwright / agent code
export WA_VLLM_VENV=${WA_VLLM_VENV:-$WA_WORK/envs/vllm}      # vLLM serving
export EVAL_ROOT=$WA_HOME                             # used by code/autoeval/cross_check.py to find tools/
mkdir -p "$WA_RUNS" "$WA_LOGS"

[ -f "$WA_AGENT_VENV/bin/activate" ] && source "$WA_AGENT_VENV/bin/activate"
export PYTHONPATH="$WA_UPSTREAM/third_party:$PYTHONPATH"

# Shopping site (run_eval.sh sets WA_SHOPPING per run). Other WebArena sites are not hosted: dummy ports.
export WA_SHOPPING=${WA_SHOPPING:-http://localhost:7770}
export WA_SHOPPING_ADMIN="http://localhost:7780/admin" WA_REDDIT="http://localhost:9999" WA_GITLAB="http://localhost:8023"
export WA_WIKIPEDIA="http://localhost:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="http://localhost:3000" WA_HOMEPAGE="http://localhost:4399"
for _k in SHOPPING SHOPPING_ADMIN REDDIT GITLAB WIKIPEDIA MAP HOMEPAGE; do eval "export $_k=\$WA_$_k"; done

# Model endpoints (vLLM, OpenAI-compatible)
export AGENT_LLM_URL=${AGENT_LLM_URL:-http://127.0.0.1:8000/v1}      # agent under test
export AGENT_LLM_MODEL=${AGENT_LLM_MODEL:-qwen3-32b}                # served model name (or LoRA adapter name)
export AGENT_TOKENIZER=${AGENT_TOKENIZER:-Qwen/Qwen3-32B}           # HF tokenizer of the agent (base model for LoRA)
export EMBED_URL=${EMBED_URL:-http://127.0.0.1:8001/v1} EMBED_MODEL=${EMBED_MODEL:-qwen3-embedding-8b}
export GT_JUDGE_URL=${GT_JUDGE_URL:-http://127.0.0.1:8002/v1} GT_JUDGE_MODEL=${GT_JUDGE_MODEL:-qwen3.5-397b}

# Site containers (enroot, rootless) keep their data in node-local /tmp
export ENROOT_CACHE_PATH=${ENROOT_CACHE_PATH:-/tmp/${USER}_enroot/cache}
export ENROOT_DATA_PATH=${ENROOT_DATA_PATH:-/tmp/${USER}_enroot/data}
export ENROOT_TEMP_PATH=${ENROOT_TEMP_PATH:-/tmp/${USER}_enroot/tmp}
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_TEMP_PATH"
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
