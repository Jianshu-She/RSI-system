#!/bin/bash
# One-time setup: upstream code + patches, python envs, browser. Idempotent; re-run is safe.
#   bash setup/install.sh            (then: bash setup/build_site_image.sh)
set -e
source "$(dirname "$0")/../env.sh" > /dev/null
UPSTREAM_REPO=https://github.com/google-research/reasoning-bank.git
UPSTREAM_COMMIT=ed80611                       # "Update BibTeX"; patches/ are made against this commit

# 1) upstream ReasoningBank code + our patches (local vLLM endpoints, judge/parser fixes, task lists, resumable runs)
if [ ! -d "$WA_UPSTREAM/.git" ]; then
  git clone "$UPSTREAM_REPO" "$WA_UPSTREAM"
  git -C "$WA_UPSTREAM" checkout -q "$UPSTREAM_COMMIT"
  git -C "$WA_UPSTREAM" apply "$WA_HOME/patches/reasoning-bank.patch"
  echo "upstream ready at $WA_UPSTREAM (patched)"
else
  echo "upstream exists at $WA_UPSTREAM (not re-patching)"
fi

# 2) python envs (uv). agent: browsergym + playwright; vllm: model serving; train: LoRA (see rsi/finetune)
command -v uv > /dev/null || { echo "install uv first: https://docs.astral.sh/uv/"; exit 1; }
if [ ! -x "$WA_AGENT_VENV/bin/python" ]; then
  uv venv "$WA_AGENT_VENV" --python 3.12
  VIRTUAL_ENV=$WA_AGENT_VENV uv pip install -r "$WA_HOME/setup/requirements-agent.txt"
  "$WA_AGENT_VENV/bin/playwright" install chromium
fi
if [ ! -x "$WA_VLLM_VENV/bin/vllm" ]; then
  uv venv "$WA_VLLM_VENV" --python 3.12
  VIRTUAL_ENV=$WA_VLLM_VENV uv pip install -r "$WA_HOME/setup/requirements-vllm.txt"
fi
echo "envs: agent=$WA_AGENT_VENV vllm=$WA_VLLM_VENV"
