# WebArena-Shopping: ReasoningBank & LoRA

Evaluate a web agent on the WebArena **shopping** site (187 tasks), with or without
[ReasoningBank](https://arxiv.org/abs/2509.25140) memory, generate SFT data from agent trajectories, fine-tune a LoRA,
and evaluate it on held-out task templates. Everything runs on one 8-GPU node, rootless (enroot, no docker/sudo).

Results so far: [`results/RESULTS.md`](results/RESULTS.md).

```
setup/      install.sh (upstream @ed80611 + patches, venvs), build_site_image.sh, requirements-*.txt
patches/    reasoning-bank.patch — every change vs upstream (local vLLM endpoints, judge/parser fixes, --task_ids_file, resume)
env.sh      ALL paths / ports / endpoints (override any variable before sourcing)
scripts/    start_site.sh stop_site.sh | serve_agent.sh serve_aux.sh | run_eval.sh stop_run.sh
tools/      report.py (scores + paired sign test), audit_answers.py, export_traces.py, split_traces.py, make_split.py, verify_cross.py
datagen/    build_sft_data.py — trajectories -> chat-JSONL SFT set (train split only, success-filtered, style-filtered)
jobs/       collect_teacher.sbatch, finetune.sbatch, eval.sbatch (one node each; resumable)
splits/     template-level train/test split (seed 0): train 101 tasks / test 86 tasks, no shared template
results/    RESULTS.md, final_test.json, EXPERIMENT_LOG.md (every run: spec, prediction, actual, incidents)
```
The LoRA trainer is shared across benchmarks: [`rsi/finetune/train_lora.py`](../../rsi/finetune/train_lora.py).

## 0. Setup (once)
```bash
cd benchmarks/webarena_reasoningbank
export WA_WORK=/shared/fs/wa_work          # big artifacts (~15 GB image + venvs + runs); default ./work
bash setup/install.sh                      # clones upstream, applies patches, creates venvs, installs chromium
bash setup/build_site_image.sh             # enroot import of the shopping site (WebArena-Verified image)
source env.sh
```
Models (vLLM 0.31): agent `Qwen/Qwen3-32B` (served with YaRN ×4 → 131k context), `Qwen/Qwen3-Embedding-8B` (memory
retrieval), `Qwen/Qwen3.5-397B-A17B-FP8` (grader for fuzzy-match answers; also teacher / cross-check judge).
Training env: see [`rsi/finetune/README.md`](../../rsi/finetune/README.md).

## 1. Evaluation
```bash
bash scripts/serve_aux.sh                                     # GPU2 embedding :8001, GPU4-7 grader :8002 (~15 min)
bash scripts/serve_agent.sh Qwen/Qwen3-32B qwen3-32b 0,1 8000 # agent; add an adapter dir as 5th arg to serve a LoRA
bash scripts/start_site.sh site_a 1                           # fresh site on :7771; one site per concurrent run
bash scripts/run_eval.sh --name base_q32 --method no_memory --model qwen3-32b --tokenizer Qwen/Qwen3-32B --site 7771 --split test
python tools/report.py --split test --baseline $WA_RUNS/base_q32 $WA_RUNS/<other runs>
```
- `--method`: `no_memory` (plain agent, use for base/LoRA), `reasoningbank` (paper method: the agent judges its own
  trajectory and writes memory; streaming, sequential), `reasoningbank_cross` (judge = agreement with an independent
  stronger-agent run given by `--cross_ref`; 86% judge accuracy vs 62–71% self-judge).
- `--split`: `test` (86) | `train` (101) | `all` (187, paper protocol) | `test_part{0,1,2}` (shards, same run name, `--resume`).
- Resumable: rerun with `--resume`; `scripts/stop_run.sh <name>` stops a run. Each run dir keeps every step
  (prompt, model output, page state, screenshot), `summary_info.json` (official reward) and `meta.json` (exact settings).
- Rules: a **fresh site per run** (tasks change state); `start_site.sh`/`run_eval.sh` refuse a site whose product search
  does not return results. Compare methods only **paired** on the same tasks; one run has ≈ ±4–5 pt noise.
- One-node batch version: `sbatch --account=… --qos=… --partition=… jobs/eval.sbatch <name> [method]`
  (`ADAPTER=<dir>` to evaluate a LoRA).

## 2. Data generation
Trajectories are the data. Generate more with a teacher on the **train** split (never the test split):
```bash
sbatch … jobs/collect_teacher.sbatch 3 teacher397_train    # Qwen3.5-397B no_memory × 3 runs on train (≈ 2–3 h)
python datagen/build_sft_data.py --out $WA_WORK/sft/v3.jsonl \
    --teacher $WA_RUNS/teacher397_train_s1 $WA_RUNS/teacher397_train_s2 $WA_RUNS/teacher397_train_s3 \
    --self $WA_RUNS/base_q32_all
```
Output: one row per agent step, `{"messages": [system, user, assistant], ...}` exactly as the agent saw it (format:
[`rsi/finetune/DATA_FORMAT.md`](../../rsi/finetune/DATA_FORMAT.md)) + `<out>.summary.json` with filter counts.
Filters: train split only (asserted disjoint from test), successful trajectories only, no memory-augmented prompts, well-formed
`<think>…</think><action>…</action>` target, no parse retries, dedup; `--self` traces must also have a clean style.
`tools/export_traces.py` + `tools/split_traces.py` dump *all* steps (any split, success or not) for analysis.

## 3. Fine-tuning and evaluating the LoRA
```bash
TEACHER="…" SELF="…" OUT_NAME=lora_v3 sbatch … jobs/finetune.sbatch     # build data + 1-epoch LoRA on 4 GPUs (≈ 45 min / 1k steps)
ADAPTER=$WA_WORK/ckpt/lora_v3/final sbatch … jobs/eval.sbatch lora_v3   # TEST split, 3 shards
python tools/report.py --split test --baseline $WA_RUNS/base_q32 $WA_RUNS/lora_v3 $WA_RUNS/rb_q32
```
Lessons from v1/v2 (details in results): keep a single clean teacher style (mixing in telegraphic traces made the adapter
loop), 1 epoch, cap generation (`--max_new_tokens 8192`) when evaluating adapters.

## Changes vs upstream (patches/reasoning-bank.patch)
Local OpenAI-compatible endpoints for agent / embedding / grader (`utils/local_llm.py`); the vLLM reasoning field is
re-wrapped as `<think>` so history and memory induction see the thoughts; self-judge sees the page *after* the last action
and the full answer (released code truncated at the first ")"); action parsed only after `</think>`; per-step generation
cap; `--task_ids_file`, `--skip_done`, `--memory_dir`, `--judge_model`, `--cross_ref` (cross-check judge); grader retries
across server restarts. Prompts and the ReasoningBank algorithm are unchanged.
