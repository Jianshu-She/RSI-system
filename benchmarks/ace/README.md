# ACE benchmark: Agentic Context Engineering reproduced with Qwen3-32B (+ LoRA context distillation)

> 中文摘要：本目录完整复现了 ACE（arXiv 2510.04618）论文中的 Finance（FiNER、Formula）和 AppWorld 实验，模型为本地 vLLM 部署的 Qwen3-32B（关闭 thinking，YaRN 扩展到 131k 上下文）。同时包含 ICL / Dynamic Cheatsheet 基线，以及把 ACE playbook“蒸馏”进权重的 LoRA SFT 实验。
> Formula 复现了论文结果（offline +GT 86.5，论文 85.5）。FiNER 和 AppWorld 的提升方向与论文一致，但幅度较小。结果见 [`results/RESULTS.md`](results/RESULTS.md)。

This directory packages a full reproduction of **ACE** (Agentic Context Engineering) on **Finance** (FiNER, Formula) and
**AppWorld**, with a local **Qwen3-32B**. It also includes the paper's context baselines (ICL, Dynamic Cheatsheet) and
**LoRA SFT** experiments that try to move the learned playbook into the weights. Upstream code is not vendored. Instead, it is cloned at
pinned commits and patched (`patches/`). Every path comes from one file, `env.sh`.

## Headline results (Qwen3-32B; paper = DeepSeek-V3.1)

| | Base | ICL | ACE offline +GT | ACE offline noGT | ACE online +GT | ACE online noGT | DC-CU +GT |
|---|---|---|---|---|---|---|---|
| Formula (acc %) | 68.9 | 82.0 | **86.5** (paper 85.5) | 81.0 (83.0) | 75.5 (76.5) | 79.5 (78.5) | 77.0 |
| FiNER (acc %) | 62.8 | 70.4 | **72.6** (78.3) | 65.8 (71.1) | 71.0 (76.7) | 60.8 (67.3) | 58.9 |
| AppWorld (avg TGC/SGC) | 27.2 (ReAct) | – | **35.2** (59.4) | 29.8 (57.2) | – | 31.4 (59.5) | – |

LoRA trained on ACE-distilled train traces, evaluated with an empty context (same-node base: Formula 69.0, FiNER 61.8):
- Formula: **75.0** (p=0.05).
- FiNER: **68.5** (p=0.0005).
- AppWorld: 28.6 (no significant effect).
- Plain label SFT on all FiNER train items scores 73.8.
- With **data selection** (difficulty-weighted, [`data/selected/`](data/selected/README.md)) the LoRA reaches Formula **81.5** and FiNER **75.2**. FiNER is now on par with the playbook in context (73.8), with no context at all.

Paired tests, val traces and caveats are in [`results/RESULTS.md`](results/RESULTS.md).

## Layout

```
env.sh                 single source of paths/ports (override via env vars or an untracked env.local.sh)
setup/install.sh       clone + patch upstream, venvs, AppWorld data, DC clone, vLLM, YaRN model dir (idempotent, per step)
setup/make_yarn_model.sh   $MODEL_DIR = symlinked Qwen3-32B weights + config.json with YaRN x4 (131072 ctx)
setup/requirements-*.lock.txt  exact package versions of the source venvs (AppWorld lock is installed verbatim)
patches/ace.patch, patches/ace-appworld.patch   diffs vs ace @82709de / ace-appworld @9f3e921
configs/qwen_*.jsonnet AppWorld experiment configs (authors' configs; provider local, model qwen3-32b, max_tokens 8192)
scripts/               serve_qwen.sh, run_finance.sh, run_icl.sh, run_dc.sh, run_appworld.sh, job_finance_remaining.sbatch
baselines/             build_icl_context.py (+ icl/ the exact ICL contexts used), run_dc_finance.py (DC-CU)
eval/                  report.py (results + paired tests), eval_split.py (score any context/LoRA on a split),
                       finance_snapshots.py + resume_finance.py (recovery). The "(empty)" playbook file is deltaserve/configs/empty_context.txt
datagen/               build_sft_data.py (ACE traces -> data_v1), make_data_v2.py (empty bullet_ids), build_sft_baselines.py,
                       ds_pool.py + ds_select.py (scored candidate pool -> selected SFT sets)
data/selected/         selected SFT sets for Formula/FiNER (ready to train) + gzipped candidate pools; see its README
finetune/              train_lora_ace.py (original trainer), job_*.sbatch (train + serve + eval), make_lora_catalog.py,
                       job_ds_pool.sbatch / job_ds_train_eval.sbatch (data selection: build pool / train + eval all sets)
results/               RESULTS.md, report_output.txt, lora_catalog/ (per-adapter data/training/test summary)
work/                  (created, git-ignored) upstream clones, venvs, model dir, results/, logs/, sft/
```

Upstream pins:
- `github.com/ace-agent/ace` @ `82709de` (submodule `ace-appworld` @ `9f3e921`)
- `github.com/suzgunmirac/dynamic-cheatsheet` @ `5cfe3c3` (unmodified)
- `Qwen/Qwen3-32B` @ `9216db57`

## 1. Setup

Requirements:
- `git`, `git-lfs` and [`uv`](https://docs.astral.sh/uv/).
- 1–8 GPUs with 80 GB or more each. The source runs used 8×H200: one TP=1 replica per GPU, and LoRA training of the 32B model fits on one H200.

```bash
cd benchmarks/ace
# optional: export ACE_WORK_DIR=/big/disk/ace  QWEN3_32B_SRC=/path/to/hf/snapshot  (or put them in env.local.sh)
setup/install.sh                 # all steps: upstream ace-venv appworld-venv appworld-data dc vllm model
setup/install.sh train-env       # optional LoRA env (torch 2.8.0 cu128, transformers 4.57.3, peft 0.18.0, flash-attn 2.8.1)
source env.sh
```

The install creates these environments (the same as the source project):

| Environment | Used for | How it is built |
|---|---|---|
| `$ACE_PY` = `work/ace/.venv` | ACE finance, DC, eval scripts | `uv sync --frozen` of upstream `uv.lock`, Python 3.12 |
| `work/ace/ace-appworld/.venv` | AppWorld | Python 3.11, deps from `requirements-appworld.lock.txt` |
| `$VLLM_BIN` | serving | vLLM 0.31.0 |
| `$TRAIN_PY` | LoRA training | the source used the conda env `verl` |

The AppWorld lock is equivalent to `uv --exclude-newer 2025-10-20` plus `click<8.2`, because 2026 versions of pydantic and click break appworld.

Variables you may need to set (see `env.sh`):
- `ACE_WORK_DIR`, `MODEL_DIR` or `QWEN3_32B_SRC`.
- `VLLM_BIN` if you already have vLLM.
- `TRAIN_PY`.
- `ACE_PORTS`. The source used GPU0 8100, GPU1 8110, and GPU2-7 8102-8107.

**SLURM**: the `*.sbatch` files have `--account=CHANGE_ME --qos=CHANGE_ME --partition=main`. The source cluster used
`account=deploy, qos=deploy, partition=main`. Edit them or override them, for example `sbatch --account=X --qos=Y finetune/job_lora.sbatch`.
Submit from `benchmarks/ace` (the jobs source `$SLURM_SUBMIT_DIR/env.sh`) or export `ACE_ROOT`. Each job writes
`slurm-ace-<jobid>.out` in the submit directory.

## 2. Serve the model

```bash
scripts/serve_qwen.sh 0 8100          # start GPU0 first (warms the torch.compile cache), then the others
for g in 1 2 3; do p=$(echo $ACE_PORTS | cut -d' ' -f$((g+1))); setsid nohup scripts/serve_qwen.sh $g $p > $ACE_LOGS_DIR/vllm_gpu$g.log 2>&1 & done
curl 127.0.0.1:8100/v1/models          # lists qwen3-32b
```

Each replica uses TP=1, max-model-len 131072 and prefix caching. The server caps `max_new_tokens` at 8192 by default.
Clients send temperature 0 and `enable_thinking=false`. Kill servers by PID, never with a `pkill -f` pattern that also
matches your own shell.

## 3. Run evaluations (ACE, baselines)

Write logs to `$ACE_LOGS_DIR/<name>.log`. `eval/report.py` reads progress from `fin_<task>_<mode>_<gt|nogt>.log`.

```bash
# Finance ACE (8 runs). Each run first tests the empty playbook (= base), then adapts, then tests.
scripts/run_finance.sh formula offline gt 8100 > $ACE_LOGS_DIR/fin_formula_offline_gt.log 2>&1
#   task: finer|formula   mode: offline|online   labels: gt|nogt
# ICL baseline (contexts shipped in baselines/icl/; rebuild: $VLLM_PY baselines/build_icl_context.py <task> $ACE_ICL_DIR/<task>_icl_context.txt)
scripts/run_icl.sh formula 8110
# Dynamic Cheatsheet (cumulative, official prompts, Python execution on); re-run resumes from steps.jsonl
scripts/run_dc.sh formula gt 8104 > $ACE_LOGS_DIR/dc_formula_gt.log 2>&1
# AppWorld smoke test (1 dev task; delete its output dir afterwards):
(cd $APPWORLD_DIR && APPWORLD_PROJECT_PATH=$PWD LOCAL_LLM_URL=$(ace_url 8104) $APPWORLD_BIN run qwen_react_eval \
    --task-id 50e1ac9_1 --override '{"config":{"dataset":"dev"}}')
# AppWorld: ReAct base / offline ACE ±GT / online ACE (a comma list of ports spreads processes over replicas)
scripts/run_appworld.sh 8104 eval:qwen_react_eval:test_normal eval:qwen_react_eval:test_challenge \
    score:qwen_react_eval:test_normal score:qwen_react_eval:test_challenge
scripts/run_appworld.sh 8105 adapt:qwen_off_gt_adapt eval:qwen_off_gt_eval:test_normal eval:qwen_off_gt_eval:test_challenge \
    score:qwen_off_gt_eval:test_normal score:qwen_off_gt_eval:test_challenge        # same with qwen_off_nogt_*
scripts/run_appworld.sh 8107 online:qwen_online_normal score:qwen_online_normal:test_normal   # + qwen_online_challenge
# Results table with paired sign tests (stdlib only; reads $ACE_RESULTS_DIR, $ACE_LOGS_DIR, $APPWORLD_DIR)
python3 eval/report.py            # --json for machine-readable output
```

To score an arbitrary context (`-` = empty, ICL file, playbook) or a LoRA adapter on any finance split, run from `$ACE_UPSTREAM_DIR`:
`LOCAL_LLM_URL=$(ace_url 8100) [EVAL_MODEL=<lora name>] [EVAL_LOG_DIR=<dir>] $ACE_PY $ACE_ROOT/eval/eval_split.py <task> <train|val|test> <ctx|-> out.json`.

Evaluation rules (fixed before any results were seen):
- Offline: test the playbook with the best val score (val is evaluated every 100 steps). Never select on test.
- Online: report cumulative accuracy, where each 15-item window is scored before it updates the playbook.
- AppWorld: every task id must have `tasks/<id>/dbs` before `appworld evaluate`. Re-run missing ids with `--override '{"config":{"task_ids":[...]}}'`.

**Recovery (node loss).** ACE processes samples in file order and its only state is the playbook, so resuming from a snapshot is exact:
- Resume a finance run with `eval/resume_finance.py <task> <mode> <gt|nogt> <N> <snapshot> <save_path> [eval_steps]`.
- `eval/finance_snapshots.py latest|best <task> <gt|nogt>` finds the latest snapshot or the best-val snapshot across segments.
- `scripts/job_finance_remaining.sbatch` is a re-submittable job that resumes FiNER offline ±GT, runs the final test, and resumes DC FiNER.

## 4. Generate SFT data (context distillation)

Inputs are only the TRAIN-split traces of the offline ACE runs. No val or test traces are used.

```bash
$VLLM_PY datagen/build_sft_data.py $ACE_SFT_DIR/data_v1          # formula/finer/appworld.jsonl + summary.json
python3 datagen/make_data_v2.py $ACE_SFT_DIR/data_v1 $ACE_SFT_DIR/data_v2   # REQUIRED fix: empty bullet_ids in targets
```

How the data is built:
- **Finance**: `build_sft_data.py` takes first-attempt generator calls on train items (`train_e_1_s_<k>_gen_initial`) whose answers are correct.
  - Input = the ACE generator prompt with `(empty)` playbook and reflection, i.e. exactly the base prompt.
  - Target = the response generated *with* the learned playbook.
- **AppWorld**: it takes the successful train trajectories of `qwen_off_{gt,nogt}_adapt`. The learned playbook is replaced by the shipped initial playbook, and +GT retry turns are dropped.
- Output format is `{"messages":[...], "meta":{...}}` JSONL.
- `FIN_RUNS` in the script lists the source project's resume segments. Edit it to match your runs.
- Reference statistics of the source build are in `datagen/reference/data_v1_summary.json`:
  - Formula: 803 examples (417 items).
  - FiNER: 277 examples (207 items).
  - AppWorld: 122 trajectories (77 tasks).
- `make_data_v2.py` reproduces the source `data_v2` byte for byte.
- SFT baselines (gold labels, base self-distillation) are built inside `finetune/job_sft_baselines.sbatch` by `datagen/build_sft_baselines.py`.

## 5. Fine-tune with LoRA

The recipe is LoRA r=16 / α=32 / dropout 0.05 on q,k,v,o,gate,up,down, with:
- lr 1e-4, cosine schedule, 5% warmup, about 32k tokens/step, 2 epochs, seed 42;
- bf16, gradient checkpointing, flash-attn 2;
- 5% of rows held out as dev.

The loss is computed only on assistant tokens. Thinking is off, and prompts are token-identical to inference.

```bash
CUDA_VISIBLE_DEVICES=0 $TRAIN_PY finetune/train_lora_ace.py --data $ACE_SFT_DIR/data_v2/formula.jsonl \
    --out $ACE_SFT_DIR/runs/my_run/adapter_formula --epochs 2 --lr 1e-4 --rank 16 --alpha 32 --dropout 0.05 \
    --tokens_per_step 32768 --seed 42
```

**Trainers.** The generic, benchmark-independent trainer of this repo is **`rsi/finetune/train_lora.py`** at the repo root. It accepts
`{"messages":[...]}` JSONL and takes `--thinking off|on` and `--loss_on all|last`. Use `--thinking off` to match this benchmark. For
single-turn finance data, `all` and `last` are the same. `finetune/train_lora_ace.py` is the original trainer used for every number in
RESULTS.md, kept for exact reproduction. For AppWorld it masks the authors' few-shot demo dialogue: loss starts only after the
"solve the actual task" message.

## 6. Evaluate the LoRA

Each SLURM job trains (or loads) the adapters, serves the base model with `--enable-lora` on free ports, and evaluates on TEST
with the `(empty)` prompt:

| Job | What it does |
|---|---|
| `finetune/job_lora.sbatch` | trains formula, finer and appworld adapters on data_v1, then evaluates all of them (AppWorld uses `configs/qwen_lora_react_eval.jsonnet`, which expects the vLLM LoRA name `appworld`) |
| `finetune/job_lora_fin.sbatch $ACE_SFT_DIR/data_v2 lora_r2_fin` | trains the finance adapters on data_v2 and evaluates LoRA, and LoRA plus the ACE playbook (base not re-run) |
| `finetune/job_lora_eval_finance.sbatch <run_dir>` | finance evaluation only, including the same-node base control |
| `finetune/job_lora_eval.sbatch <run_dir>` | AppWorld LoRA evaluation (+ Formula diagnostic with raw generations) |
| `finetune/job_sft_baselines.sbatch` | base answers on TRAIN, then 8 SFT-baseline adapters (gold/self × all/matched), then test evaluation |
| `python3 finetune/make_lora_catalog.py` | collects data, training stats and test results into `$ACE_SFT_DIR/lora_catalog/`. Edit the job ids in the script to point at your runs |

Smoke-test one request per adapter name before a long evaluation. A wrong model name returns 404s and wastes the run.

## Code changes vs upstream (all robustness; none change the method)

| File | Change |
|---|---|
| `utils.py`, `llm.py`, `eval/finance/run.py` | `local` provider (`LOCAL_LLM_URL`, comma list = replicas), thinking off unless `LOCAL_ENABLE_THINKING=1` |
| `experiments/code/ace/lite_llm_generator.py` | `local` provider. On a context-overflow 400, halve max_tokens, else return empty content (one long trajectory used to crash a whole run) |
| `experiments/code/ace/__init__.py` | removed imports of 2 modules missing from the release |
| `src/appworld/requester.py` | API-call log falls back to `repr` for values that cannot be serialized (Ellipsis crashed eval shards) |
| `configs/qwen_*.jsonnet` | authors' configs with provider `local`, model `qwen3-32b`, `max_tokens` 8192. Trained playbooks are written to `$ACE_RESULTS_DIR/appworld/playbooks/` through `std.extVar("ACE_RESULTS_DIR")`, which `run_appworld.sh` exports via `env.sh` |

## Not included

- Model weights.
- The upstream repos.
- Run outputs (about 15 GB of traces, playbooks and adapters).
- SFT JSONL data. Regenerate it with `datagen/`.

The full chronological experiment log (every spec, prediction, incident and leak audit) is `EXPERIMENTS.md` in the source project.
Its key content is summarised in `results/RESULTS.md`.
