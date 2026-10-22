# deltaserve/ — RSI on a live server: serve → learn from your own traces → train in the gaps

> 中文摘要：这个目录在 **一个 DeltaServe 服务进程** 上跑通完整的 RSI 闭环。
> 1. 服务接住 ACE 的用户请求，并记录 trace。
> 2. 把答对的 trace 转成 SFT 数据（与 `benchmarks/ace` 的 data_v2 做法完全一致），在线注入正在运行的服务（`POST /ft/ingest`）。
> 3. 服务继续接请求，同时在推理间隙训练 LoRA，训完导出 adapter。
>
> Qwen3-32B（YaRN），4×H200 上的结果：Formula test 从 68.0 提到 **75.0**（p=0.009），与离线完整 LoRA 持平。
> 训练只用了 4.6 分钟，期间推理 TTFT SLO 达标率 96.6%。用本目录脚本重新跑一遍得到 74.5 vs 66.5（p=0.007），SLO 达标率 99.1%。
> DeltaServe 的完整源码（含本实验的全部改动）就在 `deltaserve/DeltaServe-vLLM/`，不需要另外 clone。
> 一条命令复现：`bash deltaserve/scripts/setup.sh && bash deltaserve/scripts/run_e2e.sh`。

[DeltaServe](https://github.com/852866031/DeltaServe-vLLM) (source vendored here in `DeltaServe-vLLM/`) is a vLLM fork that **co-serves**: it trains a LoRA adapter
on the same GPUs, in the gaps between inference steps, under an SLO-aware admission scheduler. This folder uses it to
close the RSI loop of `benchmarks/ace` **online, on one server**, instead of the offline pipeline:

```
            ┌─────────────────────────── one DeltaServe server (Qwen3-32B, TP=4) ───────────────────────────┐
 users ───► │ serve requests ──► traces ──► keep correct ──► SFT rows ──► POST /ft/ingest ──► LoRA trains   │
            │      ▲                       (ACE scorer)     (data_v2 recipe)                 in the gaps    │
            │      └────────────────── inference keeps running (SLO-gated) ◄──────────────────────┘ final/ │
            └────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Headline results (Qwen3-32B, ACE Formula, 4× H200) — details in [results/RESULTS.md](results/RESULTS.md)

| | Formula test acc. | notes |
|---|---|---|
| base (empty playbook) | 68.0 | stock vLLM 0.31 |
| **end-to-end loop on one DeltaServe server** | **75.0** (+20/−6, p=0.009); repro run: **74.5** vs base 66.5 (p=0.007) | 427 self-generated rows, trained in 4.6 min while serving |
| offline full LoRA from `benchmarks/ace` (reference) | 75.0 | separate trainer, data from the ACE runs |

| inference latency, same traffic | inference only | while training |
|---|---|---|
| TTFT p50 / p99 | 38 / 54 ms | 84 / 725 ms (repro: 83 / 217 ms) |
| requests with TTFT < 0.3 s | 99.5 % | 96.6 % (repro: 99.1 %) |

## Layout

```
deltaserve/
  README.md               this file
  env.sh                  paths/ports; sources ../benchmarks/ace/env.sh (ACE clone, YaRN model, vLLM 0.31). Override in env.local.sh
  DeltaServe-vLLM/        DeltaServe source used for every result here (= upstream @ eea2a4a + the patch below)
    dserve-vllm/            the vLLM fork, pip-installable (`dserve-vllm`; Python import name `vllm`); DeltaServe code in vllm/deltaserve/
    configs/, eval-tp/, eval-rsi/, tests/, README.md, VLLM_FORK_CHANGES.md, EXPERIMENTS.md   DeltaServe's own configs, benches, gates, docs
    (upstream vLLM docs/ dropped to save space)
  patches/deltaserve-vllm.patch   for review only: DeltaServe-vLLM/ vs upstream eea2a4a (chat SFT data, YaRN backward,
                                  adapter export, /ft/ingest, long context, LoRA-slot fix, configs, tests, docs)
  configs/coserve_e2e.yaml        serving config for the end-to-end loop (template; run_e2e.sh fills __PLACEHOLDERS__)
  configs/train_only.yaml         the same recipe, training an existing SFT file with no traffic
  configs/empty_context.txt       the literal "(empty)" playbook used by the ACE adapters
  scripts/setup.sh                build the DeltaServe env from DeltaServe-vLLM/, write the initial LoRA
  scripts/run_e2e.sh              THE experiment: phases A → B → ingest → C → D (one command)
  scripts/run_train.sh            DeltaServe as an offline trainer on any RSI-system chat JSONL
  scripts/eval_adapter.sh         ACE test-split eval of base / an adapter (stock vLLM 0.31, same as benchmarks/ace)
  scripts/rsi_traffic.py          ACE "user traffic": ACE's Generator prompt + scorer, streamed, timed, traced
  scripts/traces_to_sft.py        traces → SFT rows (build_sft_data + make_data_v2 recipe) [+ POST /ft/ingest]
  scripts/compare.py              paired sign tests between evals
  scripts/summarize.py            one run → latency A vs C, trace yield, training, accuracy (summary.json)
  scripts/init_lora_adapter.py    zero-B PEFT LoRA for any Qwen3 checkpoint
  results/                        RESULTS.md + small artifacts of the reported runs (summary.json, bwd.csv, eval json)
```

## Requirements

- One node with **≥ 5 GPUs of ≥ 80 GB** (tested: 8× H200). 4 GPUs run the DeltaServe server (TP=4), one runs the eval.
- **The ACE benchmark set up first** (`benchmarks/ace/setup/install.sh`). This folder reuses its ACE clone + venv
  (`ACE_UPSTREAM_DIR`, `ACE_PY`), the YaRN model dir (`MODEL_DIR`), stock vLLM 0.31 (`VLLM_BIN`), and the learned
  formula playbook from an ACE formula offline +GT run (`$ACE_RESULTS_DIR/finance/formula_offline_gt/*/best_playbook.txt`).
- conda, [uv](https://docs.astral.sh/uv/), internet access for the precompiled vLLM kernels + torch, and a CUDA toolkit with `nvcc ≥ 12.9`
  (FlashInfer JIT; default `/usr/local/cuda-13.2`, set `DS_CUDA_HOME`).

## Setup (once)

```bash
cd RSI-system
# 0. paths: if your ACE dirs are not the defaults of benchmarks/ace/env.sh, put overrides in deltaserve/env.local.sh
#    (any variable of env.sh / benchmarks/ace/env.sh, e.g. ACE_WORK_DIR, MODEL_DIR, VLLM_BIN, DS_ROOT, DS_ENV)
bash deltaserve/scripts/setup.sh
#   env     -> conda env $DS_ENV (python 3.12) with deltaserve/DeltaServe-vLLM/dserve-vllm installed editable
#              (precompiled vLLM kernels are downloaded into it, git-ignored; torch cu130) + pyarrow
#   adapter -> $DS_INIT_ADAPTER: zero-B LoRA on q,k,v,o, r=16, alpha=32 (the adapter the server trains)
```
Already have a DeltaServe env? Set `DS_ENV` and run only `setup.sh adapter` (the env must be an editable install of
this `DeltaServe-vLLM/`, or of a checkout with the same code, given as `DS_ROOT`).

DeltaServe's own tests (CPU unless noted) run from the vendored tree:
```bash
cd deltaserve/DeltaServe-vLLM && source ../env.sh && ds_activate
python tests/test_chat_sft.py; python tests/test_ft_ingest.py; python tests/test_rope_yarn.py
python tests/test_long_context_backward.py; CUDA_VISIBLE_DEVICES="" python tests/test_adapter_export.py
```

## Run the end-to-end experiment

```bash
salloc --nodes=1 --gres=gpu:8 ...            # or inside an sbatch; any 8-GPU node
bash deltaserve/scripts/run_e2e.sh my_run    # outputs: deltaserve/work/runs/my_run/   (~45-60 min; first launch compiles 10+ min)
```

What happens (every phase logs to `driver.log`):

| phase | what the script does | output |
|---|---|---|
| start | launches DeltaServe on `DS_GPUS` (TP=4) with **no training data**; launch profiling fits the step-time estimator used for SLO-aware admission | `server.log`, `config.yaml` |
| **A** | formula **test** requests, empty playbook, Poisson `RPS` (0.5): inference-only baseline | `A.jsonl` (prompt, answer, correct, TTFT, E2E per request) |
| **B** | formula **train** requests **with the learned playbook**, 16 concurrent: the "user traffic" whose traces we learn from | `B.jsonl` |
| **I** | keep correct traces (ACE's scorer), re-render the prompt with an `(empty)` playbook, empty `bullet_ids` (= `benchmarks/ace` data_v2), `POST /ft/ingest` them into the live server, `POST /start_finetuning` | `sft_from_traces.jsonl` |
| **C** | the same traffic as A keeps flowing while the LoRA trains in the gaps; stops once `adapters/final/` is exported (2 epochs) | `C.jsonl`, `bwd.csv` (one row per backward cycle), `adapters/{cycle*,latest,final}` |
| **D** | stops the server, scores `adapters/final` and the base with the ACE protocol on `EVAL_GPU`, paired sign test | `eval/`, `summary.json`, `summary.txt` |

Read the result:
```bash
cat deltaserve/work/runs/my_run/summary.txt          # latency A vs C, traces kept, training cycles/loss, accuracy
python3 deltaserve/scripts/compare.py deltaserve/work/runs/my_run/eval formula base rsi
```
Knobs (environment): `RPS` (0.5), `B_CONCURRENCY` (16), `DS_GPUS` (0,1,2,3), `EVAL_GPU` (4), `DS_PORT` (8000),
`PLAYBOOK`, `SKIP_BASE_EVAL=1`. The training recipe and SLOs are in `configs/coserve_e2e.yaml`.

### Paper metrics (LoRA vs skills in context)
```bash
bash deltaserve/scripts/paper_run.sh paper1        # ~65 min on 8 GPUs: the loop + 16 evals + serving cost + leakage + GPU util
bash deltaserve/scripts/paper_run_cost2.sh paper1  # ~50 min: merged LoRA, fresh-prompt 4 rps, no-prefix-cache serving cost
cat deltaserve/work/runs/paper1/paper_metrics.md   # every table (scripts/paper_metrics.py)
```
What each part measures and the results: [results/RESULTS.md §3](results/RESULTS.md). Scripts: `eval_suite.sh`
(one vLLM server, {bench}:{empty|skills} evals), `eval_general.py` (GSM8K / MMLU), `serving_cost.sh` (TTFT/TBT/KV per
condition), `scrape_metrics.py` (vLLM /metrics at 1 Hz), `gpu_monitor.sh` (DCGM SM-active at 1 Hz),
`leak_probe.py` (prompt extraction), `merge_lora.py`.

### RSI traffic with Poisson arrivals: stop serving to train vs co-serving
```bash
export DATA=<the loop's sft_from_traces.jsonl> EPOCHS=2 TRAIN_AT=120
for r in 0.5 1 2; do
  RATE=$r DSERVE_BACKWARD_NCCL_PORT=29677 bash deltaserve/scripts/rsi_poisson.sh co  0,1,2,3 8000 deltaserve/work/runs/rsip/co_r$r &
  RATE=$r DSERVE_BACKWARD_NCCL_PORT=29777 bash deltaserve/scripts/rsi_poisson.sh seq 4,5,6,7 8100 deltaserve/work/runs/rsip/seq_r$r &
  wait
done
python3 deltaserve/scripts/rsi_poisson_metrics.py deltaserve/work/runs/rsip   # -> rsi_poisson_metrics.md
```
`rsi_traffic.py --train-at T --train-mode {co,seq} --ft-final <final adapter file>` drives the FT job from the traffic
client. Results: [results/RESULTS.md §5](results/RESULTS.md).

### The same, with TraceLab session timing instead of per-request Poisson
```bash
DB=<TraceLab syfi_coding_trace.duckdb>   # needs duckdb
for x in "0.5 0.0308" "1 0.0686" "2 0.147"; do set -- $x   # session rate chosen so the first 500 requests average $1 rps
  python deltaserve/scripts/agent_trace.py --db $DB --sessions 60 --rate $2 --out deltaserve/work/runs/rsit/workload_r$1.json
done
export DATA=<the loop's sft_from_traces.jsonl> EPOCHS=2 TRAIN_AT=120
for r in 0.5 1 2; do
  export RATE=$r WORKLOAD=deltaserve/work/runs/rsit/workload_r$r.json
  DSERVE_BACKWARD_NCCL_PORT=29677 bash deltaserve/scripts/rsi_poisson.sh co  0,1,2,3 8000 deltaserve/work/runs/rsit/co_r$r &
  DSERVE_BACKWARD_NCCL_PORT=29777 bash deltaserve/scripts/rsi_poisson.sh seq 4,5,6,7 8100 deltaserve/work/runs/rsit/seq_r$r &
  wait
done
python3 deltaserve/scripts/rsi_poisson_metrics.py deltaserve/work/runs/rsit       # + task completion table
python3 deltaserve/scripts/plot_rsi_poisson.py deltaserve/work/runs/rsit          # fig_{timeline,cdf,summary,tasks}.png
(cd deltaserve/scripts && python3 plot_rsi_poisson_simple.py ../work/runs/rsit 1 "TraceLab session timing")  # fig_simple_r1.png
```
Sessions arrive Poisson (TraceLab's own simulator does this: the corpus records no arrival process). Inside a
session, each round waits its real gap (tool latency or human think time), sends the next ACE request, and waits
for the answer. Results: [results/RESULTS.md §6](results/RESULTS.md).

### Sequential vs co-serving on a coding-agent workload (TraceLab)
```bash
# workload: real Claude Code sessions from the TraceLab trace (download the DuckDB from github.com/uw-syfi/TraceLab releases)
python deltaserve/scripts/agent_trace.py --db syfi_coding_trace.duckdb --sessions 48 --rate 0.05 --out deltaserve/work/agent/agent_workload.json  # needs duckdb
export WORKLOAD=$PWD/deltaserve/work/agent/agent_workload.json DATA=<sft.jsonl> EPOCHS=4
bash deltaserve/scripts/agent_run.sh co  0,1,2,3 8000 deltaserve/work/runs/agent1/co    # traffic + FT job on one server
DSERVE_BACKWARD_NCCL_PORT=29777 bash deltaserve/scripts/agent_run.sh ft 4,5,6,7 8100 deltaserve/work/runs/agent1/ft
bash deltaserve/scripts/agent_run.sh inf 4,5,6,7 8100 deltaserve/work/runs/agent1/inf   # stock vLLM, traffic only
python3 deltaserve/scripts/agent_metrics.py deltaserve/work/runs/agent1                  # -> agent_metrics.md
```
Two DeltaServe servers on one node need different `DSERVE_BACKWARD_NCCL_PORT`s. Results:
[results/RESULTS.md §4](results/RESULTS.md).

### Other entry points
```bash
bash deltaserve/scripts/run_train.sh $ACE_SFT_DIR/data_v2/formula.jsonl          # DeltaServe as a trainer, no traffic
bash deltaserve/scripts/eval_adapter.sh 4 9300 formula mytag <out_dir> <adapter_dir> # score any adapter (or the base)
python3 deltaserve/scripts/traces_to_sft.py traces.jsonl rows.jsonl --ingest http://127.0.0.1:8000   # feed a live server
```
`POST /ft/ingest` takes `{"rows": [{"messages": [...]}, ...]}` (the RSI-system SFT format,
`rsi/finetune/DATA_FORMAT.md`) and returns `{"added", "tokens", "corpus_samples", ...}`. Rows join the training
pool immediately; they train once `POST /start_finetuning` has opened admission.

## What the DeltaServe patch adds (all off unless configured)

| feature | config / API | why it is needed here |
|---|---|---|
| chat SFT data with assistant-only loss | `finetune.data_path: *.jsonl`, `chat_loss_on: ace\|last\|all`, `chat_thinking` | renders rows **token-for-token** like the ACE and `rsi/finetune` trainers (tested on real formula/finer/appworld rows) |
| runtime ingest | `POST /ft/ingest` | the server learns from data it produced while running; `data_path: null` starts it empty |
| YaRN in the backward | automatic from the model config | MODEL_DIR is YaRN x4; without it the LoRA gradients would use the wrong RoPE |
| PEFT adapter export | `adapter_save_dir`, `adapter_save_every` | `final/` loads directly into vLLM `--lora-modules` (TP shards merged) |
| long-context FT | `max_saved_finetuning_tokens`, `backward_batch_tokens` | samples up to 16K tokens; chunked attention + streaming LM head in the backward |
| LoRA-slot fix | — | vLLM's warm-up removed the pinned training adapter: without the fix the server published into the *inference* adapter's slot and never learned |

## Troubleshooting

| symptom | cause / fix |
|---|---|
| server dies ~10 min into the first launch with an Inductor `aten::slice ... SymInt` error | a launcher timeout killed torch.compile. The scripts use `--startup-timeout 3000`; keep it long on a cold cache |
| OOM at KV-cache allocation | vLLM's `gpu_memory_utilization` does **not** include DeltaServe's activation buffers (~58 GB/GPU at 16K) and backward process; keep **0.42** on 141 GB GPUs |
| `libcudart.so.12: cannot open` in FlashInfer sampling | a JIT cache built by another CUDA 12 env; `FLASHINFER_WORKSPACE_BASE` gives this env its own cache |
| training never starts on an idle server | the engine loop sleeps with no requests: send one request (run_train.sh does) |
| `Address already in use` when several evals start together | start them a few seconds apart or on distinct ports |
| loss identical across epochs | the LoRA-slot bug — make sure the patch is applied (`[deltaserve] re-pinned FT adapter in served LoRA slot 0` in `server.log`) |
| training starves under continuous traffic | do **not** use `fwd_token_throttle_enable: true` with threshold 0: it pauses the backward on every decode step |

## Limitations

- One loop round, formula only. The exported adapter is evaluated offline; hot-swapping it into serving for the next
  round is not automated yet.
- DeltaServe trains attention LoRA only (q,k,v,o), not the MLP. Samples > 16K tokens (AppWorld / WebArena) are
  skipped at ingest.
- The "(empty)" playbook file is `configs/empty_context.txt` (the literal `(empty)`, as in the ACE runs). The old
  0-byte `benchmarks/ace/eval/empty_context.txt` was removed and the `benchmarks/ace` sbatch scripts now point here.
