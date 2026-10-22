# Results — DeltaServe co-serving RSI loop (Qwen3-32B, ACE Finance)

All runs: Qwen3-32B with the YaRN x4 config (`benchmarks/ace` MODEL_DIR), DeltaServe TP=4 on 4× H200, LoRA
q,k,v,o r=16 α=32 (zero-B init), lr 1e-4 constant, 2 epochs, 16K tokens per optimizer step, loss on assistant tokens
only (rendered like the ACE LoRA trainer). Accuracy is always measured with the ACE protocol (`scripts/eval_adapter.sh`):
- stock vLLM 0.31, TP=1, temperature 0;
- **empty playbook**;
- formula **test** split (n=200).

p = two-sided sign test on per-item correctness vs base (as in `benchmarks/ace/results`).

## 1. End-to-end loop on one server (`scripts/run_e2e.sh`)

The server starts with **no training data**. It serves the traffic, its own correct traces become SFT rows, it
ingests them while serving, and it trains in the gaps of the continuing traffic.

### Run 1 — 2026-10-19 (`e2e_2026-10-19/`)

| phase | what | result |
|---|---|---|
| A | formula test traffic, empty playbook, 0.5 rps (inference only) | base 67.0 % on DeltaServe; TTFT p50/p99 38/54 ms |
| B | formula train traffic **with the learned playbook** (29,747 tokens, prefix-cached), 16 concurrent | 427/500 correct (85.4 %) in 3 min |
| I | correct traces → SFT rows (`(empty)` playbook, `bullet_ids` → `[]`) → `POST /ft/ingest` | 427 rows, 263,053 tokens, ingested live |
| C | same traffic as A while training | 34 backward cycles (526K tokens) in **4 min 35 s**; mean loss 0.291 (epoch 1) → 0.155 (epoch 2) |
| D | exported adapter, ACE protocol | **75.0 %** (base 68.0; +20/−6, p = 0.009) |

Inference latency, same traffic and same server:

| | A: inference only (n=200) | C: while training (n=116) |
|---|---|---|
| TTFT p50 / p90 / p99 / max | 38 / 45 / 54 / 770 ms | 84 / 176 / 725 / 1077 ms |
| requests with TTFT < 0.3 s (SLO) | 99.5 % | **96.6 %** |
| end-to-end p50 / p90 / p99 | 1.70 / 2.54 / 4.01 s | 2.30 / 5.24 / 6.66 s |
| errors | 0 | 0 |

For reference, the offline adapters in `benchmarks/ace/results` (full LoRA incl. MLP, trained on the ACE runs' data_v2)
score 75.0 on formula.

### Run 2 — reproduction with these exact scripts (`e2e_2026-10-19_repro/`)

Fresh run of `bash deltaserve/scripts/run_e2e.sh` as documented in the README: DeltaServe-vLLM cloned from GitHub at
eea2a4a plus `patches/deltaserve-vllm.patch`, which is byte-identical to the run-1 code (and to the source now
vendored in `deltaserve/DeltaServe-vLLM/`).

| | run 1 | run 2 (repro) |
|---|---|---|
| A: base accuracy on DeltaServe (inference only) | 67.0 % | 67.5 % |
| B: correct traces | 427/500 | 427/500 |
| ingested | 427 rows / 263,053 tok | 427 rows / 262,947 tok |
| C: training while serving | 34 cycles, 526K tok, 4 min 35 s | 34 cycles, 526K tok, 4 min 37 s |
| C: epoch mean loss | 0.291 → 0.155 | 0.290 → 0.155 |
| **D: adapter test accuracy** (ACE protocol) | **75.0** vs base 68.0 (+20/−6, p=0.009) | **74.5** vs base 66.5 (+24/−8, p=0.007) |
| A: TTFT p50 / p99, SLO < 0.3 s | 38 / 54 ms, 99.5 % | 36 / 51 ms, 99.5 % |
| C: TTFT p50 / p99, SLO < 0.3 s | 84 / 725 ms, 96.6 % | 83 / 217 ms, **99.1 %** |

The adapter gain reproduces: +7 and +8 points, both p < 0.01. Base accuracy moves by ~1.5 points between evals even
at temperature 0 (vLLM batching), which is why both runs re-measure the base. The co-serving p99 TTFT varies between
runs (725 vs 217 ms), so the tail is sensitive to how training cycles happen to overlap prefills.

## 2. Training only on the ACE SFT data (`scripts/run_train.sh`; `train_only_2026-10-19/`)

DeltaServe trained `sft/data_v2/{formula,finer}.jsonl` (the files behind `benchmarks/ace` LoRA results) with no
inference traffic. A control re-ran the ACE trainer **attention-only** (q,k,v,o), because DeltaServe does not train MLP
LoRA.

| test accuracy | base | ACE trainer, attention-only | **DeltaServe** | ACE trainer, full LoRA (catalog) |
|---|---|---|---|---|
| Formula (n=200) | 68.0 | 72.0 (+17/−9, p=0.17) | **75.5** (+23/−8, p=0.011) | 75.0 |
| FiNER (n=441 tags) | 61.5 | 61.3 | **71.2** (+80/−15, p<1e-4) | 68.5 |

Training time on 4× H200: formula 3 min 49 s (62 cycles), FiNER 7 min 9 s (94 cycles).

The DeltaServe recipe takes 2.2× more optimizer steps (16K vs 32K tokens/step) at a constant LR, and the control used
cosine with warmup. The gap to the attention-only control is therefore most likely the schedule, not the system. The
supported claim is that **a DeltaServe-trained adapter reproduces the offline LoRA gains under the ACE eval.**

## 3. Paper metrics — LoRA vs skills in context (`scripts/paper_run.sh` + `scripts/paper_run_cost2.sh`; `paper1_2026-10-20/`)

One more run of the loop above (adapter: same recipe, formula 68.5 → 75.0). "Skills" = the ACE formula playbook
(282 bullets, 30.1K tokens) in the prompt; "LoRA" = the adapter the server trained from its own traces. Full tables:
`paper1_2026-10-20/paper_metrics.md`.

### 3.1 Accuracy: {base, LoRA} × {no skills, skills}  (test splits; paired sign test)

| bench | n | base | base + skills | LoRA | LoRA + skills |
|---|---|---|---|---|---|
| formula (trained task) | 200 | 68.5 | **86.0** | **75.0** | 85.5 |
| FiNER (other finance task) | 441 | 61.8 | 57.2 | 61.9 | 56.1 |
| GSM8K | 1319 | 95.5 | 94.8 | 94.9 | 94.2 |
| MMLU (1000 of all/test, seed 0) | 1000 | 76.6 | 77.1 | 76.4 | 77.7 |

- **LoRA + skills > skills: not supported here.** 85.5 vs 86.0 on formula (+2/−3, p=1.0). The LoRA was distilled
  from traces made *with this playbook*, so it holds a subset of the same knowledge. Without any context it recovers
  6.5 of the playbook's 17.5 points (37 %): 75.0 vs 68.5, p=0.029.
- **Distilled experience does not hurt general ability.** LoRA vs base: FiNER +0.1 (p=0.27), GSM8K −0.5 (p=0.28),
  MMLU −0.2 (p=0.77).
- **Skills in context do hurt outside their task.** The formula playbook on FiNER costs −4.6 points (+19/−49,
  **p=0.0004**). GSM8K and MMLU are unaffected (−0.7 and +0.5, n.s.).

### 3.2 Context, KV cache, TTFT/TBT  (stock vLLM 0.31, TP=4 on 4× H200, formula traffic; KV pool 1.78M tokens)

"customized model" = the LoRA **merged** into the weights (`merge_lora.py`; 74.5 % on the test traffic). Merging
gives exactly base latency: TTFT 41 vs 43 ms, TBT 6.7 vs 7.0 ms at 1 rps. Unmerged vLLM multi-LoRA adds +67 ms
TTFT and +2.6 ms TBT.

| | prompt tokens | KV per request | TTFT p50 / p99 | TBT p50 / p99 | E2E p50 | KV pool used (mean / peak) |
|---|---|---|---|---|---|---|
| customized model, 4 rps | 391 | 0.10 GiB | 42 / 85 ms | 6.9 / 24.7 ms | 1.77 s | 0.1 / 0.3 % |
| skills, shared + prefix-cached, 4 rps | 30,137 (77×) | 7.36 GiB | 119 / 861 ms | 13.4 / 46.8 ms | 4.06 s | 2.2 / 2.7 % |
| skills, per user (no sharing), **1 rps** | 30,136 | 7.36 GiB | **34.0 / 99.2 s** | 318 / 488 ms | 117 s | **76 / 99.6 %** |
| base, no sharing, 1 rps (reference) | 390 | 0.10 GiB | 38 / 55 ms | 6.7 / 8.5 ms | 1.32 s | 0.0 / 0.2 % |

- The skills take 30.1K of Qwen3-32B's native 32K window: this model needs YaRN just to fit them plus an answer.
- Even in the best case for skills (one playbook shared by every request and prefix-cached), skills cost 2.8× TTFT
  p50, 10× TTFT p99 and 1.9× TBT.
- With per-user skills nothing is shared. Each request holds 7.36 GiB of KV, so the 1.78M-token pool fits 58 of
  them (59 measured). At just 1 rps the queue grows without bound (63 waiting, TTFT 34 s).
- The 4-rps rows of the first serving pass repeated the 1-rps prompts (full-prompt cache hits). The table above
  uses fresh prompts (formula train split) instead. Accuracy on that split is in-sample for the adapter and is not
  reported.

### 3.3 GPU utilization of co-serving  (DeltaServe, GPUs 0-3, the same 0.5-rps traffic; DCGM 1 Hz)

| | SM active (DCGM) | GPU util (nvidia-smi) | power / GPU | TTFT p50 / p99 | TTFT < 0.3 s | TBT p50 / p99 |
|---|---|---|---|---|---|---|
| A: inference only | 19.2 % | 59 % | 274 W | 35 / 51 ms | 99.5 % | 8.7 / 9.8 ms |
| C: inference + LoRA training in the gaps | **56.0 %** (2.9×) | 96 % | 493 W | 81 / 268 ms | 99.1 % | 8.3 / 25.8 ms |

In C the same GPUs also trained 526K tokens (1873 FT tok/s; 34 cycles, 281 s) from 427 self-generated rows.

### 3.4 Leakage: prompt-extraction attacks on each deployment  (12 attacks, union over attacks, 282 bullets)

| deployment | bullet ids leaked | bullets reproduced verbatim |
|---|---|---|
| skills in context | 282 (100 %) | 149 (52.8 %) |
| LoRA (no context) | 1 (0.4 %) | 7 (2.5 %)* |
| base, no context (control) | 1 (0.4 %) | 0 |

\* All 7 come from one sentence, the textbook Sharpe-ratio definition ("(R_p − R_f)/σ_p, where R_p is the portfolio
return, R_f is the risk-free rate…"), which appears in 7 bullets. It was elicited by the attack that asks how to
compute Sharpe. It is public knowledge, not playbook content. The single id (calc-00001) is guessable: the base
model emits it too.

### 3.5 How Claude Code manages skills (for comparison)

- Claude Code uses progressive disclosure. Each skill's name + one-line description is resident in the context
  for the whole session.
- The full SKILL.md body enters the context only when the model invokes that skill, and then stays for the rest of
  the conversation. Bundled scripts and references are read on demand.
- So the per-skill cost is bounded but never zero. With many skills, the description list itself is truncated to
  fit a budget.
- A LoRA moves the experience into the weights: zero context and KV cost, nothing extractable from the prompt.
  That holds for skills learned from this kind of loop.

## 4. Sequential vs co-serving on a coding-agent workload (TraceLab; `scripts/agent_*`; `agent1_2026-10-20/`)

**Question:** two jobs share 4× H200 (Qwen3-32B, TP=4):
- the agent traffic, and
- a LoRA FT job: the paper-1 loop's 427 correct trajectories × 4 epochs = 1.05M tokens.

Should they run disaggregated in time (serve, then train) or co-served on one DeltaServe server? How long does each
take to complete?

**Workload** (`agent_trace.py`; sessions are real Claude Code sessions from the TraceLab trace, [arXiv:2606.30560](https://arxiv.org/abs/2606.30560), [uw-syfi/TraceLab](https://github.com/uw-syfi/TraceLab) v0.0.1):
- 48 sessions sampled from 2,510, seed 0. Poisson arrivals at 0.05/s.
- 900 LLM rounds, 83 user tasks; each session cut at 4 tasks / 80 rounds.
- Per round, from the trace:
  - appended tokens, ×1/8 (p50/p90 141/918);
  - output tokens (p50/p90 184/1,115);
  - tool latency before a tool-initiated round (p50 0.09 s, p99 182 s);
  - human think time before a user round (p50 129 s).
  - Both gaps are capped at 300 s.
- The ×1/8 makes TraceLab's 100K+-token contexts fit the 40K window. The 300 s cap is because 1–2 h gaps are a
  person away.
- Replay (`agent_replay.py`):
  - Closed loop per session over a running context: a shared 2K system block plus history, with the output re-sent
    as input.
  - Synthetic token ids and forced output length.
  - Auto-compaction to 4K at 36,864 tokens (15 compactions).
- **Task completion** = from sending a user message to the end of the last LLM round of its tool loop. The tool and
  human waits are the trace's, identical in every mode.

**Runs** (`agent_run.sh`):
- `inf`: stock vLLM 0.31, gpu_memory_utilization 0.90 (KV 1.76M tokens).
- `ft`: DeltaServe, no traffic.
- `co`: DeltaServe (KV 562K tokens). The traffic and the FT job both start at t=0.
- Sequential = inf + ft.

### 4.1 Completion times

| | both jobs done | FT job done | agent task completion p50 / p90 / p99 |
|---|---|---|---|
| sequential: serve, then train | 2,386 s | 2,386 s | 38.9 / 272 / 905 s |
| sequential: train, then serve | 2,386 s | 255 s | 42.3 / 354 / 922 s (tasks arriving during FT wait) |
| **co-serving** | **2,161 s (0.91×)** | 1,612 s | 46.6 / 306 / 936 s (**+20 %** p50) |

### 4.2 Detail

| | inference only | co-serving |
|---|---|---|
| rounds / errors / compactions | 900 / 0 / 15 | 900 / 0 / 15 |
| replay makespan | 2,130 s | 2,160 s |
| task LLM time p50 / p90 / p99 (excl. tool/human waits) | 20.1 / 107 / 181 s | 23.9 / 130 / 214 s (+19 %) |
| round latency p50 / p90 / p99 | 1.65 / 9.4 / 33.0 s | 2.00 / 11.4 / 40.2 s |
| TTFT p50 / p90 / p99 | 152 / 328 / 959 ms | 179 / 376 / 1,130 ms |
| TBT (per-round mean) p50 / p99 | 8 / 16 ms | 10 / 23 ms |
| GPU SM_ACTIVE (DCGM, replay window) | 33.4 % | 36.4 % |
| wall time with no request in flight | 35 % | 30 % |

| FT job | alone | co-serving |
|---|---|---|
| time / throughput | 255 s / 4,122 tok/s | 1,612 s / 653 tok/s (16 % of alone) |
| loss first → last | 0.523 → 0.054 | 0.523 → 0.053 (same 68 cycles) |
| FT-only forwards aborted by an arrival (tier C) | — | 43 (≈ 0.36M token-equivalents of discarded forward) |

### 4.3 Reading

- **Co-serving finishes both jobs 225 s (9 %) sooner.**
  - The FT job completes inside the agent traffic, and the traffic itself runs only 30 s longer.
  - The saving is bounded by the FT job's standalone time (255 s). A job that big is small next to 36 minutes of
    agent traffic.
  - The saving grows with the training job: with 9.4M tokens it would be up to ~38 min of training hidden in the
    gaps. That is not measured here.
- **The agent pays for it: +20 % task completion at p50, +19 % LLM time per task.**
  - The causes are the backward child time-slicing with decode steps (TBT +25 %) and the 3× smaller KV pool.
  - It fails the agent-1 criteria: makespan ≤ 0.8× and task p50 ≤ +10 %.
- **The gaps are there, but DeltaServe does not use them well on agent traffic.**
  - The GPU is idle most of the time: SM_ACTIVE 33 %, and 35 % of wall time has no request at all.
  - Still, co-serving trains at only 16 % of its standalone rate.
  - The traffic arrives as fine-grained bursts: tool waits are p50 0.09 s, and a task is a chain of ~11 rounds.
  - A 16K-token FT-only forward (≈ 2–3 s) is aborted by the next arrival and re-run from scratch. Training stalled
    for 9 minutes at peak load (11:28–11:37).
  - The obvious fixes are smaller FT forwards in fragmented gaps, keeping the partial forward instead of discarding
    it, and letting FT ride inference prefills. They are untested.
- **Train-then-serve** gives the FT result fastest (255 s), but every session that arrives meanwhile waits: p90 task
  completion rises from 272 to 354 s.

Caveats:
- One run per mode.
- `inf` is stock vLLM 0.31 while `co`/`ft` are the DeltaServe fork (v0.21). The DeltaServe fork with finetuning off
  died on this workload with a flash-attn illegal memory access (`inf_fork_crashed/error_excerpt.txt`; open bug).
  The newer engine may be faster, so the +20 % may overstate co-serving's cost relative to the fork serving alone.
- `co` and `ft`/`inf` ran on the same node at the same time, on disjoint GPUs.
- Token lengths are scaled ×1/8 and gaps are capped, so absolute times are not TraceLab's.

## 5. RSI traffic with Poisson arrivals: stop serving to train vs co-serving (`scripts/rsi_poisson.sh`; `rsi_poisson_2026-10-20/`)

**Question:** in the RSI loop, traffic does not stop while the model trains. A single deployment can do one of two
things:
- **seq:** stop serving to train on the same GPUs. Requests that arrive meanwhile wait.
- **co:** train while serving (DeltaServe).

What does each cost in request completion time, and when is the trained adapter ready?

**Setup:**
- One DeltaServe server config for both modes (train_only.yaml plus the SLO profiling pass), so the engine and
  memory (KV 562K) are identical.
- Traffic: ACE formula, 500 requests (train split, empty playbook) to the base model. Arrivals are Poisson at 0.5,
  1 or 2 rps; this arrival process is what TraceLab's csv_export synthesizes.
- FT job: the loop's 427 correct trajectories × 2 epochs = 526K tokens (the RSI recipe). It starts at t = 120 s.
- seq: hold arrivals, drain in-flight requests (≈ 2 s), train alone, then release the held requests.
- Request completion = done − arrival.

Figures (`python scripts/plot_rsi_poisson.py results/rsi_poisson_2026-10-20`):

![timeline](rsi_poisson_2026-10-20/fig_timeline.png)

*Top: completion time of each request by arrival time. Under seq, every request that arrives during the FT job
(shaded) waits for it. Bottom: FT tokens trained over time. Under co, the FT rate drops as load rises; at 2 rps
nothing trains until the traffic ends.*

![cdf](rsi_poisson_2026-10-20/fig_cdf.png)

![summary](rsi_poisson_2026-10-20/fig_summary.png)

*The trade-off: co keeps p90 at 2.6–3.5 s, and the adapter is ready 140–309 s later.*

| rate | mode | request completion p50 / p90 / p99 | mean | requests held (mean wait) | FT job time | adapter ready at | all done at | TTFT p50 / p99 |
|---|---|---|---|---|---|---|---|---|
| 0.5 | seq | 1.64 / 6.35 / **119.8 s** | 9.06 s | 51 (70.8 s) | 126 s | 246 s | 1,042 s | 36 / 517 ms |
| 0.5 | **co** | 1.67 / 2.95 / **6.4 s** | 1.92 s | 0 | 306 s | 426 s | 1,042 s | 37 / 254 ms |
| 1 | seq | 1.92 / **86.0** / 127.3 s | 20.7 s | 135 (67.4 s) | 126 s | 248 s | 522 s | 40 / 1,712 ms |
| 1 | **co** | 1.82 / **3.54** / 6.2 s | 2.15 s | 0 | 436 s | 556 s | 556 s | 41 / 510 ms |
| 2 | seq | 6.46 / **107.6** / 131.2 s | 36.4 s | 249 (64.7 s) | 126 s | 247 s | 266 s | 907 / 2,136 ms |
| 2 | **co** | 1.71 / **2.63** / 5.5 s | 1.89 s | 0 | 267 s | 387 s | 387 s | 39 / 80 ms |

FT progress while the traffic was running (co):

| rate | FT tokens done while serving | FT rate while serving | rest trained after the traffic ended |
|---|---|---|---|
| 0.5 | all 526K | 1,720 tok/s (41 % of alone, 4,180 tok/s) | — |
| 1 | 392K / 526K | 973 tok/s (23 %) | 34 s |
| 2 | **0** | 0 | all 125 s |

Accuracy is 344–349/500 in every cell, i.e. the same; the traffic goes to the base model in both modes.

**Reading:**
- **Co-serving never interrupts service.**
  - Its request completion p90 is 2.6–3.5 s at every rate, vs 6.4 / 86 / 108 s for seq. Its p99 is ≤ 6.4 s vs
    120–131 s.
  - Under seq, every request that arrives during the 126 s training window waits ~65–71 s on average: 10 / 27 /
    50 % of all requests at 0.5 / 1 / 2 rps.
  - At 2 rps the released burst also inflates TTFT (p50 907 ms).
- **The price is the adapter arriving later.**
  - The FT job takes 2.1–3.5× longer co-served (267–436 s vs 126 s alone). The adapter is ready 180 / 309 / 140 s
    later than under seq.
  - Everything finishes at the same time at 0.5 rps, 34 s later at 1 rps, and 121 s later at 2 rps. The 500-request
    traffic is finite; under continuous RSI traffic there is no "all done", only the service gap vs the adapter's
    lateness.
- **At 2 rps DeltaServe trained nothing while serving.**
  - The FT job ran entirely after the traffic ended. Co-serving then reduces to "serve first, train later", which
    still keeps service uninterrupted.
  - The FT rate while serving falls from 41 % to 23 % to 0 % of standalone as the rate goes 0.5 → 1 → 2 rps.
  - The cause is the admission gates: the 0.1 s-window RPS throttle (`rps_throttle_close_rps: 5`) and the TTFT SLO.
    Bursts of Poisson arrivals close admission, and the remaining gaps are too short for a 16K-token FT-only step.
  - Same direction as agent-1 §4: the more fragmented the gaps, the less co-serving trains.
- Agent-1 criteria check:
  - co p90 < seq p90 at every rate: **PASS**.
  - co FT ≤ 3× seq FT: 2.4× / **3.5×** / 2.1× — **fails at 1 rps**.

Caveats:
- One run per cell.
- co and seq ran at the same time on disjoint GPUs of one node.
- The bug "FT starts at launch under TP" was fixed before these runs (`v1/worker/gpu_worker.py`). The first attempt,
  which trained from launch, is not reported.

## Caveats

- Single runs, temperature 0. Differences below ~3 points (n=200) are noise.
- Phase C sends test traffic to the **base** model; the adapter is evaluated afterwards (phase D). The loop is one
  round: the exported adapter is not yet hot-swapped in to serve the next round's traffic.
- FiNER has no learned best playbook here, so the end-to-end loop was run on formula only.
- Co-serving cost: p99 TTFT 54 → 725 ms. It comes from the backward child sharing the GPU (no MPS) plus the
  estimator's "eager" regime being under-sampled at launch (7 profiling samples).
- The empty playbook is the literal `(empty)` (`configs/empty_context.txt`), as in the ACE adapter runs. The 0-byte
  `benchmarks/ace/eval/empty_context.txt` the repo used to have was removed.
