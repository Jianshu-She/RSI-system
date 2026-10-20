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

## Caveats

- Single runs, temperature 0. Differences below ~3 points (n=200) are noise.
- Phase C sends test traffic to the **base** model; the adapter is evaluated afterwards (phase D). The loop is one
  round: the exported adapter is not yet hot-swapped in to serve the next round's traffic.
- FiNER has no learned best playbook here, so the end-to-end loop was run on formula only.
- Co-serving cost: p99 TTFT 54 → 725 ms. It comes from the backward child sharing the GPU (no MPS) plus the
  estimator's "eager" regime being under-sampled at launch (7 profiling samples).
- The empty playbook is the literal `(empty)` (`configs/empty_context.txt`), as in the ACE adapter runs. The 0-byte
  `benchmarks/ace/eval/empty_context.txt` the repo used to have was removed.
