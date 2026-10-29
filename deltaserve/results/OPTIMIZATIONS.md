# DeltaServe optimizations for the paper

One entry per optimization: the problem it solves, the mechanism, where it lives, and the measured effect. All
numbers are Qwen3-32B, TP=4 on 4 H200, the RSI mixed traffic of RESULTS.md §7 (ACE formula + finer, TraceLab
session timing, 4 FT rounds of ~230K tokens) unless stated otherwise. Every optimization is a config flag, off by
default; the vLLM-side code is in `DeltaServe-vLLM/dserve-vllm/vllm/`.

## Cumulative ablation (RESULTS.md §7, `rsi_mix_2026-10-28/rsi_mix.md`)

| step | flag(s) | round wait 1 / 2 rps (s) | request p99 1 / 2 rps (s) | TTFT p99 1 / 2 rps (ms) | max-TBT p99 1 / 2 rps (ms) |
|---|---|---|---|---|---|
| co-serving baseline (incl. O3 fix) | — | 302 / 191 | 11.9 / 13.9 | 605 / 967 | 1081 / 1177 |
| + O1 prefill-gated pause + decode-stall budget | `pause_min_prefill_tokens`, `slo.ft_decode_stall_budget` | 176 / 166 | 14.8 / 19.6 | 1959 / 2446 | 1190 / 1185 |
| + looser SLO (TTFT 1 s, max TBT 0.3 s) | SLO only, not an optimization | 134 / 131 | 15.7 / 19.5 | 1091 / 1354 | 1174 / 1262 |
| + O4 fast backward | `backward_tf32`, `backward_attn_sdpa` | **74 / 70.5** | 13.3 / 17.5 | 819 / 1114 | 1079 / 1517 |
| + O2 slow-step guard | `pause_on_slow_step_ms: 80` | 88 / 82 | **9.5 / 10.8** | **472 / 461** | **294 / 364** |
| reference: stop serving to train (`seq`) | — | 70 / 72 | 70.9 / 73.6 | 428 / 605 | 170 / 200 |
| reference: vLLM + torchtune, same GPUs | — | 112 / 121 | 23.5 / 29.9 | 1618 / 1906 | 1191 / 1191 |

Single runs. Noise from one repeat (`cofastb`, 2 rps): round wait ±0.1 s, request p99 ±4 %, max-TBT p99 ±25 %.
The table is cumulative; a leave-one-out ablation has not been run yet.

## O1. Prefill-gated backward pause + decode-stall budget (scheduling)

- **Problem.** Pausing the backward for every inference step starves training (agent traffic is mostly short
  decodes, so there is almost always a step in flight); never pausing it lets long prefills time-slice against the
  backward and blows TTFT. Separately, FT samples riding decode steps lengthen every token gap of every running
  request.
- **Mechanism.** (a) `finetune.pause_min_prefill_tokens`: the backward child yields the GPU only to prefills of at
  least N tokens (4096 here); short steps share the GPU. (b) `slo.ft_decode_stall_budget` /
  `slo.ft_stall_window_s`: a sliding-window budget on the extra time FT adds to decode-carrying steps (here at most
  0.5 s per 2 s), i.e. an average-TBT gate next to the per-step max-TBT gate.
- **Code.** `deltaserve/ft_scheduler.py`, `deltaserve/coordinator.py`, `config/finetune.py`.
- **Effect.** Round wait 302 -> 176 s (1 rps), 191 -> 166 s (2 rps). It moves the cost to the TTFT tail, which
  O2 later takes back.

## O2. Slow-step guard: completion-based feedback on top of the predictor (scheduling)

- **Problem.** With the backward running, decode steps take 20 ms at the median but ~0.5 % stall 0.3-1.2 s while
  the step-time estimator predicts 9 ms, and dozens of requests share each stall. The cause is GPU time-slicing
  between the inference process and the backward process, which no feature of the step predicts; MPS would
  partition the GPU but returned wrong bf16 GEMMs (with SM caps) or deadlocked (without) on this stack.
- **Mechanism.** A watchdog thread per TP rank records a CUDA event for every inference step issued while a backward
  is outstanding. If the oldest unfinished one is older than `finetune.pause_on_slow_step_ms` (80 ms), it pauses the
  backward; when no such step is in flight it releases it, unless another pause owner (a big prefill) still holds
  it. The pause goes through the existing rank-symmetric boundary agreement, so all four ranks stop at the same
  layer boundary and no NCCL collective is left waiting on a paused peer.
- **Code.** `deltaserve/slow_step_guard.py` (new), hooks in `v1/worker/gpu_model_runner.py`, `coord.guard_hold` in
  `deltaserve/coordinator.py`; test `tests/test_slow_step_guard.py` (12/12).
- **Effect (§7.1).** Decode steps > 0.3 s: 45 -> 0 (max 1.16 -> 0.30 s). Max-TBT p99 1079 -> 294 ms (1 rps),
  1517 -> 364 ms (2 rps); request p99 -28 / -39 %; TTFT p99 -42 / -59 %. Cost: rounds 16-19 % longer, still 22-33 %
  faster than vLLM + torchtune.
- **Why it matters for the paper.** Predictive admission can only control what it can see; the guard is the
  reactive complement for interference the model cannot predict, and it needs no MPS.

## O3. Mixed inference+FT batches in CUDA graphs, saving only the step's FT rows (forward)

- **Problem.** FT samples ride real inference batches, and their activations must be copied to the backward's
  buffers. With the save inside a CUDA graph, the op copied the whole 16K-row buffer 448 times per FT-carrying step
  (~150 ms fixed), so no FT sample could ride an inference step under a 0.3 s TBT SLO.
- **Mechanism.** `torch.ops.vllm.dserve_save_rows`: a graph-capturable gather through persistent index tensors
  whose contents change per step, writing only the step's FT rows into the persistent IPC buffer; mixed batches
  replay a second piecewise graph set keyed on "has FT".
- **Code.** `deltaserve/ft_save_op.py`, `deltaserve/accumulate.py`; tests `tests/test_ft_save_op.py`.
- **Effect.** Removes the fixed ~150 ms per FT-carrying step; without it FT could not ride inference steps at all. Earlier on Qwen3-14B TP=2 (branch
  `mixed-fwd-cuda-graph`): mixed-step FT cost 14 -> 4-8 ms, FT throughput +8-39 % at 100 % TTFT satisfaction.
  The 16K-buffer copy itself was a bug; the paper point is the graph-capturable save, not the fix.

## O4. Fast backward: TF32 for the fp32 matmuls + fused SDPA attention backward (backward)

- **Problem.** The precision contract keeps the LM head and attention scores in fp32. Profiling one TP=4 rank
  (`DeltaServe-vLLM/eval-tp/profile_backward.py`) showed those fp32 matmuls never used tensor cores and took 78 % of
  a backward cycle.
- **Mechanism.** `finetune.backward_tf32`: TF32 tensor cores for the fp32 head/attention matmuls (fp32 storage and
  accumulation kept). `finetune.backward_attn_sdpa`: the attention backward through PyTorch's fused SDPA kernel
  instead of the per-sample hand-written core.
- **Code.** `deltaserve/bwd_services/common/{attention,trainer}.py`, `config/finetune.py`.
- **Effect.** 64-layer cycle on one rank 4.66 -> 1.45 s (3.2x); standalone training 3.4K -> 5.8K tok/s (torchtune
  on the same GPUs: 5.4K); round wait 134 -> 74 s. Training unchanged: mean loss 0.2709 vs 0.2710 (`seqfast` vs
  `seq`), gradient cosine to strict fp32 >= 0.99993.

## O5. Long-context FT: chunked attention + streaming LM head (backward)

- **Problem.** Agentic trajectories are long (ACE finer ~1.8K tokens, WebArena 5-16K); the original backward was
  sized for 256-token samples.
- **Mechanism.** `backward_batch_tokens` and chunked attention bound the activation working set; the LM head streams
  vocab chunks instead of materializing `[tokens, vocab]` in fp32; YaRN RoPE in the backward follows the served
  config.
- **Code.** `deltaserve/bwd_services/common/{attention,head,ops}.py`; tests `tests/test_long_context_backward.py`
  (26 cases), `tests/test_rope_yarn.py`.
- **Effect.** Samples up to 16K tokens train on Qwen3-32B TP=4 (README "long-context FT"). No isolated throughput
  number yet.

## O6. Online data path for RSI (system, not performance)

- `POST /ft/ingest` adds training rows to a running server; chat SFT renders rows token-for-token like the offline
  trainer with assistant-only loss (`chat_loss_on`, `chat_thinking`); PEFT export (`adapter_save_dir`) loads into
  stock vLLM.
- **Code.** `entrypoints/serve/finetune/api_router.py`, `deltaserve/chat_sft.py`, `deltaserve/finetuning_store.py`;
  tests `tests/test_ft_ingest.py`, `tests/test_chat_sft.py`, `tests/test_adapter_export.py`.
- **Effect.** End-to-end on one server: ACE formula 68.0 -> 75.0 (p = 0.009), repro 66.5 -> 74.5 (RESULTS.md §1).

## Inherited from earlier DeltaServe-vLLM work (cite as system design, not new here)

Saved activations so the per-layer forward recompute is RMSNorm only (`save_attn_qkv`, `save_attn_ctx`,
`save_resid_mid`); backward CUDA graphs (3 regions per layer); TP backward-per-rank with bucketed factor grads
(240 -> 86 collectives per cycle on Qwen3-14B); the rank-symmetric, run-ahead-bounded pause; the 6-parameter
step-time estimator. See `DeltaServe-vLLM/CLAUDE.md` and `INTEGRATION_PROGRESS.md`.

## Not paper contributions

- LoRA-slot fix (vLLM's warm-up dropped the pinned training adapter, so training published into the inference
  slot) and the transformers-5 `rope_theta` fix: correctness bugs. Worth one sentence: without them training had no
  effect.
- The looser SLO: a parameter choice.

## Known gap

On WebArena (6 agents decoding long thinking continuously, FT samples ~7K tokens) training starves: rounds take
800-1070 s vs ~150 s standalone. No idle steps exist, each sample adds 0.4-0.8 s to the step it rides so the stall
budget admits ~2 per window, and the backward cycle stretches 2.6 -> 4.6 s (p50) under constant decode. Either a
limitation, or a further optimization (split long samples across several inference steps).

## To do before submission

- Leave-one-out ablation (each of O1, O2, O4 off with the rest on), 3 repeats per cell.
- Repeats of the §7 / §8 headline cells.
- An isolated throughput number for O5 (tokens/s vs sample length).
