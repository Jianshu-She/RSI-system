# Experiment log

### lc16k-a — Qwen3-32B TP4 16K-context FT bring-up (pure FT)   (2026-10-12, H200 node, GPUs 0-3 ×4)
- **Reflection/audit:** Qwen3-32B co-serving was validated only at max_model_len 2048 / FT samples ≤512 tokens.
  For 16K I changed five things: FT exempt from `long_prefill_token_threshold`; `backward_batch_tokens`
  decouples batch size from buffer capacity; `backward_cuda_graph_max_tokens` (graphs at 512 rows, long
  backwards eager); query-chunked fp32 attention fwd/bwd; streaming (online-logsumexp) LM head. Offline
  gates: the 4 gradcheck suites (816 cases) pass on both the default and the forced long paths, and
  `tests/test_long_context_backward.py` passes 26/26: chunked attention error ≤1.4e-6, head error 1e-7,
  and a 16K Qwen3-32B-shard attention backward at 125 ms/layer with +3.5 GiB peak. Not yet tested
  end-to-end on the GPU: 16K FT forward through vLLM, buffer memory, TP reduces at 16K rows, and the
  mixed long/short scheduling. This run tests exactly that, with nothing else changed (no inference traffic).
- **Goal/criteria (PASS if all hold):**
  1. Server boots on TP4 with the 16K buffer, with no OOM in workers or backward children.
  2. After POST /start_finetuning, cycles fire for both short (~512-token, graph) and long (1.2K–14.9K, eager) batches.
  3. Every long bucket appears in the bwd log (n up to ~14.9K).
  4. Loss is finite on every cycle.
  5. Per-rank peak memory is under 141 GB.
- **Hypothesis:** with chunked attention and the streaming head, the backward child's temporaries are
  O(few GB) at 16K, so the run is bounded by the ~58 GB/rank activation buffer, which vLLM accounts as
  model memory (allocated at load_model, before KV profiling).
- **Command:** `CUDA_VISIBLE_DEVICES=0,1,2,3 FLASHINFER_WORKSPACE_BASE=/tmp/claude-4536/fi CUDA_HOME=/usr/local/cuda-13.2 PATH=/usr/local/cuda-13.2/bin:$ENV/bin:$PATH HF_HUB_OFFLINE=1 $ENV/bin/python eval-tp/ft_bench_tp.py --config configs/serving_config_finetuning_qwen3_32b_tp4_16k.yaml --rps 0 --duration 300 --kill-stale`
  (ENV = miniconda3/envs/dserve-vllm, rebuilt 2026-10-12: torch 2.11.0+cu130, transformers 4.57.6.)
- **Config:**
  - Engine: max_model_len 20480; max_num_batched_tokens 16384; long_prefill_token_threshold 2048; gpu_mem_util 0.75.
  - Buffer and batching: max_saved_finetuning_tokens 16384; backward_batch_tokens 512; backward_cuda_graph_max_tokens 512.
  - Saves: save_attn_qkv, save_attn_ctx and save_resid_mid all on.
  - Optimizer: lr 3e-4, wd 0.01, epochs 100.
  - Adapter: qwen3-32b-qkv-r8-initial (q/k/v, r=8, α=32).
  - Profiling: profile_on_launch off.
- **Data:** `long_ft_corpus.txt` from `scripts/make_long_ft_corpus.py` (seed 0): 200 alpaca short samples, plus 6 OpenR1-Math problem+trace samples in each length bucket: [1024,2048), [2048,4096), [4096,8192), [8192,12288), [12288,16000). Lengths are Qwen3 tokenizer counts. No holdout (systems bring-up, not a quality eval).
- **Model/loss:** Qwen3-32B base frozen; LoRA q/k/v r=8 trained; shifted CE over all tokens (current backend); AdamW.
- **Predicted:**
  - Boot OK. Per-rank memory ≈ 106 GB (vLLM) + 10–15 GB (child).
  - Short cycles in graph mode at 0.15–0.3 s.
  - A ~15K-token cycle eager at 8–12 s: attention ≈ 64 × 125 ms, plus GEMMs, head and 128 reduces of ~168 MB.
  - Loss finite: short ≈ 1.5–2.5, long math traces lower (≈ 0.5–1.2).
  - **FALSIFIED if** any of: OOM; a 16K FT request is chunked across steps (activations misaligned → loss explodes or a crash); long samples never admitted (stall with samples left); NaN.
- **Actual (attempt 1, 21:55):** weights loaded (15.6 GiB/rank); the 16K activation buffers were allocated and shared on all 4 ranks
  (64 layers, no OOM). Then killed by ft_bench's `--startup-timeout 600` during the first-time torch.compile (fresh env,
  compile range 1..16384 + has_ft set). The `aten::slice ... SymInt` error is the SIGTERM landing inside Inductor fake
  propagation (`signal_handler` frame), not a model bug.
- **Decision:** relaunch unchanged except `--startup-timeout 2400`.
- **Actual (attempt 2, 22:07):** compile hit the cache. OOM at KV allocation: vLLM reported `Available KV cache memory:
  80.84 GiB` at util 0.75, so the ~58 GB/rank activation buffers are NOT in vLLM's budget (my hypothesis that they are
  accounted as model memory was wrong). 0.75×139.8 + 58 > 139.8.
- **Decision:** util 0.75 → 0.42. Budget: vLLM 58.7 GB (15.6 weights + ~37 KV) + buffers 58 + child ~15 ≈ 132 GB.
  Follow-up: subtract the buffers from vLLM's budget automatically.
- **Actual (attempt 3, 22:13):** PASS on the systems criteria. 115 backward cycles in 300 s, every bucket trained, no OOM,
  all 4 ranks bit-identical, every loss finite.
  - Short ~500-token batches: graph mode, ~210 ms.
  - Eager long cycles: 4,344 → 1.63 s; 6,463 → 2.58 s; 9,870 → 4.71 s; 12,698 → 7.53 s; 14,867 → 9.37 s.
  - nvidia-smi showed 142.6 GB/rank once the long cycles ran (mostly the child's caching allocator).
  - **But the model never learned:** epoch-1 losses equal epoch-0 losses to 15 digits.
- **Root cause (bisected on Qwen3-1.7B TP2; profiling-ON learns, profiling-OFF does not):**
  1. vLLM's warmup/graph capture ends in `remove_all_adapters()`, which drops the FT adapter pinned at load_model.
  2. The FT adapter is then lazily re-loaded into whichever slot is free; a debug run showed `index_to_id=[1, 1000]`.
  3. The backward keeps publishing into slot 0, which now holds the INFERENCE adapter. So the FT forwards read the initial
     weights (no learning) and the inference adapter is silently overwritten.
  4. The launch profiling pass hid this because its FT requests re-activate the FT adapter into slot 0 first.
- **Fix:** `Worker._repin_ft_served_lora()` at the end of `compile_or_warm_up_model` re-adds and re-pins the FT adapter, and
  raises if it does not land in the shared slot.
  - Verified on 1.7B TP2 with the 32B flags (profiling off, inference adapter used before FT): `[1000, 1]`, and batch-1
    loss goes 2.63 → 0.98 between epochs.

### lc16k-b — rerun of lc16k with the LoRA re-pin fix   (2026-10-12, H200 GPUs 0-3 ×4)
- **Reflection/audit:** lc16k-a proved the systems path (memory, timing, TP lock-step) but learned nothing because of the slot
  bug. Now fixed and verified on the small model. This run checks that 16K long-context FT actually learns on 32B.
- **Goal/criteria:** the same systems criteria as lc16k-a, PLUS epoch-1 loss < epoch-0 loss on the same batches. The batch
  order is deterministic, so per-batch comparison is valid. Expect clear drops on the short batches, and the long buckets
  lower too.
- **Command:** same as lc16k-a attempt 3 (config `configs/serving_config_finetuning_qwen3_32b_tp4_16k.yaml`, util 0.42,
  `--rps 0 --duration 300 --startup-timeout 2400 --kill-stale`).
- **Predicted:** systems numbers within ±5 % of lc16k-a; epoch-1 short-batch losses ≈ 40–60 % of epoch 0; long buckets
  lower by 10–30 %. **FALSIFIED if** epoch-1 losses are again identical, or re-pin raises.
- **Actual:** **PASS.**
  - Re-pin logged slot 0 on all 4 ranks; no OOM or Traceback.
  - Long-cycle time unchanged (14,867 tokens → 9.37 s).
  - Per-batch epoch-1 / epoch-0 loss ratio: **all 41 batches < 1**. Short ~500-token batches 0.70–0.88; 1.2K–4K 0.62–0.89;
    4K–10K 0.86–0.95; 10K–15K 0.96–0.97.
  - The long samples drop less than predicted. Each is seen once per epoch at lr 3e-4 r=8 (q/k/v only), and the epoch-0
    long losses already include the in-epoch learning.
  - Audit: train-set loss only, measuring "the published weights reach the forward". It is not a generalization or quality
    metric, and there is no holdout. The batch order is deterministic, so the per-batch comparison is like-for-like (the
    63-token flush batch is identical in both epochs).
- **Decision:** 16K long-context co-serving FT on Qwen3-32B TP4 works end to end, and the LoRA re-pin fix is required. Next:
  (1) make the vLLM memory budget subtract the activation buffers automatically; (2) a co-serving run with real 16K
  inference traffic (TTFT and SLO under long prefills; the estimator is linear and may under-predict quadratic attention).

### rsi-ace-1 — RSI-system ACE Finance LoRA SFT on DeltaServe vs the offline trainer   (2026-10-19, fs-mbz-gpu-198 ×8, job 2721371)
- **Reflection/audit:** DeltaServe can now train on RSI-system data. Three gaps are closed and CPU-gated:
  - G1, chat JSONL plus a loss mask: token-for-token equal to both offline trainers on 60 real rows each of
    formula/finer/appworld; masked head == autograd.
  - G2, YaRN in the backward RoPE: bit-equal to vLLM's YaRN cache.
  - G5, PEFT adapter export: TP=2 merge == TP=1.

  None of it has run on GPU yet. The question is whether a DeltaServe-trained adapter reproduces the offline LoRA gain
  under the user's exact eval protocol.

  Known recipe differences vs their adapters (formula_ace_v2 = 75.0, finer_ace_v2 = 68.5; base 69.0 / 61.8):
  - DeltaServe trains attention-only (no MLP LoRA).
  - 16K tokens/step vs ~32K.
  - Constant LR vs cosine with warmup; no dropout; no 5 % dev hold-out.

  To separate "DeltaServe's trainer" from "attention-only", their own trainer is re-run attention-only as a control.
- **Goal/criteria:**
  1. DeltaServe trains both datasets to the end: final adapter exported, loss falling, no NaN or OOM.
  2. Test accuracy (empty context, their `eval_split.py` + stock vLLM 0.31, temp 0) of the DS adapter is within noise
     (~±3 pts, n=200) of their attention-only control.
  3. Both are above base. Test = held out; the adapters train on TRAIN-split traces only (their leak audit).
- **Commands:**
  - GPUs 0–3: `eval-rsi/run_ds_ace.sh formula; eval-rsi/run_ds_ace.sh finer`. This runs
    `eval-tp/launch_deltaserve.py --config configs/serving_config_finetuning_qwen3_32b_tp4_rsi_ace.yaml` with
    data_path → `data_v2/<task>.jsonl`, `--start-finetuning`.
  - GPUs 4 / 5: `verl/python eval-rsi/train_lora_ace_targets.py --data data_v2/<task>.jsonl --out eval-rsi/ctrl/<task>
    --epochs 2 --lr 1e-4 --rank 16 --alpha 32 --dropout 0.05 --tokens_per_step 32768 --seed 42
    --targets q_proj,k_proj,v_proj,o_proj`.
  - GPUs 6–7: the regression gates (TP NCCL graph tests, long-context GPU part, optional-LoRA, adapter export).
- **DS config:**
  - Model: Qwen3-32B-yarn131k (YaRN x4), TP4, max_model_len 20480, util 0.42.
  - Buffer: 16384-token buffer, backward_batch_tokens 16384, graphs at 512 rows (big batches eager), all saves on.
  - Recipe: LoRA q,k,v,o r16 α32 (zero-B init, `adapters/qwen3-32b-qkvo-r16-initial`), lr 1e-4 constant, wd 0,
    2 epochs, chat_loss_on ace / thinking off.
- **Data:** formula_v2 803 rows (534K tokens; their 763 train / 40 dev split); finer_v2 277 rows (707K tokens).
  Test: their test split (formula 200, finer ~200; same as their catalog).
- **Predicted:**
  - DS trains to `final/` in under 30 min per task. Loss starts ~0.35 and falls to ~0.2 (their dev loss 0.34 → 0.18).
  - Attention-only control: formula ~72–75, finer ~66–68.5 (a bit below their full LoRA).
  - DS within ±3 of the control.
  - **FALSIFIED if:** DS loss doesn't fall; or the DS adapter scores ≤ base while the control is above it (that would mean
    a mask/rope/export bug); or the export fails to load in vLLM.
- **Actual:** **PASS** on every criterion.
  - Regression gates on the node: TP NCCL graph 4020/4020, NCCL trainer 222/222, long-context 26/26; the CPU gates
    (optional-LoRA 49/49, export 9/9, chat 17/17, YaRN 8/8) need `CUDA_VISIBLE_DEVICES=""` because with GPUs visible
    the TP helper picks NCCL for CPU tensors.
  - DS training:
    - Formula: 803 rows, 177,105 target tokens; 62 cycles × ~16K tokens in 3 min 49 s; epoch loss means 0.218 → 0.141.
    - FiNER: 94 cycles in 7 min 9 s; epoch means 0.288 → 0.207.
    - Adapter re-pinned in slot 0 on all ranks; final/ exported; it loads in stock vLLM 0.31 `--lora-modules`.
  - Control (their trainer, q,k,v,o only, 42 / ~28 steps): final dev loss finer 0.215, formula 0.205.
  - Test accuracy (their eval_split.py, vLLM 0.31, empty context, temp 0), with paired sign tests on per-item
    correctness from the error lists:

    | | base | ctrl (attn-only, their trainer) | **DS (attn-only, co-serving trainer)** | their full LoRA (catalog) |
    |---|---|---|---|---|
    | Formula (n=200) | 68.0 | 72.0 (+17/−9, p=0.17) | **75.5** (+23/−8 vs base, p=0.011) | 75.0 (p=0.05) |
    | FiNER (n=441 tags) | 61.5 | 61.3 | **71.2** (+80/−15 vs base, p<1e-4; +58/−15 vs ctrl) | 68.5 (p=0.0005) |

  - Base reproduces their catalog: 69.0 / 61.8.
  - FiNER per-item correctness = "no tag wrong", so the sign test can move even when tag accuracy doesn't
    (ctrl: 61.3 vs 61.5, yet +44/−22).
- **Audit:** held-out TEST split; adapters trained on TRAIN-derived rows only (their data, their leak audit). Same eval
  harness and model config as their catalog. Single runs (temp 0).
- **Caveat / not yet separated:** DS beats the attention-only control, but its recipe takes 2.2× more optimizer steps (16K
  vs 32K tokens/step) at a CONSTANT lr 1e-4 (theirs: cosine with warmup), no dropout, and trains the 5 % dev rows too. The
  gap vs ctrl is therefore most likely the schedule, not the system. Claim only that a DS-trained adapter reproduces (and
  here matches or exceeds) their offline LoRA gains under their eval.
- **Decision:** the RSI-system ACE Finance workload runs on DeltaServe end to end. Next:
  1. A matched-schedule control (their trainer at 16K tokens/step, constant LR) to attribute the gap.
  2. Co-serving: their eval traffic as inference while training in the gaps.
  3. AppWorld (28K samples; G3) and MLP LoRA (G4).

### rsi-e2e-1 — end-to-end RSI on ONE DeltaServe server: serve → traces → ingest → train in the gaps → eval   (2026-10-19, fs-mbz-gpu-198, job 2721371)
- **Reflection/audit:** rsi-ace-1 showed that DeltaServe trains RSI-system data to their offline LoRA level, but it ran
  training ONLY, on data produced offline. The RSI claim needs the loop closed on one live server:
  - the server's own traffic produces the traces;
  - the traces become training data;
  - the LoRA trains in the gaps of continuing inference, without hurting it.

  New for this run:
  - `POST /ft/ingest`: runtime append to the FT store (EngineCore utility → `FinetuneScheduler.deltaserve_ingest`
    → `FinetuningStore.add_rows`; `tests/test_ft_ingest.py` 8/8).
  - `eval-rsi/rsi_traffic.py`: ACE's Generator request + scorer, streamed, with timing.
  - `eval-rsi/traces_to_sft.py`: their build_sft_data + make_data_v2 recipe applied to served traces.
  - The standard pause policy (prefill-only), not the strict 0-token throttle of the baselines YAML: that one would
    pause the backward on every decode step.
- **Goal/criteria:**
  1. The loop completes on one server: traces → ≥ 300 correct rows → ingested → final/ exported while the phase-C
     traffic runs.
  2. The final adapter beats base on the formula test split, with their protocol (stock vLLM 0.31, empty context;
     base 68.0 on this node).
  3. Inference under co-serving: TTFT p50 within 1.3× of phase A on the same traffic, and ≥ 90 % of requests under
     the 0.3 s TTFT SLO.
- **Command:** `CUDA_VISIBLE_DEVICES=0,1,2,3 eval-rsi/run_rsi_e2e.sh`.
  - Config: `configs/serving_config_finetuning_qwen3_32b_tp4_rsi_e2e.yaml` — rsi_ace + data_path null,
    profile_on_launch true, max_model_len 40960, fwd_token_throttle / pause_prefill_always off.
  - Traffic:
    - A / C: formula test (200), empty playbook, Poisson 0.5 rps.
    - B: formula train (500) with the learned best_playbook (29,747 tokens, prefix-cached), 16 concurrent.
  - Training recipe as rsi-ace-1: q,k,v,o r16 α32, lr 1e-4, 2 epochs over the ingested rows, 16K tokens/backward.
- **Predicted:**
  - B: ~80 % correct → ~400 rows.
  - C: FT completes in 10–30 min, contended.
  - Adapter: 72–76 on test, below rsi-ace-1's 75.5 because there are fewer rows (≈ 400 vs 803).
  - A vs C: TTFT p50 ×1.0–1.2, p99 higher; SLO satisfaction ≥ 95 %.
  - **FALSIFIED if** FT starves (no final/ within 60 min of traffic), the adapter is ≤ base, or SLO satisfaction < 90 %.
- **Actual:**
  - Server up at 14:28. Launch profiling fit 152 samples; the eager regime got only 7 (rmse n/a).
  - A (inference only): base 67.0 % (134/200; vLLM 0.31 on this node: 68.0).
  - B: 427/500 correct (85.4 %) in 180 s; the 30K-token playbook prompts were prefix-cached.
  - Ingest: 427 rows / 263,053 tokens accepted live (`/ft/ingest`), epoch 0/2.
  - C (co-serving): 34 backward cycles (526K tokens) in **4 min 35 s** while the test traffic ran; loss epoch means
    0.291 → 0.155; final/ exported at cycle 34.
  - D: **75.0 % (150/200)** vs base 68.0: +20/−6, sign-test p = 0.0094. For comparison: their offline full LoRA 75.0;
    rsi-ace-1 DS on their data_v2 75.5; the attention-only control 72.0.

  Latency on the same traffic and server:

  | | A: inference only (n=200) | C: during training (n=116) |
  |---|---|---|
  | TTFT p50 / p90 / p99 / max | 38 / 45 / 54 / 770 ms | 84 / 176 / 725 / 1077 ms |
  | TTFT < 0.3 s SLO | 99.5 % | 96.6 % |
  | E2E p50 / p90 / p99 | 1.70 / 2.54 / 4.01 s | 2.30 / 5.24 / 6.66 s |
  | errors | 0 | 0 |

  Criteria:
  1. Loop completes — PASS.
  2. Beats base — PASS.
  3. SLO ≥ 90 % — PASS (96.6). TTFT p50 ≤ 1.3× — **FAIL**: 2.2×, though 84 ms is still 3.5× under the SLO.
- **Audit:**
  - Test split held out. Training rows come only from TRAIN-split traces the server itself produced (correct by ACE's
    own scorer). The eval is their exact protocol, on a different engine (stock vLLM 0.31).
  - C's accuracy line (69.8 %) is the BASE model on 116 requests (the traffic targets the base, not the adapter) —
    not an adapter number.
  - Single run, temp 0.
- **Decision:**
  - The RSI loop runs end to end on one DeltaServe server and reproduces the offline LoRA gain from its own served
    traces.
  - The co-serving latency cost is real: p99 TTFT 54 → 725 ms. It comes from the backward sharing the GPU (no MPS)
    plus the eager regime being under-fitted at launch.
  - Next levers: the matched-schedule control; more profiling coverage of the eager/FT shapes; an A/B of
    `pause_prefill_always` / fwd_token_throttle at a non-zero threshold for p99.

### rsi-e2e-2 — reproduction of rsi-e2e-1 with the packaged RSI-system scripts   (2026-10-19, fs-mbz-gpu-198, job 2721371)
- **Why:** before publishing `RSI-system/deltaserve/`, prove its scripts (env.sh + configs/coserve_e2e.yaml template +
  scripts/run_e2e.sh) reproduce rsi-e2e-1 as written in its README. The DeltaServe code is unchanged: a fresh clone at
  eea2a4a + the shipped patch is byte-identical to the validated tree.
- **Command:** `bash deltaserve/scripts/run_e2e.sh verify` (env.local.sh points at this cluster's ACE dirs).
- **Predicted:**
  - Same phases complete.
  - Adapter test accuracy within ±3 of 75.0 (temp 0; the traces may differ slightly between runs).
  - Co-serving SLO ≥ 90 %.
  - **FAIL if** any script step breaks, or the result deviates beyond noise.
- **Actual:** **PASS.** Every step ran as documented.
  - A: base 67.5 %, TTFT p50/p99 36/51 ms.
  - B: 427/500 correct.
  - Ingest: 427 rows / 262,947 tokens.
  - C: 34 cycles, 4 min 37 s; epoch loss 0.290 → 0.155; TTFT p50/p99 83/217 ms; SLO 99.1 %.
  - D: **74.5 %** vs base 66.5 % on the same eval engine (+24/−8, p = 0.007).
  - Audit: the co-serving tail varies run to run (p99 725 → 217 ms). The base moves ~1.5 points between evals at temp 0.
- **Decision:** publish `RSI-system/deltaserve/` with both runs in results/RESULTS.md.

### paper-1 — paper metrics on the RSI loop: LoRA vs skills-in-context (accuracy, context, KV, TTFT/TBT, GPU util, leakage, general ability)   (2026-10-19, fs-mbz-gpu-198 ×8, job 2721371)
- **Reflection / audit:**
  - Done so far: rsi-e2e-1/2 showed the loop works. Self-generated traces → LoRA trained in the gaps → formula
    68 → 75 / 66.5 → 74.5 (p < 0.01). Co-serving SLO was 96.6 / 99.1 %.
  - Missing for the paper (the user's list):
    1. Is LoRA + skills > skills?
    2. How much context, KV cache and TTFT/TBT do skills cost versus a LoRA?
    3. Does co-serving raise GPU utilization?
    4. Can skills / user data leak out of the prompt, and does the LoRA remove that?
    5. Does distilled experience (or a pile of skills in context) hurt general ability?
  - None of these was measured. The earlier runs evaluated only {base, LoRA} with an EMPTY playbook. They had no
    TBT, no KV numbers and no GPU utilization.
  - Why now: same loop, same code, plus instruments. One run on one node gives every number from one adapter.
- **Goal / criteria (what each number answers):**
  - H1, "LoRA+skills > skills": formula test, 2×2 {base, rsi} × {empty, skills = the ACE best playbook}. Paired sign
    test rsi_skills vs base_skills.
  - H2, context / KV / latency (stock vLLM 0.31, TP=4, GPUs 0-3, the same formula test traffic at 1 and 4 rps):
    - skills = base + playbook in the prompt; lora = adapter + "(empty)".
    - Per request: prompt tokens, TTFT, TBT (inter-chunk gaps), E2E.
    - Per second: vLLM KV-cache usage, running/waiting, prefix-cache hits.
    - Also the KV pool size → max concurrency.
  - H3, GPU utilization: DCGM SM_ACTIVE (1002) at 1 Hz on GPUs 0-3, DeltaServe phase A (inference only) vs phase C
    (co-serving), same traffic.
  - H4, leakage: 12 prompt-extraction attacks in the deployment's own prompt. Measure the fraction of the playbook's
    282 bullets leaked by id, and verbatim (LCS ≥ min(60 chars, 50 %)). Compare skills deployment vs LoRA deployment
    vs base (control).
  - H5, general ability: on {base, rsi} × {empty, skills}, score:
    - FiNER test: the other finance task; the LoRA never saw it, and the formula playbook is "irrelevant skills" there.
    - GSM8K test (1319).
    - MMLU (1000 of all/test, seed 0).
- **Command:** `srun --jobid 2721371 --overlap bash deltaserve/scripts/paper_run.sh paper1` (RSI-system).
  - It runs `run_e2e.sh` with SKIP_EVAL=1 on GPUs 0-3. The DeltaServe config is unchanged from rsi-e2e-2
    (configs/coserve_e2e.yaml: LoRA q,k,v,o r16 α32, lr 1e-4 const, 2 epochs, 16K tok/step, ace loss spans,
    0.5 rps traffic).
  - `eval_suite.sh`: base on GPUs 4,5 during the e2e; rsi on 6,7 afterwards. Uses stock vLLM 0.31, TP=1, temp 0,
    prefix cache on.
  - `serving_cost.sh` on GPUs 0-3 after the e2e. RATES = "1 4".
  - `gpu_monitor.sh` (dcgmi dmon -e 155,203,252,1002,1003,1004) runs throughout.
- **Data:**
  - Formula train 500 (traffic B → SFT rows), formula test 200, FiNER test.
  - GSM8K test 1319, MMLU 1000.
  - Playbook = ACE formula offline+GT best_playbook.txt (282 bullets, ~29.7K tokens).
  - Leak check: the adapter is trained only on formula TRAIN traces; every accuracy number is on test splits / other
    benchmarks.
- **Predicted (committed before the run):**
  - H1:
    - base_empty ≈ 67, rsi_empty ≈ 74-75.
    - base_skills ≈ 74-78; rsi_skills ≥ base_skills by +1-3. That gap is inside noise for n=200, so H1 likely
      needs p < 0.05 to be claimed.
    - FAIL if rsi_skills < base_skills − 3.
  - H2:
    - Prompt tokens ≈ 30.1K vs ≈ 0.4K (~75×).
    - With a warm prefix cache: skills TTFT p50 ≈ 2-4× lora's; TBT 1.2-2× (attention over 30K context per decode
      token).
    - KV-usage peak higher by ≥ 12 % of the pool. That is ≈ 30K of 1.0M tokens at TP=4, shared through the prefix
      cache. Without a shared prefix it would be 30K tokens per request.
    - FAIL (for the paper claim) if skills TBT ≤ 1.1× lora.
  - H3: phase C SM_ACTIVE ≥ 2× phase A (A ≈ 0.1-0.25 at 0.5 rps). FAIL if < 1.3×.
  - H4:
    - Skills: ids ≈ 100 %, verbatim ≈ 40 % (smoke: 282/282, 118/282).
    - LoRA and base: ids ≤ 2 %, verbatim ≤ 2 %.
    - FAIL if LoRA verbatim > 10 %.
  - H5:
    - GSM8K / MMLU: rsi within ±2 of base.
    - Skills in context within ±2 on GSM8K/MMLU, but −1 to −5 on FiNER.
    - rsi on FiNER: −3 to +2.
    - FAIL ("distillation harms") if rsi drops > 3 on GSM8K or MMLU with p < 0.05.
- **Actual** (run dir `RSI-system/deltaserve/work/runs/paper1`, report `paper_metrics.md`):
  - Loop: B 427/500 correct; 427 rows; C 34 cycles, 526K tokens in 281 s (1873 FT tok/s); loss 0.289 → 0.154.
    Same as rsi-e2e-1/2.
  - **H1 — FAIL as predicted.** Formula (n=200), base / base+skills / LoRA / LoRA+skills = 68.5 / 86.0 / 75.0 / 85.5.
    - rsi_skills vs base_skills: −0.5 (+2/−3, p=1.0).
    - The playbook alone is far stronger than I predicted (86 vs my 74-78).
    - The LoRA recovers 6.5 of the playbook's 17.5 points (37 %) with zero context (p=0.029 vs base). It adds nothing
      on top of the playbook, which is expected in hindsight: it was distilled from traces generated WITH that same
      playbook, so it carries a subset of the same information.
    - "LoRA + skills > skills" needs new knowledge in the LoRA (e.g. traces from user feedback that the playbook does
      not contain), not a distillation of the playbook itself.
  - **H3 — PASS.** DeltaServe GPUs 0-3, same 0.5-rps traffic:
    - SM_ACTIVE 19.2 % (A) → 56.0 % (C), 2.9×; nvidia-smi util 59 → 96 %; power 274 → 493 W/GPU.
    - TTFT p50/p99 35/51 → 81/268 ms; SLO<0.3 s 99.5 → 99.1 %; TBT p50 8.7 → 8.3 ms, p99 9.8 → 25.8 ms.
  - **H4 — PASS.**
    - Skills deployment: ids 282/282 (100 %), verbatim 149/282 (52.8 %).
    - Base control: 1/282, 0/282.
    - LoRA: 1/282 ids, 7/282 verbatim (2.5 %). All 7 are one textbook sentence (the Sharpe-ratio definition "(R_p −
      R_f)/σ_p, where R_p is the portfolio return, R_f is the risk-free rate…") repeated in 7 bullets, elicited by
      the attack that asks how to compute Sharpe. That is public knowledge, not playbook content.
    - The single id hit (calc-00001) also appears for the base model: a guessable id.
  - **H5 — PASS (no harm from distillation; harm from irrelevant skills in context).**
    - LoRA vs base: FiNER +0.1 (p=0.27), GSM8K −0.5 (p=0.28), MMLU −0.2 (p=0.77). All noise.
    - Formula skills in context on FiNER: −4.6 (+19/−49, **p=0.0004**) for base and −5.8 for the LoRA. On GSM8K
      −0.7 (p=0.23), on MMLU +0.5 (n.s.).
    - Skills in context also cost eval time: GSM8K 649 s vs 223 s, FiNER 566 s vs 181 s on one GPU.
  - H2: see paper-1b. The 4-rps pass here repeated prompts and is invalid; the LoRA rows are unmerged kernels.
  - Audit:
    - Every accuracy is on test splits or other benchmarks. The adapter was trained only on formula-train traces.
    - Temp 0, one run each.
    - Sign tests are paired on identical items.
    - The FiNER drop with skills is the strongest "skills harm" evidence; GSM8K/MMLU are not affected.

### paper-1b — serving-cost follow-up: fresh prompts at 4 rps, merged LoRA, no prefix sharing   (2026-10-20, fs-mbz-gpu-198 GPUs 0-3, job 2721371)
- **Why (audit of paper-1 §2):**
  - (a) The 4-rps pass repeated the 200 test prompts already sent at 1 rps. Prefix-cache hit was 98 % vs 54 %, so whole
    prompts hit the cache and the 4-rps TTFT is not valid.
  - (b) The "LoRA" rows run stock vLLM's unmerged multi-LoRA kernels: +67 ms TTFT and +2.6 ms TBT vs base at 1 rps.
    That cost belongs to the serving stack, not to the idea; a single customized model is served merged.
  - (c) With one shared playbook and prefix caching, the skills' KV is shared. Per-user skills are not shared, and that
    case was never measured.
- **Command:** `srun --jobid 2721371 --overlap bash deltaserve/scripts/paper_run_cost2.sh paper1`. The steps:
  1. `merge_lora.py`: W += (32/16)·B·A on q,k,v,o → `runs/paper1/merged`.
  2. `cost_fresh`: P / L / M servers, every condition at 4 rps on the formula TRAIN split (500 new prompts).
  3. `cost`: adds merged / merged_skills at 1 rps on the test split, next to paper-1's 1-rps rows.
  4. `cost_nopc`: base and skills, prefix caching OFF, 1 rps.
- **Predicted:**
  - Merged is within ±10 % of base on TTFT and TBT, and its formula accuracy on the traffic is within ±3 of lora@1's
    73.5 (sanity check of the merge).
  - Fresh 4 rps: skills TTFT p50 ≥ 2× merged.
  - No-prefix-cache skills: TTFT p50 ≥ 1 s (a 30K-token prefill per request); peak KV usage ≥ 5× its prefix-cached run.
  - FAIL / merge bug if merged accuracy is < 70 on the 200 test items.
- **Actual — all PASS** (`runs/paper1/paper_metrics.md` §2, three tables):
  - **Merge sanity:**
    - merged@1 test 74.5 % (lora@1 73.5, the eval 75.0).
    - On the same 500 train prompts, merged 76.2 vs lora 75.8. The train-split numbers are in-sample for the
      adapter and are used for latency only.
  - **Merged = base latency:**
    - 1 rps: TTFT p50/p99 41/66 vs base 43/62 ms; TBT p50 6.7 vs 7.0 ms.
    - Fresh 4 rps: 42/85 vs 42/68 ms; TBT 6.9 vs 6.8 ms.
    - Unmerged vLLM multi-LoRA costs +67 ms TTFT and +2.6 ms TBT. That is a serving-stack cost the merged model
      does not pay.
  - **Fresh 4 rps, shared prefix cached:**
    - Skills vs merged: TTFT p50 119 vs 42 ms (2.8×), p99 861 vs 85 ms (10×).
    - TBT p50 13.4 vs 6.9 ms (1.9×), p99 46.8 vs 24.7 ms.
    - E2E p50 4.06 vs 1.77 s.
    - KV usage mean 2.2 vs 0.1 %; running peak 33 vs 15.
    - Prompt tokens 30,137 vs 391 (77×).
  - **Prefix cache OFF (per-user skills), 1 rps:**
    - Skills: TTFT p50 **34.0 s**, p99 99 s; TBT p50 318 ms; KV usage mean 76 %, peak 99.6 %; 59 running (KV-bound:
      1.78M-token pool / 30.5K tokens = 58), 63 waiting.
    - Base: 38 ms / 6.7 ms. The server cannot sustain 1 rps with per-request 30K-token skills (7.36 GiB KV each).
  - Skills + LoRA (merged_skills@4) is the slowest cached condition: the outputs are longer (441 vs 337 tokens).
- **Decision:**
  - The paper's cost claim should compare skills against the MERGED customized model.
  - Report both the shared-cache (best case for skills) and the no-sharing (per-user skills) settings.
  - Never report the paper-1 4-rps rows (repeated prompts).

### agent-1 — sequential (inference, then training) vs co-serving, TraceLab coding-agent workload   (2026-10-20, fs-mbz-gpu-198 ×8 H200)
- **Reflection / audit:**
  - Done so far: paper-1/1b measured LoRA vs skills (accuracy, context/KV, TTFT/TBT, GPU utilization, leakage).
  - The co-serving phase C so far used one simple Poisson ACE workload (0.5 rps, short prompts). It said nothing
    about agentic traffic, or about what the alternative costs in wall-clock time.
  - The user asks for the comparison against running the two jobs disaggregated in time (serve, then train), with
    requests following TraceLab (arXiv:2606.30560), and for task completion time.
  - Why TraceLab: its traffic is long-context, short-output, with heavy-tailed tool latencies and human-paced gaps
    (human think time is 92 % of session wall-clock). Those gaps are exactly where co-serving trains.
  - Honest risks:
    1. The co-serving server's KV pool is 562K tokens vs 1.78M for a plain deployment (the FT buffers take the
       memory). Agent contexts are long, so prefix-cache evictions can make co-serving's agent latency worse. This
       is a real cost and will be reported.
    2. The FT job is small (sft_from_traces, 427 rows, 263K tokens/epoch).
- **Goal/criteria:**
  - Agent task = one user request plus its tool loop. Report its completion time (p50/p90/p99), and per-round
    TTFT/TBT.
  - Report the FT job's completion time (/start_finetuning → final/ adapter).
  - Report the makespan of both jobs:
    - Sequential = inf makespan + ft time.
    - FT-first variant: every task delayed by the ft time.
    - Co = max(replay end, ft end).
  - PASS (the co-serving claim holds) if:
    - co makespan ≤ 0.8 × sequential makespan, AND
    - co agent-task p50 completion is within +10 % of inference-only.
- **Workload** (`deltaserve/scripts/agent_trace.py`, TraceLab v0.0.1 DuckDB, Claude rounds, seed 0):
  - 48 sessions from a pool of 2,510, Poisson arrivals at 0.05/s (last at 995 s).
  - 900 rounds, 83 tasks; each session cut at its first 4 tasks / 80 rounds.
  - Real appends ×1/8 (p50/p90 141/918), real outputs (p50/p90 184/1115).
  - Real tool gaps (p50 0.09 s, p99 182 s) and human gaps (p50 129 s), each capped at 300 s.
  - Context: shared 2K system block plus the session history, with output re-sent as input. Compaction to
    system + 4K when the context reaches 36,864.
  - Replay (`agent_replay.py`) is closed-loop per session, uses synthetic token ids, and forces output length.
- **FT job:** `runs/paper1/sft_from_traces.jsonl` (427 correct ACE trajectories), 4 epochs ≈ 1.05M tokens. Same
  recipe as train_only.yaml: q,k,v,o r16, lr 1e-4, 16K batch tokens.
- **Commands** (all three on node 198, Qwen3-32B YaRN x4, TP=4, DeltaServe fork):
  - `WORKLOAD=work/agent/agent_workload.json DATA=.../sft_from_traces.jsonl EPOCHS=4 agent_run.sh co 0,1,2,3 8000 work/runs/agent1/co`
  - `... agent_run.sh inf 4,5,6,7 8100 work/runs/agent1/inf`
    - Finetuning off, gpu_memory_utilization 0.90. Runs concurrently with co on the other 4 GPUs; host-side
      interference is possible.
  - Then `... agent_run.sh ft 4,5,6,7 8100 work/runs/agent1/ft`.
- **Predicted:**
  - Inf replay makespan ≈ 40 min (≈ 33 min of trace waits plus LLM time).
  - FT alone ≈ 9–10 min (~1.9K tok/s).
  - Co: FT finishes within the replay window at ≈ 15–25 min (it trains in the gaps). Makespan ≈ replay ≈ 41 min vs
    sequential ≈ 50 min (~0.82×).
  - Co agent-task p50 completion +5–15 % vs inf (TTFT p50 ~2×, smaller KV).
  - FALSIFIED if co FT does not finish before the replay ends, or co task p50 is > +25 %.
- **Deviation (11:20):**
  - The `inf` run on the DeltaServe fork with finetuning off (gpu_memory_utilization 0.90) died 50 s in. The
    error was a CUDA illegal memory access in flash-attn (`hopper/flash_fwd_launch_template.h:199`, a 7,180-token
    prefill); 169 of its 184 rounds errored. The log is kept in `runs/agent1/inf_fork_crashed/`.
  - It was killed. The inference-only baseline is now stock vLLM 0.31 (the paper-1 serving server), run on GPUs
    4-7 after `ft` via `agent_run2.sh` (since renamed `agent_run.sh`). It keeps the same max_model_len 40960, step budget 16384 and chunking 2048.
  - Consequence: the inf baseline uses a different (newer) engine than co. Its latency is, if anything, better
    than the fork's would be, which makes it conservative for the co-serving claim.
  - Early co observation: FT-only forwards (15,936 tokens ≈ 2–3 s) are repeatedly aborted at layer 13–18 by agent
    arrivals (tier C). In the first 4 min, FT was at 5 cycles / 80K tokens.
- Also: the first `ft` launch died with EADDRINUSE on the backward NCCL port 29677 (held by `co`), and was rerun
  with `DSERVE_BACKWARD_NCCL_PORT=29777`.
- **Actual** (`RSI-system/deltaserve/results/agent1_2026-10-20/agent_metrics.md`; RESULTS.md §4) — **FAIL** on both
  criteria:
  - **Makespan:** co 2,161 s vs sequential 2,386 s = **0.906×** (criterion ≤ 0.8×). The inf replay took 2,130 s
    (predicted ~40 min, actual 35.5).
  - **FT job:**
    - Alone: **255 s** at 4,122 tok/s (predicted 9–10 min).
    - Co-served: 1,612 s at 653 tok/s (16 % of alone). It finished 9 min before the replay ended, as predicted.
    - Loss is identical, 0.523 → 0.053.
  - **Agent task completion** (user message → last LLM round of its tool loop):
    - p50 / p90 / p99: inf 38.9 / 272 / 905 s, co 46.6 / 306 / 936 s.
    - **+20 % at p50** (criterion ≤ +10 %; not past the +25 % falsification line).
    - LLM time per task: +19 %.
  - **Per-round latency:** TTFT p50 152 → 179 ms; TBT 8 → 10 ms.
  - **Utilization:** SM_ACTIVE 33.4 / 36.4 %. 35 % of the inf wall time had no request in flight.
  - **Aborts:** 43 tier-C aborts of FT-only forwards (≈0.36M token-equivalents discarded). FT stalled 11:28–11:37
    at peak load.
  - Train-then-serve variant: FT done at 255 s, task p90 272 → 354 s.
- **Decision / insight:**
  - Co-serving wins makespan by at most the FT job's standalone time. Here that job is small (4 min) next to 36 min
    of agent traffic, so the win is 9 %.
  - The real gap: on agent traffic the GPU is idle about two-thirds of the time, yet DeltaServe trains at only 16 %
    of its standalone rate. The idle time is fragmented: tool gaps are p50 0.09 s, and a 16K FT forward is
    all-or-nothing and gets aborted.
  - Next, if pursued:
    1. Gap-sized FT forwards: cap FT tokens per FT-only step from the observed inter-arrival gap.
    2. Keep the partial forward on abort instead of discarding it.
    3. A larger FT job (e.g. 10×), so the makespan effect is measurable.
  - Separately: the DeltaServe fork with finetuning off crashes in flash-attn on this workload — open bug.

### rsi-poisson — RSI traffic with Poisson arrivals: sequential (stop serving to train) vs co-serving   (2026-10-20, fs-mbz-gpu-198 ×8 H200)
- **Reflection / audit:**
  - agent-1 replayed TraceLab sessions. The user clarified that the intent was different: take the RSI workload
    and borrow only TraceLab's way of sending requests, a Poisson arrival process.
  - The sequential baseline there was also unrealistic: it served first and then trained, with no traffic during
    training. Under continuous traffic, a single deployment that stops to train makes the requests arriving during
    training wait. That is the cost co-serving removes.
  - So the comparison now runs on one server config with identical engines (DeltaServe, FT enabled), and the only
    difference is whether serving stops for the FT job.
- **Goal/criteria:**
  - Per rate, report request completion time (arrival → done, including any hold) at p50/p90/p99, the FT job time,
    and when everything is done.
  - PASS (the co-serving claim holds) if, at every rate:
    - co p90 request completion < seq p90, AND
    - co FT done ≤ 3× the sequential FT time.
- **Setup:**
  - Traffic: ACE formula train split (500 items, empty playbook) to the base model. Poisson at RATE ∈ {0.5, 1, 2}
    rps, one pass. Temp 0, thinking off, max 4096 tokens.
  - FT job: `runs/paper1/sft_from_traces.jsonl` (427 correct trajectories from the RSI loop) × 2 epochs = 526K
    tokens (the RSI recipe). It starts at t = 120 s.
  - seq: hold new arrivals, drain in-flight requests, POST /start_finetuning, wait for final/, release.
  - co: POST /start_finetuning and keep serving.
  - Server: train_only.yaml with profile_on_launch true, Qwen3-32B YaRN x4, TP=4, KV 562K.
- **Commands:**
  - `RATE=r DATA=... EPOCHS=2 rsi_poisson.sh co 0,1,2,3 8000 work/runs/rsip/co_r<r>` (NCCL 29677)
  - `rsi_poisson.sh seq 4,5,6,7 8100 work/runs/rsip/seq_r<r>` (NCCL 29777)
  - The pairs run in parallel on disjoint GPUs, one rate after another.
- **Predicted:**
  - FT alone ≈ 130 s (agent-1 measured 4.1K tok/s alone).
  - seq: the ~65 / 130 / 260 requests arriving during the FT window (0.5 / 1 / 2 rps) are held for ~65 s on
    average. That shows up at p90 for 1–2 rps and at p99 for 0.5 rps: p90 > 30 s at 1 and 2 rps.
  - co: FT ≈ 280 s at 0.5 rps (paper-1: 281 s). Longer at 2 rps (~400–700 s), and it could stall the way agent-1
    did. Request p50 is +20 % (paper-1: e2e p50 1.72 → 2.06 s) and p90 stays < 10 s.
  - Everything done at ≈ the traffic length (1000 / 500 / 250 s + tail) in both modes. At 2 rps, seq ends later,
    because the held burst is served after FT.
  - FALSIFIED if co p90 ≥ seq p90 at any rate, or co FT does not finish before the traffic ends at 0.5 rps.
- **Deviation (12:36):**
  - The first attempt (`runs/rsip_invalid_ftstart/`) is invalid: FT trained from server launch in both modes, long
    before t = 120 s.
  - Cause, a bug: under TP the worker-side coordinator keeps the default `ft_started=True`. Only the scheduler's
    coordinator reads `start_on_launch`. The runner relays the worker's flag (`finetune_ft_started`), and the
    scheduler then opens admission, so `start_on_launch: false` is ignored at TP > 1.
  - Fixed in `v1/worker/gpu_worker.py`: the worker coordinator now takes `start_on_launch` too.
  - Earlier runs affected:
    - paper-1: no corpus at launch, so no effect.
    - agent-1 co: FT began ~5 s before its POST, which is negligible.
  - Relaunched.
- **Actual** (`RSI-system/deltaserve/results/rsi_poisson_2026-10-20/`; RESULTS.md §5). Request completion
  p50/p90/p99 in s, with the FT time:

  | rate | seq | co |
  |---|---|---|
  | 0.5 | 1.64 / 6.35 / 119.8, FT 126 s | 1.67 / 2.95 / 6.4, FT 306 s |
  | 1 | 1.92 / 86.0 / 127.3, FT 126 s | 1.82 / 3.54 / 6.2, FT 436 s |
  | 2 | 6.46 / 107.6 / 131.2, FT 126 s | 1.71 / 2.63 / 5.5, FT 267 s |

  - Under seq, 51 / 135 / 249 requests were held for ~65–71 s each.
  - Criterion 1 (co p90 < seq p90 at every rate): **PASS**.
  - Criterion 2 (co FT ≤ 3× seq): **FAIL at 1 rps** (3.5×). It held at 0.5 rps (2.4×) and 2 rps (2.1×).
  - Predictions:
    - FT alone ≈ 130 s: actual 126. ✔
    - seq p90 > 30 s at 1–2 rps: actual 86 / 108. ✔
    - co FT ≈ 280 s at 0.5 rps: actual 306. ✔
    - co p50 +20 %: actual +2 % at 0.5 rps and −6 % / −74 % at 1 / 2 rps (seq's p50 includes holds). ✘ — better
      than predicted.
    - co FT ~400–700 s at 2 rps: actual 267 s, but **0 tokens trained while serving**. The whole job ran after
      the traffic ended, so the prediction was right for the wrong reason.
  - FT rate while serving: 41 / 23 / 0 % of standalone (4.18K tok/s).
- **Decision / insight:**
  - For RSI the right comparison is service availability vs adapter freshness.
    - Co-serving keeps request p99 ≤ 6.4 s at every rate, where stopping to train costs a 65–71 s wait for 10–50 %
      of the requests.
    - The cost is an adapter that is ready 140–309 s later.
  - DeltaServe's FT share collapses with load (0 % at 2 rps on 4×H200 with ACE prompts). This is the same
    fragmented-gap problem as agent-1.
  - The next lever is still FT steps sized to the gaps, plus revisiting the 0.1 s-window RPS throttle.

### rsi-tracelab — RSI traffic with TraceLab session timing: stop serving to train vs co-serving   (2026-10-22, fs-mbz-gpu-198 ×8: co on 0-3, seq on 4-7)
- Reflection/audit: rsi-poisson (2026-10-20) sent each ACE request as an independent Poisson arrival. The user asked
  for request arrivals that follow TraceLab. Checked TraceLab: the corpus records no session arrival process
  (csv_export: "session arrival times are synthetic"); what it does record is per-session timing. The real
  per-user inter-request gaps are p50 7.4 s, p90 58 s; an active user sends ~0.03 req/s; the aggregate gap CV is 28
  (Poisson = 1). So: sessions arrive Poisson (as TraceLab's simulator does), and inside a session every round waits its
  real TraceLab gap (tool latency / human think time, capped at 300 s), then sends the next ACE request and waits for
  its answer. Mean request rate matched to rsi-poisson (0.5 / 1 / 2 rps) by the session rate, so the only change vs
  rsi-poisson is the arrival shape. The FT admission scheduler is unchanged (the user: fix "too conservative" later),
  so this run measures the workload, not a scheduler fix.
- Goal/criteria (same as rsi-poisson): co request-completion p90 < seq p90 at every rate; co FT time ≤ 3× seq FT time.
  Also report per-task completion (a user round + its tool loop), since a held request now stalls its whole session.
- Workload: `agent_trace.py --db syfi_coding_trace.duckdb --sessions 60 --rate {0.0308,0.0686,0.147} --seed 0`
  (scale/caps default) -> `work/runs/rsit/workload_r{0.5,1,2}.json`. With 2 s responses this gives the first 500
  requests at 0.47 / 0.98 / 1.88 rps; gap CV 1.86 / 0.83 / 1.04; peak 10-s rate 1.5 / 2.4 / 3.7 rps; 33 / 36 / 41 sessions.
  ACE formula train split, 500 requests, each sent once in order across sessions.
- Command: `work/rsit.sh` = for r in 0.5 1 2: `RATE=$r WORKLOAD=work/runs/rsit/workload_r$r.json
  DATA=work/runs/paper1/sft_from_traces.jsonl EPOCHS=2 TRAIN_AT=120 scripts/rsi_poisson.sh {co 0,1,2,3 8000 |
  seq 4,5,6,7 8100} work/runs/rsit/{co,seq}_r$r` (backward NCCL ports 29677 / 29777). Same server config as
  rsi-poisson (train_only.yaml + profile_on_launch, Qwen3-32B YaRN TP=4, FT 427 trajectories × 2 = 526K tokens).
- Predicted:
  - seq: FT ≈ 126 s at every rate. Fewer requests held than rsi-poisson, because a held request stalls its session
    (closed loop), but held tasks take ≥ 60 s longer.
  - co: p90 ≈ 3–4 s at every rate. 0.5 rps is burstier (CV 1.9), so the idle gaps are longer and the FT share is a
    bit higher than rsi-poisson's 41 % (adapter ready ≈ 350–420 s). At 1 and 2 rps the shape is near-Poisson, so
    expect the same collapse (≈ 20 % and ≈ 0 %).
  - Criterion 1 PASS at all rates. Criterion 2 likely FAILS at 1 rps again.
  - FALSIFIED if co p90 > 10 s at any rate (bursts break the SLO gate), or if co FT share at 2 rps is > 20 % (the
    bursts leave usable gaps that Poisson did not).
- Actual (2026-10-22, all 6 cells, 500/500 answered, 0 errors; accuracy 343–348/500 everywhere):

  | rps | mode | request p50 / p90 / p99 (s) | held (mean wait) | FT time | adapter ready | task p50 / p90 (s) |
  |---|---|---|---|---|---|---|
  | 0.5 | seq | 1.58 / 2.39 / 6.1 | 1 (125 s) | 125 s | 247 s | 29.8 / 231 |
  | 0.5 | co | 1.60 / 2.54 / 5.2 | 0 | 254 s | 374 s | 29.6 / 212 |
  | 1 | seq | 1.62 / 2.76 / 106 | 13 (77 s) | 126 s | 247 s | 36.2 / 222 |
  | 1 | co | 1.60 / 2.56 / 5.9 | 0 | 491 s | 611 s | 25.9 / 135 |
  | 2 | seq | 1.77 / 3.28 / 124 | 29 (77 s) | 125 s | 246 s | 74.8 / 187 |
  | 2 | co | 1.70 / 2.69 / 5.6 | 0 | 272 s | 392 s | 25.0 / 100 |

  - Co FT trained during traffic: 50 % / 7 % / 0 % of standalone speed. At 1 and 2 rps, most or all of the FT
    ran after the traffic ended.
  - Criterion 1 (co request p90 < seq): FAIL at 0.5 (2.54 vs 2.39 s, a tie: 1 request was held); PASS at 1 and 2.
  - Criterion 2 (co FT ≤ 3× seq): PASS at 0.5 (2.0×) and 2 (2.2×, but only because the traffic ended); FAIL at 1
    (3.9×).
  - Predictions: seq held fewer than under Poisson ✓ (1 / 13 / 29 vs 51 / 135 / 249: a held request stalls its
    session, so the backlog does not grow). Co p90 2.5–2.7 s, better than predicted. FT share at 0.5 ✓ (50 % vs
    the predicted > 41 %). At 1 rps it was worse than predicted (7 % vs ≈ 20 %). At 2 rps ✓ (0 %). Neither
    falsifier fired.
  - Insight: with agent sessions, the cost of stopping to train shows up per task, not per request. Most
    requests are untouched, but a task whose request is held stalls its whole tool loop: seq task p50 75 s vs
    co 25 s at 2 rps, and p90 187 vs 100 s. Co-serving still does not use the gaps: the 1 rps run trained 7 %
    while serving.
  - Cell-to-cell variation: which ACE question lands in which session depends on response timing, so the task
    counts differ slightly (45 / 43 / 47 vs 45 / 43 / 45).
- Decision: report to the user. The scheduler fix (gap-sized FT steps plus the RPS throttle) is the next lever,
  on the user's go.

### rsi-tracelab-ride — the co-serving half of rsi-tracelab with DeltaServe's standard FT admission   (2026-10-22, fs-mbz-gpu-198 ×8)
- Reflection/audit: rsi-tracelab's co-serving trained only 50 / 7 / 0 % of standalone speed while serving. The user
  asked why DeltaServe's scheduler did not train during serving; that is its whole purpose. The audit found the
  cause in our config, not in DeltaServe. Every RSI config since 2026-10-19 (train_only.yaml, rsi_e2e, rsi_ace)
  inherited two settings from the 16K long-context config (qwen3_32b_tp4_16k):
  1. `ft_tokens_admission_constrain_factor: 0`: an FT sample may never ride an inference prefill step. FT runs only
     in fully idle steps.
  2. The RPS throttle (0.1 s window, close at 5 rps, 1 s hold): any single arrival reads as 10 rps and closes FT.

  Those settings made sense for 16K-token samples, which cannot share a step without blowing TTFT. The RSI samples
  are 502–907 tokens (ft-store log). The validated Qwen3-14B config uses factor -1 (the SLO estimator decides) and no
  throttle. So rsi-poisson and rsi-tracelab measured DeltaServe with its core mechanism off. This run measures it on.
- Change vs rsi-tracelab, co only: `FT_ADMIT=ride` -> `ft_tokens_admission_constrain_factor: -1`,
  `rps_throttle_enable: false`. Everything else is identical: workloads, ACE requests, FT data, 16K activation
  buffer, 16K tokens per optimizer step, SLO (ttft 0.3, max_tbt 0.05), forward_interruptible, admission phase
  "prefill". The seq baseline is reused from rsi-tracelab: there FT runs alone, so neither setting is active.
- Command: `work/rsit2.sh`:
  - 0.5 rps on GPUs 0-3 (port 8000, NCCL 29677) and 1 rps on GPUs 4-7 (port 8100, NCCL 29777), in parallel;
  - then 2 rps on GPUs 0-3;
  - each: `FT_ADMIT=ride RATE=$r WORKLOAD=work/runs/rsit/workload_r$r.json DATA=work/runs/paper1/sft_from_traces.jsonl
    EPOCHS=2 TRAIN_AT=120 scripts/rsi_poisson.sh co <gpus> <port> work/runs/rsit2/co_r$r`.
- Criteria:
  - (1) co request p90 ≤ 4 s at every rate (rsi-tracelab co: 2.5–2.7 s; seq: 2.4–3.3 s).
  - (2) co FT trained during traffic ≥ 40 % of standalone at 1 rps (was 7 %).
  - (3) adapter ready ≤ 1.5× seq's at every rate (seq ≈ 247 s -> ≤ 370 s).
- Predicted:
  - FT share during traffic ≈ 80 / 50 / 30 % at 0.5 / 1 / 2 rps.
  - Adapter ready ≈ 290 / 360 / 380 s. At 2 rps the 500 requests end at about 270 s, so its tail still trains
    alone after the traffic.
  - Request p90 ≈ 3 s (the estimator holds each step inside the TTFT SLO). TTFT p99 rises toward the SLO (≈ 0.3 s).
  - FALSIFIED if FT share at 1 rps < 25 % (then the bottleneck is elsewhere: the backward child, the pause contract,
    or tier-C aborts) or if request p90 > 5 s (riding costs the users).
- Actual (work/runs/ridecmp = rsit seq + rsit2 co, `rsi_poisson_metrics.py`):
  - adapter ready 378 / 617 / 391 s at 0.5 / 1 / 2 rps (rsi-tracelab co with the switches off: 374 / 611 / 392; seq
    247 / 247 / 246). Identical to the switches-off run within noise.
  - request p90 2.52 / 2.58 / 2.50 s (seq 2.39 / 2.76 / 3.28); TTFT p99 86 / 174 / 52 ms; max-TBT p99 ≤ 50 ms.
  - At 1 rps FT reached ~64K tokens after ~3 min of traffic (≈ 6-10 % of standalone), 63 tier-C aborts; the
    server log shows no FT sample on any inference step, only FT-only 16K forwards in idle gaps.
  - accuracy 346 / 345 / 341 of 500 (unchanged).
  - Criteria: (1) PASS, (2) FAIL (≈ 6-10 % at 1 rps), (3) FAIL at 1 rps (617 s > 370 s). FALSIFIED.
- Decision: the two switches were not the binding constraint. The prompts are ~390 tokens and a prefill step is
  ~38 ms, so with max_tbt_slo 0.05 s and phase "prefill" (decode-only steps admit nothing) almost no step has room
  for a 500-900-token sample. Next run (rsi-mix) relaxes the SLO (ttft 0.5, max_tbt 0.2 = DeltaServe's default) and
  uses phase "both", and at the same time makes the workload less thin (mixed formula + finer traffic, 4 FT rounds)
  and adds the vLLM + torchtune baseline the user asked for.

### rsi-mix — mixed RSI traffic (formula + finer) + 4 FT rounds: seq vs DeltaServe co-serving (relaxed SLO) vs vLLM+torchtune   (2026-10-26, fs-mbz-gpu-198 ×8 H200, job 2721371)
- Reflection/audit: rsi-poisson / rsi-tracelab / rsi-tracelab-ride all found DeltaServe co-serving barely trains
  under traffic (≈ 6-10 % of standalone at 1 rps). ride showed the factor-0 / RPS-throttle switches were not the
  cause; the binding constraint is the SLO + admission phase (max_tbt 0.05 s vs a ~38 ms prefill step; decode-only
  steps admit nothing under phase "prefill"). The user also called the experiments thin (one task, one FT job,
  no external baseline) and asked for (a) a relaxed SLO, (b) several FT rounds per run, (c) finer + formula
  together, (d) a comparison with vLLM + torchtune (DeltaServe-Baselines).
- What changed vs rsi-tracelab:
  - traffic: ACE formula + finer train splits interleaved 1:2 (500 + 1000 requests; finer prompts ~1.8K tokens,
    formula ~390), TraceLab session timing (agent_trace.py: r1 = 200 sessions @ 0.072/s ≈ 1.0 rps, r2 = 300 @ 0.15/s
    ≈ 2.1 rps over 750 s), sending stops at 750 s (or when the 1500 requests are used up).
  - FT: 4 rounds (rsi_mix_prep.py: round k = chunk k of formula traces SFT (427 rows) + chunk k of data_v2/finer
    (277 rows); 172 samples / ~230K tokens each, 1 epoch), started at 120 / 270 / 420 / 570 s. DeltaServe: round 0
    = launch corpus + /start_finetuning, rounds 1-3 POST /ft/ingest. Round k "done" = cumulative trained input
    tokens reach rounds 0..k.
  - DeltaServe co: ttft_slo 0.5, max_tbt_slo 0.2 (DeltaServe default), coserving_admission_phase both, factor -1,
    no RPS throttle. seq: same server, FT alone per round (traffic held).
  - tt: stock vLLM 0.21.0 (baseline-vllm venv, plain Qwen3-32B, TP=4, gpu_mem 0.45, LoRA on, max_num_batched 16384,
    long_prefill 2048, MPS 80 %) + torchtune 0.6.1 lora_finetune_distributed (FSDP 4 ranks, batch 1/rank, its own
    weights, activation checkpointing, MPS 20 %), same LoRA recipe (qkvo r16 a32, lr 1e-4 constant, wd 0), the same
    rendered + masked samples (tt_rsi.rounds_dataset; a round's samples block until the driver releases it).
    Caveats: plain RoPE vs DeltaServe's YaRN config (no compute difference at these lengths); different engine
    version (0.21.0 vs our 0.21.1rc0 fork); torchtune steps every 4 samples, DeltaServe every 16K tokens.
- Standalone FT (measured): torchtune alone on GPUs 4-7, all 4 rounds: 5.3K tok/s. DeltaServe's from the seq cells.
- Command: `ROUNDS=work/runs/rsimix/rounds WORKLOAD=work/runs/rsimix/workload_r$r.json [TTFT_SLO=0.5 MAX_TBT_SLO=0.2
  PHASE=both] scripts/rsi_mix.sh <seq|co|tt> <gpus> <port> work/runs/rsimix/<cell>_r$r` via work/rsimix.sh
  (batch 1: co r1 on 0-3 + tt r1 on 4-7; batch 2: seq r1 + co r2; batch 3: seq r2 + tt r2).
- Criteria:
  - (1) co FT share during traffic at 1 rps ≥ 40 % (was 6-10 %).
  - (2) co mean round wait ≤ 150 s (keeps up with a round every 150 s) at 1 rps.
  - (3) co request p90 < seq request p90, and co agent-task p90 < seq's (seq holds traffic ~4 × 60 s).
  - (4) co beats tt on mean round wait at equal or better request p90.
- Predicted:
  - seq: round wait ≈ 60 s (230K / ~4.5K tok/s + drain); request p90 ≈ 20-40 s at 2 rps from the holds.
  - co: FT share ≈ 50 % at 1 rps, 30 % at 2 rps; round wait ≈ 100 / 160 s; request p90 within 1.3× co-strict
    (≈ 3-5 s on this longer traffic); max-TBT p99 rises to ~150-200 ms (the price of max_tbt 0.2).
  - tt: torchtune at a 20 % SM cap ≈ 25-35 % of its 5.3K under traffic -> round wait ≈ 150-250 s (falls behind);
    request p90 ≈ 1.2-1.5× co (vLLM at an 80 % SM cap plus memory-bandwidth contention).
  - FALSIFIED if co FT share at 1 rps < 25 %, or co request p90 > 2× co's TTFT-only change would explain, i.e. > 8 s.
- Incidents (all invalid cells kept under work/runs/rsimix/{prefix,tt_failed,confounded}):
  - editing rsi_mix.sh while a cell ran it broke batch 1 (bash reads scripts lazily); teardown did not kill the
    DeltaServe serve process (own session), so the next cell's health check hit the old server; the fixed teardown
    then killed the batch's process group once. Fixed: cleanup scoped to the cell's out-dir/port.
  - vLLM under MPS: a client cap of 60/75/80/85 % (and 70 % on the full model) makes cuBLAS bf16 GEMMs (torch
    2.11 + cu129) return garbage ("!!!!" outputs); 100 % + torchtune 20 % deadlocked vLLM's TP all-reduce once
    torchtune trained. The vLLM+torchtune baseline therefore runs time-sliced (no MPS), like DeltaServe's child.
    (DeltaServe-Baselines' 80/20 results on fs-mbz-gpu-903 probably measured latency on corrupted outputs.)
  - GPU 2 of fs-mbz-gpu-198 thermal-throttles to half speed (83 C, reason 0x20, 356 vs ~665 TFLOP/s) since
    ~16:55 2026-10-26; every TP=4 run on GPUs 0-3 after that was ~2x slower (co_r2 twice). Later runs use 4-7.
- DeltaServe bug found and fixed (dserve-vllm deltaserve/ft_save_op.py, accumulate.py, v1/worker/gpu_worker.py):
  the graph-capturable save op gathered max_saved (16384) rows on each of its 448 calls per mixed step, ~150 ms
  of GPU time per FT-carrying step on Qwen3-32B TP=4 regardless of FT size; the estimator learned "any FT step
  >= 180 ms" and FT never rode an inference step (pre-fix co r1: FT ~10 % of standalone during traffic). Now a
  capture gathers graph_ft_max_tokens rows and the uncaptured path the step's own count (tests/test_ft_save_op.py
  42/42). Pre-fix co cells are in prefix/.
- Actual (rsi_mix_report.py; standalone DeltaServe FT 3.4K tok/s from seq; co = post-fix, ttft 0.5 / max_tbt 0.2
  / phase both; tt = time-sliced):
  | rps | sys | request p50/p90/p99 s | agent task p50/p90 s | round wait mean s | all rounds done s | max-TBT p99 ms |
  | 1 | seq | 2.6 / 4.1 / 70.9 | 80 / 221 | 70 | 640 | 170 |
  | 1 | co  | 3.1 / 7.3 / 11.9 | 55 / 217 | 302 | 902 | 1081 |
  | 1 | tt  | 2.9 / 11.9 / 23.5 | 60 / 258 | 112 | 683 | 1191 |
  | 2 | seq | 2.8 / 4.6 / 73.6 | 57 / 252 | 72 | 642 | 200 |
  | 2 | tt  | 3.4 / 16.8 / 29.9 | 73 / 267 | 121 | 689 | 1191 |
  | 2 | co  | 4.9 / 9.3 / 13.9 | 55 / 242 | 191 | 781 | 1177 |
  (2 rps co on healthy GPUs 4-7; the two runs on GPUs 0-3 were slowed by the throttled GPU 2 and are in confounded/.)
  Accuracy equal across systems (formula ~70 %, finer ~27 %: the base model serves all traffic).
  Criteria at 1 rps: (1) FT share 29 % < 40 % FAIL; (2) round wait 302 s > 150 s FAIL; (3) p90 7.3 > seq 4.1 FAIL,
  agent-task p90 217 < 221 PASS (barely); (4) co round wait 302 > tt 112 FAIL. Not falsified (share > 25 %), but
  co-serving loses on keeping up with the RSI loop.
- Decision: the user asked to fix the training scheduler. Diagnosis on diag2 (1 rps, post-fix trace): forward is
  no longer the limit (84 mixed steps carried 178K FT tokens in 14 s); backward pending (admission closed) 80 of
  240 s with cycles of 6.7 s vs ~3.8 s alone (paused on every prefill); ~146 s admission open but nothing fit
  (finer samples' mixed step ~170 ms vs the 0.7 x 0.2 = 140 ms decode-only budget). Next run: rsi-mix-sched.

### rsi-mix-sched — DeltaServe co-serving with the scheduler fixes   (2026-10-28, fs-mbz-gpu-198 GPUs 4-7, job 2721371)
- Changes (all config-gated, defaults = previous behaviour):
  - finetune.pause_min_prefill_tokens (new): pause the backward child only for steps with >= N inference prefill
    tokens. Run: 4096 (no single step of this traffic reaches it; prefills are time-sliced against the child).
  - slo.ft_decode_stall_budget / ft_stall_window_s (new): average-TBT gate, the share of wall time in a window that
    FT may add (predicted) to steps carrying decodes. Run: 0.3 over 2 s. tests/test_ft_stall_gate.py 4/4.
  - slo.decode_only_ft_safety_margin 0.7 -> 1.0 (the ft_mixed regime is fitted by the profiling pass).
- Command: `ROUNDS=work/runs/rsimix/rounds WORKLOAD=work/runs/rsimix/workload_r$r.json TTFT_SLO=0.5 MAX_TBT_SLO=0.2
  PHASE=both DECODE_MARGIN=1.0 PAUSE_MIN_PREFILL=4096 STALL_BUDGET=0.3 STEP_TRACE=1 scripts/rsi_mix_v2.sh co 4,5,6,7
  8100 work/runs/rsimix/cofix_r$r` (r1 then r2, work/rsimix11.sh). Same traffic, rounds and data as rsi-mix.
- Predicted (1 rps): backward cycles ~4 s; FT share during traffic 55-70 %; round wait 100-150 s (keeps up);
  request p90 6-9 s (more stall than co, bounded by the gate); max-TBT p99 ~1 s (backward time-slicing, unchanged).
  2 rps: no starvation (backward completes during traffic); round wait 150-250 s.
- Criteria: (1) 1 rps round wait <= 150 s; (2) FT share >= 50 %; (3) request p99 < tt's (23.5 s) and agent-task
  p50 <= tt's; (4) 2 rps trains during traffic (share > 20 %).
- FALSIFIED if 1 rps FT share < 40 % or round wait > 200 s (then the binding limit is the closed admission during
  backward, i.e. the single activation buffer, not pausing or the decode budget).
- Actual (cofix_r1 / cofix_r2 vs co_r1 / co_r2, same GPUs 4-7 class, rsi_mix_report.py):
  | rps | sys | request p50/p90/p99 s | TTFT p99 ms | agent task p50/p90 s | round wait mean s | all done s | FT share |
  | 1 | co    | 3.1 / 7.3 / 11.9 | 605  | 55 / 217 | 302 | 902 | 0.29 |
  | 1 | cofix | 5.1 / 10.1 / 14.8 | 1959 | 83 / 239 | 176 | 769 | 0.41 |
  | 2 | co    | 4.9 / 9.3 / 13.9 | 967  | 55 / 242 | 191 | 781 | 0.40 |
  | 2 | cofix | 6.3 / 12.0 / 19.6 | 2446 | 67 / 266 | 166 | 758 | 0.43 |
  (tt: 1 rps 2.9 / 11.9 / 23.5, round wait 112; 2 rps 3.4 / 16.8 / 29.9, round wait 121.)
  Criteria: (1) 176 > 150 FAIL; (2) 0.41 < 0.50 FAIL; (3) p99 14.8 < 23.5 PASS, agent p50 83 > tt 60 FAIL;
  (4) 2 rps trains during traffic PASS (it already did on healthy GPUs). Not falsified (share > 0.40, wait < 200).
  Trace (cofix_r1): backward pending 444 of 632 s; cycles 7.7 s (3.5 s alone) — the child time-slices ~50/50 with
  the continuous decodes (decode step 8.5 -> 19 ms while it runs); forward is not a limit (895K FT tokens rode
  357 steps in 57 s). Prefills no longer pause the child: their p99 with a backward running is 1.07 s (TTFT p99 2 s).
- Decision: the fixes move DeltaServe along the trade-off (round wait -42 % at 1 rps, -13 % at 2 rps; request
  p50/p90 +30-60 %), not past torchtune. The remaining training gap is the backward's GPU efficiency: alone it
  does 4.3K tok/s (+ forward = 3.4K) vs torchtune's 5.4K fwd+bwd with recompute (~25 % MFU on 32B TP=4). Next:
  profile the backward cycle (graph coverage, per-layer pause boundaries, NCCL residual reduces), and sweep
  pause_min_prefill / stall budget for the latency side.

### rsi-mix-loose — the scheduler fixes with a looser SLO   (2026-10-28, fs-mbz-gpu-198 GPUs 4-7, queued behind carbon3)
- Why: the user asked to relax the SLO. cofix (ttft 0.5, max_tbt 0.2, stall 0.3) cut the round wait 42 % at 1 rps but
  the binding limit is now the backward's time-sliced GPU share, so a looser SLO mostly buys FT admission that is
  already not the bottleneck. Expect a small gain in training and the latency results judged against a looser bar.
- Changes vs cofix: ttft_slo 1.0 (DeltaServe default), max_tbt_slo 0.3, ft_decode_stall_budget 0.5.
- Command: work/rsimix12.sh (waits for GPUs 4-7 to be idle 2 min, then coloose_r1, coloose_r2 via rsi_mix_v2.sh).
- Caveat: cofix_r2 (13:13-13:30) overlapped another session's carbon2 run on GPUs 0-3 (same node, CPU/PCIe shared).
- Predicted: round wait 1 rps 150-170 s, 2 rps 150-160 s; request p50 +10-20 % vs cofix; TTFT p99 ~2 s (unchanged:
  prefill time-slicing, not admission). FALSIFIED if round wait drops below 130 s at 1 rps (then admission still
  limited training, i.e. my bottleneck read is wrong).
- Actual: (pending)
- Actual (coloose_r1, 1 rps): round wait 130 / 134 / 134 / 137 s (mean 134; cofix 176, tt 112); request
  2.6 -> 4.2 / 10.2 / 15.7 s p50/p90/p99; TTFT p99 1.09 s (cofix 1.96); agent task p50 89 s. Better than predicted
  (150-170 s), just above the 130 s falsification line: admission was still part of the limit. 2 rps: round wait
  131 s (tt 121), request 4.9 / 12.6 / 19.5 s, agent task p50 68 s.

### rsi-mix-fast — a 3.2x faster backward (TF32 + fused SDPA attention) under the loose-SLO scheduler   (2026-10-28, GPUs 4-7)
- Audit: the user's goal is to beat vLLM + torchtune outright. DeltaServe's backward was the gap: alone 4.3K tok/s
  vs torchtune 5.4K. Profile (eval-tp/profile_backward.py, one TP=4 rank shard, 15.6K tokens, real
  Qwen3BackwardService on random weights): 78 % of the cycle in fp32 GEMMs on CUDA cores — the backward child is a
  fresh spawned process, so its fp32 LM head (~1.4 s/cycle) and per-sample fp32 attention never used tensor cores.
- Changes (config-gated, default off = bit-identical; default-path gates re-pass: qwen3 18/18, llama3 12/12,
  long-context 26/26, qwen3 graph 475/475):
  - finetune.backward_tf32: TF32 for the child's fp32 matmuls.
  - finetune.backward_attn_sdpa: attention backward through torch SDPA (cuDNN flash kernel, bf16 in, fp32 acc)
    instead of the materialized fp32 [Hq, L, L] scores (tests/test_attn_sdpa.py 18/18, rel-L2 2.5e-3 vs fp32).
- Microbenchmark per 64-layer cycle (one rank): fp32 4.66 s -> TF32 2.56 s -> TF32+SDPA 1.45 s (per layer 50 ->
  36 -> 19 ms; fixed 1429 -> 238 ms). Precision vs strict fp32 (eval-tp/tf32_check.py): loss rel 7e-7, LoRA grads
  rel-L2 0.9-1.2 %, cosine >= 0.99993 — the bf16 bulk matmuls' own error.
- Command: work/rsimix13.sh after coloose: cofast_r1, cofast_r2 (coloose knobs + BWD_TF32=1 BWD_SDPA=1,
  rsi_mix_v3.sh), then seqfast_r1 (seq with the fast backward: standalone speed + loss-curve sanity vs seq_r1).
- Predicted: backward cycle under traffic 7.7 s -> ~3.5 s; 1 rps round wait ~80-100 s (< tt 112), FT share
  ~0.8-1.0 of the OLD standalone; request latency similar to coloose (backward occupies the GPU half as long,
  so maybe better); seqfast round ~45 s (standalone ~5.5-6.5K tok/s, > torchtune 5.4K); seqfast loss curve
  within noise of seq_r1. FALSIFIED if cofast_r1 round wait > 112 s (tt) or seqfast loss diverges from seq_r1.
- Actual (rsi_mix_report.py; results/rsi_mix_2026-10-28 in RSI-system):
  | rps | sys | request p50/p90/p99 s | TTFT p50/p99 ms | max-TBT p50/p99 ms | agent task p50/p90 s | round wait s | all done s |
  | 1 | cofast | 2.70 / 8.2 / 13.3 | 50 / 819 | 28 / 1079 | 61 / 181 | 74 (78, 71, 73, 76) | 646 |
  | 1 | tt     | 2.87 / 11.9 / 23.5 | 56 / 1618 | 72 / 1191 | 60 / 258 | 112 | 683 |
  | 1 | seq    | 2.57 / 4.1 / 70.9 | 50 / 428 | 27 / 170 | 80 / 221 | 70 | 640 |
  | 2 | cofast | 2.87 / 9.0 / 17.5 | 52 / 1114 | 80 / 1517 | 43 / 210 | 70.5 | 639 |
  | 2 | tt     | 3.40 / 16.8 / 29.9 | 89 / 1906 | 88 / 1191 | 73 / 267 | 121 | 689 |
  | 2 | coloose| 4.9 / 12.6 / 19.5 | 100 / 1354 | 202 / 1262 | 68 / 252 | 131 | 699 |
  seqfast_r1: rounds trained in 41 / 39 / 40 / 39 s -> standalone 5791 tok/s (seq_r1 3397; torchtune 5426); loss
  per cycle identical for the first 3 cycles, mean 0.2709 vs 0.2710. Backward cycle under 1 rps traffic 3.8 s
  (cofix 7.7 s). Accuracy equal to tt/seq.
  Prediction check: round wait 74 s at 1 rps (predicted 80-100, PASS), standalone 5.8K (predicted 5.5-6.5K, PASS),
  loss within noise (PASS); latency better than coloose (predicted similar). Not falsified.
- Decision: DeltaServe (scheduler fixes + loose SLO + fast backward) beats vLLM + torchtune on round wait (-34 %
  at 1 rps, -42 % at 2 rps), request p50/p90/p99, TTFT p50/p99 and agent-task p90; ties agent-task p50 at 1 rps
  (61 vs 60 s) and wins it at 2 rps (43 vs 73 s); loses only max-TBT p99 at 2 rps (1.5 vs 1.2 s: decodes
  time-sliced with the backward). Next if needed: the worst-token-gap tail (pause the backward for a decode step
  that has waited > X ms), and the remaining ~12 ms/layer of elementwise backward ops (fusion).

### rsi-e2e-fast — the end-to-end ACE loop (run_e2e.sh) with the RESULTS §7 co-serving changes   (2026-10-28, GPUs 4-7, eval GPU 0)
- Why: the user asked to rerun the end-to-end ACE experiment on the improved DeltaServe. Prior runs (e2e 10-19 x2):
  C trained 526K tok in 4 min 35 s / 4 min 37 s; D adapter 75.0 / 74.5 % vs base 68.0 / 66.5 % (p < 0.01).
- Command: `CONFIG=deltaserve/configs/coserve_e2e_fast.yaml DS_GPUS=4,5,6,7 EVAL_GPU=0 bash deltaserve/scripts/run_e2e.sh e2e_fast`
- Predicted: C ~2.5 min; C TTFT p50 <= 84 ms; D 74-75 %. FALSIFIED if D < 72 % or C slower than 4.5 min.
- Actual (work/runs/e2e_fast): C trained 34 cycles / 526K tok in 2 min 38 s (was 4:35); loss 0.287 -> 0.149;
  D adapter 77.0 % vs base 68.5 % (+22/-5, p = 0.0015); B 427/500. BUT C latency (n=77): TTFT p50/p90/p99
  96 / 1497 / 2229 ms (was 81 / 173 / 262), < 0.3 s 69 % (was 99 %), e2e p50/p90 5.9 / 11.0 s (was 2.1 / 3.9),
  per-request max TBT p50 557 ms (was 20 ms). Criteria: training PASS, accuracy PASS, TTFT p50 96 > 84 FAIL.
- Decision: the §7 admission settings (phase both, stall budget 0.5, no prefill pause) are tuned for throughput
  under mixed traffic; at 0.5 rps light formula traffic the old conservative admission already finds enough idle
  GPU time, and the aggressive one just stalls users. Next: rsi-e2e-fastbwd — the ORIGINAL admission + only the
  fast backward (backward_tf32, backward_attn_sdpa).

### rsi-e2e-fastbwd — end-to-end loop, original admission + only the fast backward   (2026-10-28, GPUs 4-7, eval GPU 0)
- Why: rsi-e2e-fast trained 42 % faster but hurt C latency badly (TTFT p90 1.5 s). Separate the two effects: keep
  coserve_e2e.yaml's admission (idle-gap FT, RPS throttle, prefill phase, SLO 0.3 / 0.05) and turn on only
  backward_tf32 + backward_attn_sdpa.
- Command: `CONFIG=deltaserve/configs/coserve_e2e_fastbwd.yaml DS_GPUS=4,5,6,7 EVAL_GPU=0 bash deltaserve/scripts/run_e2e.sh e2e_fastbwd`
- Predicted: C ~3-3.5 min (backward 1.7x faster, forward/idle-gap admission unchanged); C TTFT like the old runs
  (p50 ~85 ms, p99 < 0.8 s, < 0.3 s >= 95 %); D 74-77 %. FALSIFIED if C TTFT p90 > 0.4 s or C not faster than 4:35.
- Actual (work/runs/e2e_fastbwd): C 34 cycles / 527K tok in 3 min 25 s (old 4:35, -25 %); loss 0.286 -> 0.152;
  C TTFT p50/p90/p99 77 / 123 / 246 ms, < 0.3 s 99 % (old 81 / 173 / 262, 99 %); e2e p50/p90 1.99 / 3.12 s (old
  2.06 / 3.94); max TBT p50 17 ms (old 20). D 74.0 % vs base 68.0 % (+19/-7, p = 0.029; old 75.0 / 74.5). PASS on
  all three predictions.
- Decision: for the light single-task e2e loop the right setting is the original admission + the fast backward
  (configs/coserve_e2e_fastbwd.yaml): faster training at unchanged latency. The aggressive §7 admission is for
  heavy mixed traffic, where idle GPU time is scarce.

### rsi-mix-guard — slow-step guard against the max-TBT tail   (2026-10-28, fs-mbz-gpu-198 GPUs 4-7)
- Audit: rsi-mix final (cofast) beats vLLM+torchtune on 15/18 metrics, ties 2, loses 1: 2 rps max-TBT p99 1517 vs
  1191 ms. Single run per cell, no variance yet. Step trace of cofast_r2: decode-only steps with a backward in
  flight (decode_bwd) are p50 20 / p99 100 ms, but ~0.5 % stall 0.3-1.2 s (predicted 9 ms), at random phase of
  the 4.3 s backward cycle; many requests share one 1.517 s gap. Those steps run unpaused (cofast pauses only for
  prefills >= 4096 tok), so time-slicing against the 4-rank backward (incl. its spinning NCCL reduces) is the
  likely cause.
- Change: finetune.pause_on_slow_step_ms (new, default 0 = off; deltaserve/slow_step_guard.py): a per-rank
  watchdog pauses the backward child when an inference step dispatched during a backward is still not complete
  after X ms, released when no such step is in flight; the child's rank-symmetric pause agreement stops all ranks.
  Gate: tests/test_slow_step_guard.py 12/12 (GPU).
- Command (work/rsimix14.sh): cofast knobs `TTFT_SLO=1.0 MAX_TBT_SLO=0.3 PHASE=both DECODE_MARGIN=1.0
  PAUSE_MIN_PREFILL=4096 STALL_BUDGET=0.5 BWD_TF32=1 BWD_SDPA=1 STEP_TRACE=1` + `SLOW_GUARD_MS=80`,
  scripts/rsi_mix_v4.sh co 4,5,6,7 8100 -> coguard_r2, coguard_r1; then a cofast_r2 repeat (cofastb_r2, guard off)
  for run-to-run noise. Same workload_r{1,2}.json + rounds as rsi-mix.
- Predicted: coguard_r2 max-TBT p99 <= 400 ms (beats tt 1191), decode_bwd max < 0.3 s; round wait +0-10 %
  (70.5 -> <= 78 s); request p99 unchanged or better (17.5 s). FALSIFIED if max-TBT p99 stays > 1 s (then the
  stall is not the child's GPU share, e.g. host-side or NCCL coupling the pause doesn't reach) or round wait > 90 s.
- Actual coguard_r2 (2 rps, 1500/1500 ok): max-TBT p50/p99 79 / 364 ms (cofast 80 / 1517, tt 88 / 1191); TTFT
  p50/p99 53 / 461 ms (cofast 52 / 1114, tt 89 / 1906); request p50/p90/p99 2.96 / 6.58 / 10.77 s (cofast 2.87 /
  8.98 / 17.53, tt 3.40 / 16.8 / 29.9); agent task p50/p90 40 / 196 s (cofast 43 / 210, tt 73 / 267); round wait
  79.6, 83.3, 85.9, 78.0 = mean 81.7 s (cofast 70.5, tt 121.1); correct 347/500 + 277/1000. decode_bwd steps p99
  36 ms, max 0.299 s, 0 over 0.3 s (cofast p99 100 ms, max 1.16 s, 45 over 0.3 s). Predictions: max-TBT <= 400
  PASS, decode max < 0.3 s PASS, request p99 not worse PASS (better), round wait <= 78 s FAIL (81.7, +16 %).
  Now beats tt on every 2 rps metric. Single run; coguard_r1 + cofastb_r2 (noise) still running.

### rsi-burst — main comparison on the trace's REAL bursty timeline   (2026-10-28, fs-mbz-gpu-198 GPUs 4-7)
- Audit: every rsi-mix / rsi-tracelab run used agent_trace.py: real per-session gaps, but sessions start as a Poisson
  process; at 1-2 rps dozens of sessions overlap and the traffic is Poisson-like (measured on the runs: 10 s-window
  CV 0.33-0.52, inter-arrival CV 1.0-1.26, 0 % idle windows). The TraceLab trace itself (140K Claude requests, 243
  days) is very bursty: in hours with >= 100 req, 10 s-window CV 1.81, 70 % idle windows, peak 10x mean (>= 20 req/h:
  2.69, 84 %, 16x). The paper reports no arrival-process statistics. Our claim (agent traffic is bursty; co-serving
  trains in the valleys) was therefore never tested. MMPP session arrivals (agent_trace.py --arrival mmpp) did not
  reach the trace's statistics (CV <= 1.6, idle <= 27 %: overlapping sessions fill the gaps); a closed-loop replay of
  the real timeline flattens it too (CV 1.88 -> 0.6: ~10 sessions, one request in flight each).
- Workload: scripts/trace_replay.py --hours 24 --requests 1125 --duration 750 -> the real request timeline of a 24 h
  window (median-CV window of 462 candidates, start 2026-03-06 15:30 UTC, 10 sessions, 862 requests), compressed
  115x into 750 s and replayed OPEN-LOOP (rsi_traffic.py --replay-open): 1.15 rps mean, 10 s-window CV 1.88, 55 %
  idle windows, peak 9.3x mean. Second workload: --hours 12 (1336 req, 1.78 rps, CV 1.26, 29 % idle, peak 7x).
  Request content, FT rounds (4 x ~230K tok at 120/270/420/570 s) and metrics as rsi-mix.
- Cells (work/rsiburst.sh, scripts/rsi_mix_v5.sh, runs/rsiburst/<cell>_r<1|2>, r1 = 24 h, r2 = 12 h): coguard
  (cofast knobs + SLOW_GUARD_MS=80), tt (vLLM + torchtune, time-sliced), seq (stop serving to train, fast backward).
- Predicted (24 h): co round wait <= 75 s (more valleys than Poisson 2 rps: 81.7) and request p99 < tt's by >= 30 %;
  tt TTFT p99 > 2 s in the bursts; seq holds a burst whenever a round lands in one (request p99 > 60 s). FALSIFIED
  if co's round wait is not below tt's, or co's request p99 is worse than tt's.
- Actual coguard_r1 (1 rps): max-TBT p50/p99 78 / 294 ms (cofast 28 / 1079, tt 72 / 1191); TTFT p99 472 (cofast
  819, tt 1618); request p50/p90/p99 2.77 / 5.64 / 9.50 s (cofast 2.70 / 8.21 / 13.26, tt 2.87 / 11.9 / 23.5);
  agent task p50/p90 43 / 229 s (cofast 61 / 181, tt 60 / 258); round wait mean 88.2 s (cofast 74.1, tt 112.2).
  decode_bwd max 0.29 s (cofast 1.07). The max-TBT p50 jump 28 -> 78 ms is a bimodality artifact, not slower steps:
  per-request max-TBT is ~25 ms (no slow step met) or ~80-160 ms (met an FT-carrying mixed step, ~157 ms p50 in
  both runs); requests meeting >= 1 gap > 60 ms went 47 % -> 56 % (training spans more of the run: round wait +19 %)
  while gaps > 60 ms per request fell 3.43 -> 2.38. Cost of the guard at both rates: round wait +16-19 %.
- Actual cofastb_r2 (cofast repeat, guard off, 2 rps): round wait 74.1, 68.2, 70.4, 68.9 = 70.4 s (run 1: 70.5);
  request p50/p90/p99 2.86 / 9.28 / 16.77 s (2.87 / 8.98 / 17.53); TTFT p99 1138 (1114); max-TBT p99 1156 (1517);
  correct 346 + 271 (345 + 271). Run-to-run noise: round wait +-0.1 s, request p99 +-4 %, but max-TBT p99 +-25 %
  (a p99 of per-request maxima set by a handful of 1 s stalls) -- the original 1517-vs-1191 "loss" to tt was inside
  that noise; coguard's 364 ms is far outside it.
- Decision: the guard removes the decode stalls (max-TBT p99 3-4x lower, request p99 -35 %) for +16-19 % round wait,
  still 22-33 % below tt. Keep it on (SLOW_GUARD_MS=80) for the bursty main comparison (rsi-burst).
- Actual r1 (24 h window, 862 req, 0 errors in all three): request p50/p90/p99 seq 3.58 / 5.95 / 29.3 s (73 held),
  coguard 3.94 / 7.90 / 12.7 s (0 held), tt 23.6 / 63.9 / 78.6 s; TTFT p99 754 / 558 / 3784 ms; max-TBT p99 282 /
  331 / 1962 ms; agent task p50/p90 6.6 / 30.6, 9.6 / 28.8, 36.2 / 82.1 s; round wait mean 40.5 / 55.1 / 93.4 s;
  correct 366 / 368 / 374 of 862. Predictions: co round wait <= 75 PASS (55.1); co request p99 >= 30 % below tt
  PASS (-84 %); tt TTFT p99 > 2 s PASS (3.8); seq request p99 > 60 s FAIL (29.3: the valleys are long enough that
  a 40 s seq round holds only 73 requests). Not falsified vs tt; but seq is strong here: fastest rounds (40.5 vs
  55.1) and best median latency / task p50; co wins only the tails (request p99 2.3x, TTFT p99, task p90, 0 held).
- Decision: report honestly -- under bursty traffic co-serving dominates the shared-GPU baseline (tt) and trades
  ~36 % slower rounds vs seq for never blocking users. Checked: rounds 0-2 do NOT sit in valleys -- 73 / 46 /
  59 requests arrive during them (mean 12 / 3.5 / 7 in flight); backward cycles stretch 2.4 s -> 3.5-3.9 s (+50-60 %)
  while sharing the GPU, even at 3.5 requests in flight. Round 3 has no traffic and matches seq (39.8 vs 39.4 s).
  So the gap to seq is the price of serving during training, not a scheduler bug in the valleys.
- Actual r2 (12 h window, 1336 req, 0 errors): request p50/p90/p99 seq 3.59 / 38.8 / 49.4 s (418 held), coguard
  3.91 / 11.8 / 18.3 s (0 held), tt 17.5 / 99.0 / 130.6 s; TTFT p99 2753 / 662 / 4069 ms; max-TBT p99 576 / 506 /
  1884 ms; agent task p50/p90 9.7 / 46.9, 7.2 / 27.9, 25.9 / 76.7 s; round wait mean 42.6 / 60.7 / 104.8 s; correct
  565 / 551 / 562 of 1336. At this denser load co beats seq on every latency metric and on agent tasks (p90 -40 %);
  seq still has the fastest rounds (-30 %). tt is last everywhere (rounds +73 % vs co, request p99 7x).
