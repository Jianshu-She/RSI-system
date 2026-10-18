# ReasoningBank reproduction on WebArena-Shopping with Qwen3.8-27B

Paper: ReasoningBank (arXiv 2509.25140, ICLR 2026). Code: google-research/reasoning-bank (WebArena/).
Target numbers (paper Table 1, Shopping subset, 187 tasks, SR / avg steps):
- Gemini-2.5-flash: No Memory 39.0 / 8.2, Synapse 40.6, AWM 44.4, ReasoningBank 49.7 / 6.1, +MaTTS 53.0
- Gemini-2.5-pro:   No Memory 45.5 / 7.6, ReasoningBank 51.9 / 6.0
- Gemma-3-12B (Table 6): No Memory 17.1 / 13.7, ReasoningBank 24.1 / 11.8

## Setup (fs-mbz-gpu-388, 8x H200), 2026-10-12

- Environment: `am1n3e/webarena-verified-shopping:0.1.0` (Docker Hub) run with enroot, non-root
  (no docker group / sudo on this node; the original `shopping_final_0712.tar` mirrors are all dead:
  archive.org dark, CMU 403, Google Drive refused). Patched to run unprivileged: nginx on 7770, no `su`,
  mailcatcher + cron disabled, mysql/redis bound to 127.0.0.1. Base URL `http://localhost:7770`.
  Data check: task 117 GT "3/2/22" == first order of emma.lopez in DB (2022-03-02); 104,368 products.
- Tasks: vendored `third_party/webarena/test.raw.json` (patched shopping annotations), 187 shopping tasks.
- Backbone (agent + self-judge + memory induction): `Qwen/Qwen3.8-27B`, vLLM 0.31, TP=2 (GPU 0,1),
  thinking on, reasoning parser qwen3, max-model-len 262144. Agent temperature 0.7, max_new_tokens 65536 (run.py defaults);
  autoeval temperature 0; memory induction temperature 1.0 (paper code defaults).
- Retrieval embedding: `Qwen/Qwen3-Embedding-8B` via vLLM pooling (GPU 2) instead of gemini-embedding-001; top-1 (paper default n=1).
- GT grader LLM (llm_fuzzy_match / ua_match): `Qwen/Qwen3.5-397B-A17B-FP8` (GPU 4-7), thinking off, temp 0, instead of gemini-2.5-pro.
- Agent harness: unchanged paper code (BrowserGym 0.14.1 legacy GenericAgent, AX-tree obs, max 30 steps).
- Deviations from paper: environment build (WebArena-Verified image), backbone, embedding model, GT grader model.

## Runs

### smoke0 — single-task end-to-end check (2026-10-12 ~13:45, fs-mbz-gpu-388, GPU0-1 agent / GPU2 emb / GPU4-7 judge)
- Reflection: nothing run yet; all infra is new (enroot site, local endpoints swapped in). Need to prove the
  harness works before spending ~10 GPU-hours per full run.
- Goal/criteria: task webarena.117 (first purchase date, fuzzy_match "3/2/22") runs to completion, produces
  step pkl files + summary_info.json with cum_reward, no API/parse crash; autoeval json written.
- Command: `cd reasoning-bank/WebArena && python run.py --task_name webarena.117 --model_name qwen3.8-27b --results_path $RB_ROOT/results/smoke0`
  then `python -m autoeval.evaluate_trajectory --result_dir $RB_ROOT/results/smoke0/webarena.117 --model qwen3.8-27b --log_dir $RB_ROOT/results/smoke0/autoeval_logs`
- Predicted: completes in <15 steps, likely success (paper case study: baseline answered wrong date 3/11/23
  because it only looked at recent orders, so a failure here is also plausible). Per-step latency 10-40 s.
  FALSIFIED (harness broken) if: parse errors on every step, empty actions, or crash in evaluator.
- Actual: PASS. First attempt crashed (tokenizer lookup by served name -> fixed via AGENT_TOKENIZER; GT judge not yet up).
  Final: 4 steps, 52 s wall, answer "March 2, 2022", GT cum_reward=1.0; autoeval rm=true (agrees with GT);
  induce_memory produced 3 well-formed items (Title/Description/Content); retrieval via Qwen3-Embedding (dim 4096) works.
  Note BrowserGym calls the GT evaluator every step (answer="whatever"), same as the paper code.
- Decision: harness verified. Per task ~1 min agent + ~0.5 min autoeval/induction -> ~5 h per 187-task run.
  Launch No-Memory and ReasoningBank in parallel on two independent shopping instances (state isolation).

### shop_nomem_r1 + shop_rb_r1 — full Shopping (187) No-Memory vs ReasoningBank   (2026-10-12 ~13:58, fs-mbz-gpu-388 ×8 H200)
- Reflection/audit: smoke0 proved the harness end-to-end (agent -> GT reward -> self-judge -> induction -> retrieval).
  No full run yet. Paper reference on this subset: Gemini-2.5-flash 39.0 -> 49.7 (+10.7), Gemini-2.5-pro 45.5 -> 51.9 (+6.4),
  Gemma-3-12B 17.1 -> 24.1 (+7.0). Qwen3.8-27B is a much stronger web agent (vendor-reported WebArena-Verified 64.8,
  different harness), so the baseline should sit above the paper's Gemini numbers and headroom for memory is smaller.
- Goal/criteria: reproduce the *direction and rough size* of the ReasoningBank gain: RB SR - NoMem SR >= +3 pts and
  RB avg steps < NoMem avg steps, on all 187 shopping tasks, GT reward from the (patched) WebArena evaluator.
  No train/test split exists in this protocol (test-time streaming, task-id order, memory built only from
  past tasks' self-judged trajectories; GT never used for memory) -> no leakage by construction.
- Hypothesis: distilled strategy memory from both successes and failures transfers across shopping templates
  (many templates repeat with different parameters), improving SR and cutting steps.
- Commands (cwd reasoning-bank/WebArena, `source wa_env.sh`):
  - NoMem (instance B, WA_SHOPPING=http://localhost:7771):
    `python pipeline_memory.py --website shopping --model qwen3.8-27b --memory_mode no_memory --output_dir $RB_ROOT/results/shop_nomem_r1`
  - RB (instance A, WA_SHOPPING=http://localhost:7770):
    `python pipeline_memory.py --website shopping --model qwen3.8-27b --memory_mode reasoningbank --judge autoeval --output_dir $RB_ROOT/results/shop_rb_r1`
  - run.py defaults: temperature 0.7, max_steps 30, AX-tree obs, thinking prompt, multi_actions, top-1 memory.
- Predicted: NoMem SR ≈ 50 (range 42-58), avg steps ≈ 7-9; RB SR ≈ 56 (+6), avg steps ≈ 1 fewer.
  Self-judge accuracy vs GT ≈ 75-85%.
  Noise: single seed, 187 tasks -> SE of a difference ≈ 5 pts, so a gain < ~5 is not significant on its own.
  FALSIFIED if RB <= NoMem (gain <= 0) or RB uses more steps.
- Actual: KILLED after 2 tasks (21, 22; both 0, 30 steps). Bug: vLLM qwen3 reasoning parser moved the agent's
  <think> section into `reasoning`, so `think` was empty in every step -> history, self-judge and memory
  induction saw actions only (not the paper setting). Archived to results/_discarded/*_nothink.
- Decision: fix chat_api.py (re-wrap `reasoning` as <think>...</think> when content has none); smoke1 on task 117:
  think 228-1044 chars/step, reward 1.0. Relaunch as *_r2 with identical params. Instances not reset
  (tasks 21/22 are read-only review queries; no state change).

### shop_nomem_r2 + shop_rb_r2 — relaunch of r1 with think-section fix   (2026-10-12 ~14:10, fs-mbz-gpu-388 ×8)
- Same goal/criteria/hypothesis/prediction as r1 (NoMem ≈ 50, RB ≈ 56, RB fewer steps; FALSIFIED if gain <= 0).
- Commands: identical to r1 with output_dir results/shop_{nomem,rb}_r2 (NoMem -> localhost:7771, RB -> localhost:7770).
- Interruption (2026-10-12 17:09): all processes (vLLM, both enroot sites, both pipelines) died when the Claude session
  exited; /tmp/jianshu_enroot (container rootfs) was also wiped. State at the time: NoMem 103/187 done (last tid 334),
  SR 56.3, steps 6.51; RB 78/187 (last tid 283), SR 55.1, steps 6.92; memory bank 78 items. Self-judge vs GT agreement
  only ~50% (NoMem confusion GT+/rm-: 22, GT-/rm+: 29), far below the paper's 72.7%. Main false-negative pattern: the
  autoeval sees the page *before* the final action (e.g. search/navigation tasks look 'not done'). Paper code
  behaviour, kept unchanged for comparability.
- Resume (2026-10-12 ~21:10): both sites recreated fresh from shopping.sqsh (so state edits made by earlier tasks are gone ->
  later tasks start from the original DB, which is the standard WebArena assumption anyway). Servers relaunched with
  setsid/nohup, identical args. Partial dirs for the interrupted tasks (335, 284) moved to results/_discarded.
  Same commands + `--prev_id 334` (NoMem) / `--prev_id 283` (RB); memory bank continues from its 78 items.
- GT-eval audit (2026-10-12 ~21:30, `audit_answers.py` -> results/audit_{nomem,rb}.jsonl), NoMem 45 failures read by hand:
  - genuine agent errors (wrong numbers, incomplete lists, wrong product/category/filter, stayed on list page instead of
    product page for url_match, 30-step timeout): ~34
  - 'N/A' tasks where the agent answered something instead of saying it doesn't exist: 4 (166, 191, 235, 301)
  - questionable reference: 330 (ref 81.31 but task 332's ref says Mar = 83.31, agent said 83.31); 50 (count 21 vs ref 24,
    total $6560.69 exact). Strict must_include on quoted sentences: 167.
  - HARNESS BUG: model's think mentions the literal string "<action>", parse_html_tags grabs the first <action> ->
    action = think text + real action -> step fails. NoMem 2 tasks (165, 226), RB 4 (158, 163, 264, 270).
  - fuzzy_match positives spot-checked (47, 48, 49, 96, 117, 144, 146, 22, 24): all correct.
  Conclusion: GT evaluator is working; SR is trustworthy within ~2-3 pts. Parser bug left as is for this run (both arms,
  fixing mid-run would confound the comparison); fix (parse action only after the last </think>) before the next runs.
- Actual: STOPPED by user request (2026-10-12 ~21:40). Final partial: NoMem 107/187 SR 56.1 steps 7.35; RB 87/187 SR 56.3 steps 7.18 (judge agree 46%).
  On the 80 shared tasks at the last paired count: NoMem 53.8 vs RB 56.2 (+2.4, n.s.). Not reproduced (no significant gain, no step reduction).

#### Diagnosis at 82 RB tasks (2026-10-12 ~21:35) — why no gain so far
- Memory injection verified (retrieved items appear in the system prompt with the paper's instruction).
- Memory bank labels vs GT: judged success & GT fail 27, success & success 28, fail & success 17, fail & fail 9 -> 54% mislabeled.
- Retrieved top-1 memory: 49/80 from the same intent template (mostly the immediately preceding sibling task).
  61% of retrieved memories (49/80) are mislabeled; 39/80 are "success strategies" distilled from GT-failed runs.
- Paired NoMem vs RB SR by slice: same-template 53.1 vs 49.0 (n=49); diff-template 61.3 vs 67.7 (n=31);
  label-correct 77.4 vs 80.6 (n=31); label-wrong 42.9 vs 40.8 (n=49). Losses e.g. 145 <- memory from 141 judged success but GT-failed.
- Interpretation: the self-judge (Qwen3.8-27B, ~50% agreement; it sees the page *before* the final action) feeds wrong
  strategies into memory, and same-template retrieval copies the sibling's mistake. Small n, noisy slices.

### shop_rbgt_r2 — ReasoningBank with GT-labelled memory (oracle judge; diagnostic upper bound)   (2026-10-12 ~21:40, fs-mbz-gpu-388 ×8)
- Reflection/audit: NoMem vs RB (self-judge) shows +2.4 on 80 shared tasks (noise). GT evaluator audited and OK.
  Main suspect = self-judge mislabeling 54% of memories. Paper Fig. (simulated judge accuracy) shows 100% judge -> 52.4.
  This run isolates the judge: identical pipeline, only `--judge gt` (paper code option: induce_memory --criteria gt).
- Goal/criteria: is the method itself effective with Qwen3.8-27B when labels are correct? Compare to NoMem on the same 187 tasks.
  Not a deployable setting (uses GT labels for memory of *past* tasks only; current task's GT never seen before acting) —
  diagnostic only, reported as such.
- Command (separate copy of WebArena dir so memory files don't collide; instance shop_c on localhost:7772):
  `cd reasoning-bank/WebArena_gt && WA_SHOPPING=http://localhost:7772 python pipeline_memory.py --website shopping --model qwen3.8-27b --memory_mode reasoningbank --judge gt --output_dir $RB_ROOT/results/shop_rbgt_r2`
  Same harness as r2 (parse bug not fixed, for comparability). Shares the agent vLLM server (3 concurrent pipelines).
- Predicted: RB-GT SR ≈ NoMem + 5 (≈ 61 vs 56), steps slightly fewer. If RB-GT ≤ NoMem + 2, the method itself gives
  little on this strong backbone (headroom problem, not judge problem) -> FALSIFIES "judge is the bottleneck".
- Actual: STOPPED at 6/187 (SR 66.7) by user request; no conclusion.

#### Self-judge fix (WebArena_v2, 2026-10-12 ~21:40)
- Bugs in released autoeval/evaluate_trajectory.py: (1) the 'final page state' given to the judge is the page *before*
  the last action (the post-action obs in the last step file was dropped); (2) extract_response cut the answer at the
  first ')' and took the first '(' of a multi-action string. Agent parser bug: '<action>' written inside <think> was
  parsed as the action -> now parse only after the last </think> (dynamic_prompting.py).
- Offline re-judge of 85 r2 trajectories with Qwen3.8-27B: original 51.8% -> fixed 63.5% agreement with GT (FP 24, FN 7).
  (397B judge attempt failed: prompt > its 32k context; not pursued.)

### shop_nomem_q32 + shop_rb_q32 — Qwen3-32B, No-Memory vs ReasoningBank, full Shopping (187)   (2026-10-12 ~21:55, fs-mbz-gpu-388 ×8)
- Reflection/audit: with Qwen3.8-27B, RB showed no significant gain (+2.4 on 80 shared tasks), traced to a ~50%-accurate
  self-judge (54% of memories mislabeled) plus a strong baseline (~56 SR, above the paper's Gemini+RB 49.7). User asked
  to stop and reproduce on Qwen/Qwen3-32B — an older/weaker agent, closer to the paper's regime (more headroom).
- Goal/criteria: reproduce the paper's direction on Shopping: RB SR - NoMem SR >= +3 and RB avg steps < NoMem avg steps;
  GT reward from the (audited) WebArena evaluator. Streaming test-time protocol; memory built only from past tasks'
  self-judged trajectories (no GT in memory) -> no leakage.
- Changes vs r2: backbone Qwen3-32B (vLLM 0.31, TP2 on GPU0-1, YaRN x4 -> 131072 ctx as recommended by Qwen for long
  inputs, reasoning parser qwen3, thinking on); code = WebArena_v2 (judge fixes + think/action parse fix, applied to both arms);
  fresh site instances from shopping.sqsh (A: 7770 RB, B: 7771 NoMem); fresh memory bank. Everything else identical
  (temp 0.7, 30 steps, AX tree, top-1 Qwen3-Embedding-8B retrieval, 397B GT grader).
- Smoke (task 117): 6 steps, ~2.5 min incl. judge+induction; answered 5/5/22 (GT 3/2/22) -> reward 0; fixed judge correctly
  said fail citing the final page; memory item well-formed. Pipeline OK.
- Commands (cwd reasoning-bank/WebArena_v2, `source wa_env_32b.sh`):
  - NoMem: `WA_SHOPPING=http://localhost:7771 python pipeline_memory.py --website shopping --model qwen3-32b --memory_mode no_memory --output_dir $RB_ROOT/results/shop_nomem_q32`
  - RB:    `WA_SHOPPING=http://localhost:7770 python pipeline_memory.py --website shopping --model qwen3-32b --memory_mode reasoningbank --judge autoeval --output_dir $RB_ROOT/results/shop_rb_q32`
- Predicted: NoMem SR ≈ 35 (25-45; between Gemma-12B 17 and Gemini-flash 39), avg steps ≈ 9; RB ≈ +5 (≈ 40), steps -1.
  Self-judge agreement ≈ 65-70%. ETA ~2.5-3 min/task -> ~8 h.
  FALSIFIED if RB - NoMem <= 0 or RB uses more steps.
- Actual (NoMem DONE 2026-10-13 ~12:00): shop_nomem_q32 187/187 SR 32.6, avg steps 10.64 (1 browser-crash error). RB/RBX still running; see later entries.

#### Stronger self-judge test (2026-10-12 ~23:45): Qwen3.5-397B as judge, fixed autoeval (judge_offline.py, judge_compare.py)
- 397B server restarted with max-model-len 131072 (GT grader retry extended to wait across restarts; running q32 tasks resumed fine).
- Offline re-judge of 180 Qwen3.8-27B trajectories: original backbone judge 49.4% -> 397B fixed judge 59.4% (FP 54, FN 19).
  => a bigger *transcript-only* judge does not fix it. FP dominate: plausible-but-wrong answers (wrong counts/totals,
  incomplete lists, wrong product URL) cannot be detected from the trajectory alone; the judge would have to re-verify on the site.

### shop_nomem_397 — Qwen3.5-397B as the *agent*, No-Memory, full Shopping (187)   (2026-10-12 ~23:50, GPU4-7 shared with grader)
- Why: user wants a stronger checker and does not care where skills come from. An independent strong agent run gives
  (a) a cross-check signal for the 32B agent's answers (agreement of two independent solvers, no GT), and
  (b) teacher trajectories to induce memory from. Also gives the 397B's own SR on this env.
- Command (cwd reasoning-bank/WebArena_v3; AGENT_LLM_URL=:8002, AGENT_LLM_MODEL=qwen3.5-397b, AGENT_TOKENIZER=Qwen/Qwen3.5-397B-A17B-FP8):
  `WA_SHOPPING=http://localhost:7772 python pipeline_memory.py --website shopping --model qwen3.5-397b --memory_mode no_memory --output_dir $RB_ROOT/results/shop_nomem_397`
  Instance shop_c (fresh from sqsh), same harness/params (temp 0.7, 30 steps, AX tree).
- Predicted: 397B SR ≈ 55-65 (above Qwen3.8-27B's ~56). If it agrees with the 32B agent's answer, P(32B correct) should be
  ≥85% (vs 32B self-judge ~70%).
- Actual: DONE 2026-10-13 04:4x. SR 51.3 (187/187), avg steps 5.62, 1 task errored (166, parse failure after 6 retries).
  Used as the cross-check reference for shop_rbx_q32 (86.4% label accuracy offline).

#### Cross-check verifier (2026-10-13 ~04:15, verify_cross.py): 32B outcome vs independent 397B agent outcome (397B, no thinking, SAME/DIFFERENT)
- On 206 q32 trajectories (NoMem+RB) with a 397B reference run: cross-check acc 86.4% (FP 17, FN 11, precision 76.1, recall 83.1)
  vs 32B self-judge 62.1% (FP 57, FN 21). SAME AND self-judge 80.1%; SAME OR self-judge 68.4%. -> use cross-check alone.
- Status at 04:03: q32 shared 83 tasks NoMem 31.3 vs RB 33.7 (+2.4, W/L 12/10, p=0.42); 397B NoMem 155/187 SR 52.9.

### shop_rbx_q32 — ReasoningBank, Qwen3-32B agent, cross-check judge + 397B memory inducer   (2026-10-13 ~04:20, fs-mbz-gpu-388 ×8)
- Reflection/audit: paper-faithful RB on Qwen3-32B: early +8 faded to +2.4 at 83 shared tasks (n.s.); self-judge ~60-64%
  -> ~40% of memories mislabeled. Transcript-only 397B judge only 59%. Cross-check with an independent 397B agent run: 86%.
  User: "I don't care where the skills come from" -> stronger labels + stronger memory writer is in scope.
- Change vs shop_rb_q32: label = agreement with the 397B agent's run of the same task (no GT; the reference run is a
  separate No-Memory agent run, shop_nomem_397); memory induced by Qwen3.5-397B from the 32B trajectory, with the
  reference agent's answer in the induction context. Agent, retrieval, harness, prompts identical (WebArena_v3).
- Leakage: no GT anywhere in memory or decisions; only past tasks' memories are retrieved (streaming). Note the reference
  answer for task t is computed from an agent run on task t, but it only enters memory *after* the 32B agent finished t.
- Command (cwd reasoning-bank/WebArena_v3, source wa_env_32b.sh, instance shop_d on :7773):
  `WA_SHOPPING=http://localhost:7773 python pipeline_memory.py --website shopping --model qwen3-32b --memory_mode reasoningbank --judge autoeval --judge_model qwen3.5-397b --cross_ref $RB_ROOT/results/shop_nomem_397 --output_dir $RB_ROOT/results/shop_rbx_q32`
- Compared against shop_nomem_q32 (same agent, same params, fresh instance).
- Predicted: RBX - NoMem ≈ +6 to +8 (≈ 37-39 vs ~31), label accuracy ≈ 85%. FALSIFIED (memory gives little even with
  good labels) if RBX - NoMem <= +2 over >= 120 shared tasks.
- Actual: (pending)
- Note: shop_nomem_q32 task 510 errored (Chromium 'Target crashed' during obs extraction) -> scored 0, pipeline continued.
- Note: shop_rb_q32 task 387 errored ('Could not parse a valid value after 6 retries': agent output format failure) -> scored 0, pipeline continued.

#### Interruption 2 (2026-10-13 08:48): SLURM job ended -> all processes + /tmp site data lost (node fs-mbz-gpu-388 -> new job on fs-mbz-gpu-066)
- State: nomem_q32 182/187 (last 793), rb_q32 131/187 (last 437), rbx_q32 78/187 (last 283), nomem_397 done.
- Resume (~10:55): servers relaunched with identical args; sites recreated fresh from shopping.sqsh (A 7770 RB, B 7771 NoMem,
  D 7773 RBX); partial task dirs (794, 438, 284) -> results/_discarded; dropped the interrupted task's query embedding
  from each retrieval cache (else it retrieves itself and gets no memory); memory banks unchanged (v2: 131, v3: 78).
  Same commands + --prev_id 793 / 437 / 283.

## Standard eval pipeline (2026-10-13 ~11:40): eval_pipeline/
- Frozen code (WebArena_v3 + fixes), fresh site per run, template-level split (seed 0): train 101 tasks / 24 templates,
  test 86 tasks / 24 templates (splits/). tools/report.py = SR, steps, paired sign test. Smoke-tested no_memory + reasoningbank.
- Existing runs restricted to TEST split: NoMem q32 13.1 (84/86); RB q32 28.1 (64/86; paired n=64: 17.2 -> 28.1, W/L 10/3, p=0.046);
  RBX q32 12.1 (33/86); 397B 33.7.

### lora_r1 — LoRA SFT of Qwen3-32B on TRAIN-split successful trajectories; eval on TEST split   (2026-10-13 ~12:00, fs-mbz-gpu-066, train GPU3)
- Reflection/audit: RB on Qwen3-32B gives +3.0 overall (n.s.) but +10.9 on test templates (p=0.046, n=64). User wants
  LoRA fine-tuning compared against RB. Fair protocol: LoRA trains only on TRAIN templates (no test task/template seen),
  evaluated on the 86 TEST tasks; RB/NoMem compared on the same tasks.
- Data (lora/build_sft_data.py): no-memory runs (397B, Qwen3-32B, Qwen3.8-27B) on train tasks, trajectories with GT
  reward>0 (rejection-sampling SFT; GT of TRAIN tasks only), drop memory prompts / malformed / parse-retry / >24576 tok.
  -> 574 steps (397B 235 / 27B 205 / 32B 134), 74 of 101 train tasks, 4.27M tokens.
  NOTE: RB uses no GT at all (self-judge, streaming on test tasks); LoRA uses GT on train tasks. Different supervision.
- Train (lora/train_lora.py): LoRA r16 a32 dropout .05 on q,k,v,o,gate,up,down; lr 1e-4 cosine, warmup 5%, 2 epochs,
  16 seqs/step, bf16, grad ckpt, loss on assistant turn only, single H200 (GPU3), seed 0.
- Eval: eval_pipeline run_eval.sh --method no_memory --split test, LoRA served via vLLM --enable-lora; plus a fresh
  base-Qwen3-32B no_memory run through the same pipeline (same-pipeline baseline + run-to-run variance check).
- Predicted: base test SR ≈ 13-17; LoRA ≈ 18-22 (+4-6, templates are new so transfer is limited); RB(test) 28.1 stays
  higher. FALSIFIED (LoRA worthless here) if LoRA <= base; LoRA beats RB if LoRA > 28 on test.
- Actual (DONE 2026-10-14 08:34; all on the 86 TEST tasks, paired vs the new-pipeline base run, same tasks):
  base Qwen3-32B 16.3 (13.6 steps) | LoRA step36 16.3 (+0.0, W/L 4/4, 15.8 steps) | LoRA final 12.8 (-3.5, W/L 3/6, 15.1 steps)
  | RB (32B self-judge) 20.9 (+4.7, W/L 9/5, p=0.21) | RBX 14.0 (-2.3) | old NoMem rerun 12.8 (-3.5 = run-to-run noise).
  Without the 2 site-fact-leak tasks (117,150; 84 tasks): LoRA s36 -1.2, LoRA final -4.8, RB +4.8.
  LoRA vs RB directly: s36 -4.7 (RB wins 10/6), final -8.1 (RB wins 11/4).
  Prediction (LoRA 18-22, +4-6 over base) -> FAIL: LoRA <= base (falsification condition met). Early +16 at n=24 was
  task-order luck (Switch-storage family solved early). Final adapter overfit: repetition loops (8192-token degenerate
  generations, repeated clicks), worse than base. Note LoRA used GT on TRAIN tasks + 397B teacher; RB used no GT.
  Conclusion: on unseen templates, this LoRA (574 steps, 74 train tasks) does not beat the base model; RB has the best
  point estimate but no method differs from base beyond single-run noise (~±4-5 pts on 86 tasks).

- Note (12:25): test_base_q32 runs on site 'smoke_site' (7774; only read-only smoke tasks 96/117 had run on it). A second site 'site_base' was mistakenly created on the same offset and never bound its ports; removed. start_site.sh now refuses an occupied offset.
- lora eval: lora_r1 step18 checkpoint served on GPU2 (port 8003, vLLM --enable-lora, runtime LoRA loading on) next to the embedder (restarted at 12:58 with gpu-mem 0.15; RB pipelines paused during restart, no task affected). Eval run eval_pipeline/runs/test_lora_s18 on site :7775, test split.
- 13:30 test_lora_s18 discarded (vLLM max-model-len 65536 < requested max_tokens 65536 -> every request 400; my config bug). LoRA server restarted with 131072 ctx; now evaluating step36 checkpoint (50% of training) as runs/test_lora_s36 on site :7775.
- 15:00 lora_r1 training DONE (72 steps, final loss 0.43, 177 min). Final adapter served on GPU3 :8004 (131k ctx); eval runs/test_lora_final on site :7777 (offset 6 port 9306 occupied by a foreign process). Note: same-model rerun test_base_q32 vs shop_nomem_q32 differ by +6.2 on 48 shared test tasks -> single-run noise is large.

#### Interruption 3 (2026-10-13 17:05): SLURM job ended again (node fs-mbz-gpu-066 -> fs-mbz-gpu-181), all processes + /tmp sites lost.
- State: rb_q32 170/187 (last 589), rbx_q32 142/187 (last 508), test_base_q32 63/86, test_lora_s36 24/86, test_lora_final 2/86.
- test_lora_final finding: on task 146 the final adapter fell into repetition loops; 3 single generations ran to the 65536-token
  cap (~30-60 min each at ~38 tok/s), outputs unparseable -> retries; actions also looped (click('1619') x9). Saved steps are
  short (p50 175 tok), so it is intermittent but costly.
- Decision: eval_pipeline gets a per-step generation cap (run_eval.sh --max_new_tokens, env AGENT_MAX_NEW_TOKENS). Base
  Qwen3-32B never exceeded 5,657 tokens/step in 509 test steps, so a cap of 8192 does not change base behaviour; apply 8192
  to the LoRA evals (s36 + final) from now on. Partial task dirs moved to results/_discarded; embedding caches trimmed.
- restart_all.sh: one command to bring back all servers (GPU0-1 32B :8000, GPU2 embed :8001, GPU3 32B+LoRA s36/final :8004,
  GPU4-7 397B :8002) and the 5 sites.

#### Leakage audit (2026-10-13 ~22:40)
- LoRA SFT data: 0 test task ids, 0 test templates (asserted in build_sft_data.py). Most similar test/train templates are
  different questions (max Jaccard 0.53: "status of my latest order" vs "total cost of my latest {status} order").
- Test reference strings inside SFT targets: 4 hits. 2 are coincidental ("Amazon Customer" = generic reviewer name).
  2 are SITE FACTS shared through the same user account: '3/2/22' (test 117, first purchase date) appears in train task 145's
  reasoning while paging order history; '$260.69' (test 150) appears in train task 143's reasoning. -> mild environment-level
  leakage for tests 117 and 150. LoRA s36 solved 117 (base_q32 new run failed it, old NoMem solved it), failed 150.
  Report LoRA numbers with and without tasks 117/150.
- ReasoningBank (rb_q32): no GT anywhere; memory = self-judged distilled strategies from earlier tasks (incl. earlier TEST
  tasks, by design of the method, as in the paper). Same site-fact caveat applies to memory text.
- RBX: memory text includes the independent 397B agent's answer to the same task (added after the 32B finished it) ->
  later sibling test tasks can see a stronger model's answer for a sibling (e.g. 158/160/161 share one reference URL). Not GT,
  but teacher knowledge on test tasks; RBX did not outperform anyway.
- Split fixed with seed 0 before LoRA training, not tuned on results.
- 22:50 Moved everything to a batch job (run_all.sbatch: account/qos deploy, partition main, 1 node x8 GPU, exclusive) = restart_all.sh + resume_runs.sh, keeps allocation until all pipelines end; resubmit to resume after any interruption. Interactive-node servers stopped.

#### TEST split, complete for base + RB (2026-10-14 01:31)
- test_base_q32 (new pipeline, Qwen3-32B no memory) 86/86: SR 16.3, steps 13.64.  Old NoMem run on the same 86: 12.8 (-3.5 vs new run; same model = run-to-run noise).
- shop_rb_q32 (paper RB, 32B self-judge) on the 86 test tasks: 20.9 -> paired vs new base +4.7 (W/L 9/5, p=0.21). Not significant.
- Partial at this time: lora_s36 41/86 +7.3 vs base (W/L 4/1, p=0.19); lora_final 19/86 -10.5 (W/L 1/3); rbx 82/86 -6.1.
- FINAL full-187 (paper protocol, streaming over all shopping tasks): NoMem 32.6 (10.64 steps) vs RB (32B self-judge) 34.2
  (12.14 steps), self-judge acc 71.1%. Prediction was RB ≈ +5 with fewer steps -> FAIL on both (gain +1.6, n.s.; +1.5 steps).
- 02:20 Bottleneck found: the GPU3 LoRA server serves two adapters (s36, final) but vLLM --max-loras defaults to 1, so the two
  evals were serialized (one waits while the other runs an 8192-token degenerate generation). rb/rbx/base all finished
  (rbx full 187: 33.2 vs NoMem 32.6). Moved test_lora_final to its own batch job 2675774
  (eval_pipeline/scripts/eval_lora_job.sbatch: adapter TP2 on GPU0-1, fresh site, --resume; the partial task 239 was
  discarded and reruns). Adapter is now served under the name 'test_lora_final' (same weights as 'lora-final').
  test_lora_s36 keeps GPU3 alone in job 2673676.
- 03:10 self-judge (autoeval) of lora_s36 task 331 ran ~30 min: LOCAL_Client.chat had no max_tokens, the looping adapter generated until context. eval_pipeline/code/utils/local_llm.py: self-judge now uses the same AGENT_MAX_NEW_TOKENS cap (only affects the diagnostic self-judge, not GT).

### lora_v2 — LoRA with filtered data, 1 epoch; eval on the same 86 TEST tasks   (2026-10-14 ~10:45, batch job, 1 node x8)
- Reflection/audit: lora_r1 failed (s36 = base 16.3; final 12.8, looping). Diagnosis: (1) 36% of SFT steps were Qwen3.8-27B
  traces with a telegraphic "We need ... maybe ..." style (2.44 hedge markers/100 words); the final adapter's outputs reached
  2.02 vs base 0.43 -> it copied the style and looped. (2) only 574 steps / 74 tasks, 2 epochs -> overfit. (3) the template
  split is uneven in difficulty (397B: train 66% vs test 34%; Qwen3-32B: 49.5% vs 12.8%) — kept fixed to stay comparable
  with the existing base/RB test runs.
- Changes: drop 27B traces; add 3 extra Qwen3.5-397B no-memory runs on the TRAIN split (temp 0.7) for more clean teacher
  successes; keep Qwen3-32B successes only if hedge <= 1.0/100 words and reply <= 2000 tokens; dedupe; 1 epoch;
  4-GPU data parallel (lora/train_lora_ddp.py), same LoRA config (r16 a32, all linear, lr 1e-4 cosine, 16 seqs/step).
  Dry-run without extra runs: 360 steps (397B 235 / 32B 125). Expected with extra runs ≈ 1000 steps.
- Command: `sbatch lora/lora_v2.sbatch` (phases A data -> B build -> C train -> D eval; resumable).
  Eval: no_memory, test split in 3 shards on 3 fresh sites, --max_new_tokens 8192 (same as v1 evals).
- Predicted: test SR ≈ 18-21 (+2-5 over base 16.3), no looping (hedge rate ≈ base), steps ≈ base.
  FALSIFIED if SR <= 16.3 (no gain) or hedge rate > 1.0 again.
- Actual (job 2679508 done 2026-10-14 16:49; audited 2026-10-18): SFT 919 steps (397B 235+209+195+155, 32B 125),
  6.48M tokens, 58 steps, loss 0.47 -> 0.25, 41 min. TEST 86: 10.5 (vs base 16.3, W/L 4/9), steps 11.6, hedge 0.03/100w.
  INVALID: all six sites of this job (data collection 7771-7773, eval 7781-7783) had broken product search (100% of 437
  search pages "no results"; earlier runs: 0-3 of 186-236). Root cause: the image uses Magento's OpenSearch module;
  start_site.sh set only elasticsearch7_server_port, so every site queried localhost:9200 (= the offset-0 site, present in
  all earlier jobs, absent here). start_site.sh also hid the failure (set -e exit before the search check, masked by
  `| tail -1`). Both the teacher data and the eval are contaminated -> no conclusion about LoRA v2. Fix verified by hand
  (config:set opensearch_server_port -> 19120 hits); repo version of start_site.sh/run_eval.sh now refuse broken sites.
  Earlier experiments are unaffected (the product index is read-only and identical; search worked).

