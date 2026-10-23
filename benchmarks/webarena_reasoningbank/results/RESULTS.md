# Results — WebArena-Shopping: ReasoningBank vs LoRA (Qwen3-32B)

All numbers are single runs (agent temperature 0.7, max 30 steps, AX-tree observation, thinking on). Success = official
WebArena reward. "Paired" = same tasks only; p = one-sided exact sign test on tasks where exactly one run succeeded.
**Run-to-run noise of one model is ≈ ±4–5 points** on 86 tasks (the same Qwen3-32B run twice: 16.3 vs 12.8).

## 1. ReasoningBank reproduction (paper protocol: streaming over all 187 shopping tasks)

| Agent | No memory | ReasoningBank | Δ (W/L, p) | Steps NoMem → RB |
|---|---|---|---|---|
| Paper: Gemini-2.5-flash | 39.0 | 49.7 | +10.7 | 8.2 → 6.1 |
| Paper: Gemma-3-12B | 17.1 | 24.1 | +7.0 | 13.7 → 11.8 |
| **Qwen3-32B** (self-judge = 32B) | 32.6 | 34.2 | +1.6 (18/15, p=0.36) | 10.6 → 12.1 |
| Qwen3-32B, judge = cross-check vs Qwen3.5-397B | 32.6 | 33.2 | +0.5 (20/19) | 10.6 → 10.3 |
| Qwen3.8-27B (stopped early; 80 shared tasks) | 53.8 | 56.2 | +2.4 | – |
| Qwen3.5-397B, no memory (reference / teacher) | 51.3 | – | – | 5.6 |

Not reproduced: no significant gain, and ReasoningBank used *more* steps. Self-judge accuracy (agreement with the
official reward) is the main lever of the method:

| Judge | Accuracy |
|---|---|
| released code, Qwen3.8-27B judging itself | 49–52% (it judged the page *before* the last action, and truncated answers at the first ")" — fixed in `patches/`) |
| fixed judge, Qwen3-32B judging itself | 62–71% |
| fixed judge, Qwen3.5-397B reading the transcript | 59% |
| cross-check: same answer as an independent Qwen3.5-397B run | 86% |

Better labels did not translate into gains: the memories are sensible strategies ("sort by price, check all pages"),
but Qwen3-32B mostly fails at *executing* them (dropdowns, pagination, arithmetic).

## 2. ReasoningBank vs LoRA on unseen templates (TEST split: 86 tasks / 24 templates)

Split by `intent_template_id` (seed 0, `splits/`): train 101 tasks / 24 templates, test 86 tasks / 24 templates.
LoRA trains on TRAIN templates only; ReasoningBank learns online over the test stream without labels.

| Method | Success | vs base (W/L, p) | vs ReasoningBank | Avg steps |
|---|---|---|---|---|
| Qwen3-32B base | 16.3 | – | – | 13.6 |
| **ReasoningBank** (32B self-judge) | **20.9** | **+4.7** (9/5, 0.21) | – | 14.8 |
| ReasoningBank, 397B cross-check judge | 14.0 | −2.3 (5/7) | | 13.3 |
| LoRA v1, half-trained (step 36/72) | 16.3 | ±0 (4/4) | −4.7 (6/10) | 15.8 |
| LoRA v1, final (2 epochs) | 12.8 | −3.5 (3/6) | −8.1 (4/11) | 15.1 |
| LoRA v2 (filtered data, 1 epoch) | 10.5 | **invalid** — see below | | 11.6 |

Without the two tasks whose answer appears as a site fact in TRAIN trajectories (117, 150; `splits/test_noleak_ids.txt`):
LoRA v1-s36 −1.2, LoRA v1-final −4.8, ReasoningBank +4.8.

**LoRA v1** (574 steps from 74 train tasks: Qwen3.5-397B 235 / Qwen3.8-27B 205 / Qwen3-32B 134 successful steps;
r16, all linear layers, lr 1e-4, 2 epochs): no gain. The final adapter overfit to the Qwen3.8-27B traces' telegraphic
"We need … maybe …" style (2.02 hedge markers / 100 words vs 0.43 for the base model; 2.44 in those traces) and fell
into repetition loops (8k-token degenerate generations, repeated clicks).

**LoRA v2** (27B traces dropped, 3 extra Qwen3.5-397B runs on train, style filter, dedup → 919 steps; 1 epoch, 4 GPUs)
scored 10.5 but the run is **invalid**: every site it used (data collection *and* evaluation) had broken product search
(100% "no results"). Cause: the site image uses Magento's OpenSearch module, and the site script only set
`elasticsearch7_server_port`, so every site queried `localhost:9200` = the offset-0 site; v2 was the first job without
an offset-0 site. All earlier runs had working search (0–3 empty searches out of 186–236). Fixed in
`scripts/start_site.sh` (sets `opensearch_server_*` and refuses to start unless a search returns 19,120 hits) and
`scripts/run_eval.sh` (refuses to evaluate on a site with broken search). LoRA v2 has to be re-collected and re-evaluated.

## 3. Caveats
- One run per method; differences under ~5 points are within run-to-run noise.
- The template split is uneven in difficulty (Qwen3.5-397B: train 66% vs test 34%; Qwen3-32B: 49.5% vs 12.8%).
- LoRA used official rewards on TRAIN tasks to select trajectories plus a 397B teacher; ReasoningBank used no labels.
- Grader for fuzzy-match answers: Qwen3.5-397B-A17B-FP8 instead of the paper's gemini-2.5-pro. Site: WebArena-Verified
  shopping image (same data as the original).
- LoRA evals cap generation at 8,192 tokens/step (base Qwen3-32B never exceeded 5,657 tokens/step in 509 test steps).

Raw per-run numbers: `final_test.json`. Full chronological log with every run spec, prediction and incident:
`EXPERIMENT_LOG.md`.

## Completed 7B / 8B reproduction

See [full 187-task results and audit](qwen-7b-8b-187-20261022/README.md).
