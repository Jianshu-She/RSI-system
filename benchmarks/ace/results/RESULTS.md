# Results: ACE reproduction with Qwen3-32B (plus LoRA context distillation)

Paper: *Agentic Context Engineering* (arXiv 2510.04618). The paper uses DeepSeek-V3.1 (671B, non-thinking). We use
**Qwen3-32B**, thinking off, temperature 0, YaRN ×4 (131k context), vLLM 0.31. Runs date from 2026-10-12 to 2026-10-14.
All numbers come from result files (`eval/report.py` → [`report_output.txt`](report_output.txt)). The LoRA tables come
from `finetune/make_lora_catalog.py` → [`lora_catalog/`](lora_catalog/README.md). The full run log, with predictions
and incidents, is `EXPERIMENTS.md` in the source project.

Paired tests are two-sided sign tests on per-item correctness, using the same test items in the same order
(`+a/−b` means a items only this method gets right and b items only the comparison gets right). Each run is one temp-0 run.
Identical temp-0 runs differ by up to about 1.5 pts because vLLM batching is not deterministic, so treat any gain below about 2 pts as noise.

## Additional model reproduction

The completed [Qwen2.5-7B-Instruct Finance reproduction](qwen25-7b-repo-20261022/README.md) reports all eight ACE offline/online results, repository-default configuration, comparison limits, and result-file hashes. It is separate from the Qwen3-32B results below.

## 1. Finance (Table 2): test accuracy %

FiNER: 441 test items, scored per tag. Formula: 200 test items, exact float match. Base = the same model with an empty playbook, taken from each ACE run's initial test and averaged over 4 runs (Formula 69.5/69.0/69.0/68.0, FiNER 62.5/63.5/63.1/61.9).

| Method | FiNER ours | FiNER paper | Formula ours | Formula paper |
|---|---|---|---|---|
| Base LLM | 62.8 | 70.7 | 68.9 | 67.5 |
| ICL (offline; Formula all 500 demos, FiNER 65 demos = window full) | 70.4 | 72.3 | 82.0 | 67.0 |
| ACE offline + GT | **72.6** | 78.3 | **86.5** | 85.5 |
| ACE offline, no GT | 65.8 | 71.1 | 81.0 | 83.0 |
| ACE online + GT | 71.0 | 76.7 | 75.5 | 76.5 |
| ACE online, no GT | 60.8 | 67.3 | 79.5 | 78.5 |
| DC-CU + GT (online) | 58.9 | 74.2 | 77.0 | 69.5 |
| DC-CU, no GT (online) | 57.0 | 68.3 | 80.0 | 62.5 |

Paired tests:
- **Formula offline +GT** vs its run's base: +35/−1 (p<1e-8). vs ICL: +12/−3 (p=0.035). The +GT result is significantly above ICL.
- Formula offline noGT vs base: +36/−12 (p=0.0007). vs ICL: +10/−12 (p=0.83), so the two are tied.
- Formula online, scored only on the windows where the playbook is active (windows 2–14, 185 items): +GT scores 140 vs the base's 128. noGT scores 148 vs 125.
- Formula DC +GT vs base: +26/−10 (p=0.011). ACE online +GT vs DC: 26 vs 29 (p=0.79). DC noGT vs base: +29/−5 (p<1e-4). ACE online noGT vs DC: 19 vs 20 (p=1). ACE and DC are tied on Formula, but DC gets Python code execution and ACE does not.
- **FiNER offline +GT**, sample level (all 4 tags correct): vs base +91/−30 (p=2.5e-8). vs ICL +61/−47 (p=0.21, not significant).
- FiNER offline noGT vs base: +74/−47 (p=0.018). vs ICL: +44/−62 (p=0.098).
- FiNER online +GT: +7.9 over its run's base (63.1). Online noGT: −1.1 vs its run's base (61.9). The paper also reports online noGT FiNER below base.
- FiNER DC +GT vs base: +45/−54 (p=0.42). ACE online +GT vs DC: 81 vs 31 (p=2.5e-6), so ACE wins clearly. DC noGT vs base: 26 vs 46 (p=0.025), so DC significantly hurts FiNER.

Offline selection uses the best playbook on val, never test. Val traces:
- Formula +GT: 81.3 / 88.3 / **89.7** / 87.3 / 81.3 (steps 100–500).
- Formula noGT: best 83.0 at step 200.
- FiNER +GT: best 75.8 at step 300, then a plateau at 74.0.
- FiNER noGT: best 69.6 at step 700.
- Val baselines: base Formula 72.0, FiNER 62.0. ICL Formula 84.7, FiNER 70.5.

**Verdict on finance.** The Formula column reproduces the paper closely (offline 86.5/81.0 vs 85.5/83.0; online 75.5/79.5 vs 76.5/78.5).
On FiNER, ACE gives a similar or larger gain over our (weaker) base (+9.8 offline +GT vs the paper's +7.6), but the
absolute numbers are 5–6 pts below the paper. The ACE-vs-ICL gap on FiNER is not significant. Our ICL baseline is much
stronger than the paper's on both tasks (Formula ICL 82.0 vs the paper's 67.0).

## 2. AppWorld (Table 1): TGC / SGC %

Test splits: test_normal (168 tasks) and test_challenge (417 tasks). Headline = mean of the 4 numbers.

| Method | normal TGC/SGC | challenge TGC/SGC | avg ours | avg paper | paired vs ReAct (task success): normal; challenge |
|---|---|---|---|---|---|
| ReAct (shipped initial playbook) | 44.6 / 26.8 | 26.6 / 10.8 | 27.2 | 42.4 | — |
| ACE offline + GT | 53.0 / 33.9 | 34.5 / 19.4 | **35.2** | 59.4 | +30/−16 p=0.054; **+61/−28 p=0.0006** |
| ACE offline, no GT | 45.8 / 28.6 | 29.7 / 15.1 | 29.8 | 57.2 | +20/−18 p=0.87; +48/−35 p=0.19 |
| ACE online (no GT) | 47.0 / 28.6 | 31.9 / 18.0 | 31.4 | 59.5 | +24/−20 p=0.65; **+54/−32 p=0.023** |
| LoRA (ACE-distilled trajectories) + ReAct | 46.4 / 26.8 | 29.0 / 12.2 | 28.6 | — | +21/−18 p=0.75; +38/−28 p=0.27 |

Paper per-cell numbers (DeepSeek-V3.1):
- ReAct: 63.7/42.9, 41.5/21.6.
- ACE offline +GT: 76.2/64.3, 57.3/39.6.
- ACE offline noGT: 75.0/64.3, 54.4/35.2.
- ACE online: 69.6/53.6, 66.0/48.9.

**Verdict on AppWorld.** The direction reproduces: ACE+GT gains +8.0 avg (significant on challenge), and online ACE gains +4.2
(significant on challenge). The size of the gain is about ¼ to ½ of the paper's (+17). Without labels, offline ACE has no
significant effect (+2.6). The Qwen3-32B ReAct base is 15 pts below DeepSeek-V3.1.

## 3. LoRA SFT: moving the playbook into the weights (context distillation)

Data comes from TRAIN splits only. The input is the base prompt with an empty playbook `(empty)`. The target is the
first-attempt answer that ACE produced *with* its learned playbook, kept only if correct. Recipe: LoRA r16/α32, lr 1e-4,
2 epochs, about 32k tokens/step, seed 42. Test evaluation uses the same `(empty)` prompt. The base is a same-node control with the same prompt:
**Formula 69.0, FiNER 61.8**.

| Adapter (training data) | Formula test | vs base | FiNER test | vs base |
|---|---|---|---|---|
| ACE-distilled v1 (targets cite playbook `bullet_ids`) | 26.0 ✗ | +1/−87 | 66.6 | +54/−24 p=0.0009 |
| **ACE-distilled v2** (`bullet_ids` emptied; 803 / 277 ex) | **75.0** | +22/−10 p=0.050 | **68.5** | +56/−24 p=0.0005 |
| v2 LoRA + ACE offline+GT playbook in context | 86.5 (ACE alone 86.5; +3/−3) | — | — | — |
| SFT gold labels, all train items (answer only) | 50.0 | +10/−48 | **73.8** | +100/−26 p<1e-4 |
| SFT gold labels, same items as ACE set | 51.0 | +11/−47 | 64.8 | +51/−42 p=0.41 |
| SFT self-distillation (base's own correct answers), all | 67.5 | +6/−9 p=0.61 | 64.1 | +26/−18 p=0.29 |
| SFT self-distillation, same items as ACE set | 65.5 | +4/−11 p=0.12 | 62.6 | +25/−21 p=0.66 |

Comparisons involving ACE-LoRA v2:
- Formula: v2 is below ICL 82.0 (7 vs 21 items, p=0.013) and below ACE offline 86.5 (2 vs 25, p<1e-5).
- FiNER: v2 vs ICL 70.4 is not significant (50 vs 66, p=0.16).
- On the same train items, v2 beats both self-distillation and label-only SFT:
  - Formula, vs self-distillation (matched items): p=0.002.
  - FiNER, vs self-distillation (matched items): p=0.006.
  - FiNER, vs label-only SFT (matched items): p=0.019.

  So the ACE traces add value per item.
- Plain label SFT on all 1000 FiNER train items (73.8) beats ACE-LoRA (207 items). For tag classification, more labelled data matters most.
- Answer-only label SFT ruins Formula (50.0): the model stops reasoning.
- AppWorld LoRA was trained on 122 trajectories (77 unique train tasks) and gives no significant effect (28.6 vs 27.2).
- Lesson from v1: SFT targets must not reference context that is absent from the SFT input. Otherwise the adapter invents bullet-id lists until it hits max_tokens, and 126/200 Formula answers were unparseable.

<a id="3b"></a>
### 3b. Data selection for the LoRA (ds_r1)

Question: does choosing *which* traces to train on matter more than how many?

Data: a scored candidate pool on train items only. Teacher traces come from Qwen3-32B + offline+GT playbook (K=4) and from
ACE first attempts. Difficulty is probed with the base model with an empty playbook (K=4). Each trace is scored by NLL, by a
397B judge, and by a playbook-citation flag.

Six selections were made. All have citations stripped. Data and build details are in
[`data/selected/`](../data/selected/README.md), with code in `datagen/ds_pool.py` and `datagen/ds_select.py`.

Training: one LoRA per set (r16, 2 epochs, lr 1e-4, thinking off, `rsi/finetune/train_lora.py`).

Evaluation: TEST split, empty context, temperature 0, paired two-sided sign tests. Same-node controls reproduce the earlier
numbers: base 68.5 / 62.1, v2 74.5 / 68.8, gold 73.6.

| set (rows F / N) | Formula | FiNER |
|---|---|---|
| base | 68.5 | 62.1 |
| ACE-LoRA v2 (803 / 207 items) | 74.5 | 68.8 |
| gold-label LoRA (– / 1000) | – | 73.6 |
| rand (431 / 634) | 77.5 (+9.0 vs base, p=.002) | 73.7 |
| clean (431 / 452) | – | 72.2 |
| quality (431 / 634) | – | 72.1 |
| hard (145 / 496) | 79.5 (vs v2 +5.0, p=.031) | 73.9 |
| hard_ctrl (145 / 496) | 67.0 | 72.3 |
| **rand_hardup (706 / 1277)** | **81.5** (vs v2 +7.0, 20/6, p=.009) | **75.2** (vs v2 +6.3, 68/26, p<1e-4) |
| ACE offline+GT playbook, no LoRA | 86.0 (rand_hardup −4.5, p=.022) | 73.8 (rand_hardup +1.4, n.s.) |

Findings:
- **Item difficulty is the deciding signal.**
  - On Formula, 145 traces on items the base model cannot reliably solve give 79.5.
  - 145 random items give 67.0: hard − hard_ctrl = +12.5 (28 vs 3 discordant items, p<1e-4).
  - The 145 hard items also beat 431 random items.
  - On FiNER the same contrast is +1.6 (p=.10), because 78% of the covered FiNER items are already "hard".
- **Keep every item and add extra traces on the hard ones (`rand_hardup`).** This is the best set on both tasks.
- **The 397B judge and "uncited only" filters do not help.** The judge score mostly tracks the citation flag. Stripping citations is safe: rand ≥ clean.
- **LoRA vs skills.**
  - On FiNER the selected LoRA (75.2) now matches the playbook (73.8) with zero context, from self-generated traces on 634 items.
    It is also ≥ the LoRA trained on 1000 gold labels.
  - On Formula the gap to the playbook shrinks from 17.5 to 4.5 points.
- **Caveats.**
  - One seed per set.
  - The playbook numbers come from an earlier, separate eval.
  - The teacher pool used the offline+GT playbook, which was learned on these same train items. No val or test data enters the pool.

## 4. Caveats (read before citing)

- **Backbone and decoding differ from the paper**: Qwen3-32B instead of DeepSeek-V3.1. Static YaRN ×4 can slightly hurt short inputs. Thinking is off.
  The released code defaults are used: 1 offline epoch and 3 reflection rounds. The paper describes up to 5 of each.
- **FiNER offline training did not reach 1000 steps**: +GT stopped at 936 and noGT at 834. The playbook grew to more than 116k tokens and
  overflowed the 131k context. ACE's `playbook_token_budget` is only a prompt hint and is not enforced, and dedup is off by default.
  Best-val selection had already settled (+GT at step 300, noGT at step 700), so the reported test number is very unlikely to change.
- **Resumes after two node losses**: FiNER offline ±GT and online noGT, AppWorld online-challenge (3 times), and DC FiNER were resumed.
  The resumes are exact, because samples are processed in file order and the playbook is the only state, but the in-flight AppWorld task restarted.
  Crashed AppWorld eval shards were re-run for the missing task ids before scoring.
- **Harness fixes** (none change the method): a `local` provider, a server-side `max_tokens` cap of 8192 (runaway repetition), and a context-overflow
  fallback in the AppWorld generator. There is also an Ellipsis-safe API log and two removed imports of modules missing from the release. See `patches/`.
- **DC-CU** uses DC's official prompts *with Python code execution*, which gives DC an arithmetic tool that ACE lacks. The "+GT" variant
  (ground truth appended to the curator input) is our own design, because the paper does not specify one.
- **FiNER ICL** fits only 65 of the 1000 demos in the window. **FiNER benchmark split**: 13/441 test items overlap train passages (12 are identical
  questions). Removing them changes every method by ≤0.2 pts.
- **LoRA**: each configuration was run once (one seed). The Formula v2 gain is borderline (p=0.050). The LoRA base control (69.0 / 61.8) is a separate
  same-node eval and differs slightly from the ACE runs' base (68.9 / 62.8).
- The paper also reports GEPA / MIPROv2 and AppWorld ICL/DC baselines. These were **not run**.
