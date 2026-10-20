# Original LoRA SFT data (before data selection)

This is the data the first LoRA experiments were trained on, kept as built. Results are in [`results/RESULTS.md`](../../results/RESULTS.md) §3.
The newer selected sets are in [`../selected/`](../selected/README.md).

Every row is `{"messages": [user, assistant], "meta": {...}}`, train items only.
- **Input:** the ACE generator prompt with an **empty** playbook (`(empty)`), which is exactly what the base model sees at test time.
- **Training:** thinking off, loss on the last turn.

## `data_v1/`: ACE-distilled, raw

Built by `datagen/build_sft_data.py`.

**Finance targets.** Each target is the first answer ACE gave on a train item *with* its learned playbook, from the offline ±GT
runs, and only correct answers are kept. Calls made after reflection or curation are excluded, because in the +GT runs those calls saw the ground truth.

**AppWorld targets.** Each target is the full trajectory of a train task that passed AppWorld's evaluator. The playbook block is
replaced by the initial playbook, and the 18 trajectories that contain +GT retry turns are dropped.

| file | rows | unique train items | source runs |
|---|---|---|---|
| `formula.jsonl` | 803 | 417 / 500 | formula_offline_gt (408) + formula_offline_nogt (395) |
| `finer.jsonl` | 277 | 207 / 1000 | finer_offline_gt (+resume) 182 + finer_offline_nogt (+resume) 95 |
| `appworld.jsonl` | 122 | 77 train tasks | qwen_off_gt_adapt (64) + qwen_off_nogt_adapt (58) |

`summary.json` gives the per-run counts and token lengths.

⚠ **Do not train the Finance files as-is.** Their targets still contain playbook `bullet_ids` (for example `["calc-00057"]`) that the
empty-playbook input cannot justify. The resulting Formula LoRA invents id lists until it hits max_tokens: test score 26.0, with
126/200 answers unparseable. Use `data_v2` instead. The AppWorld file has no such ids and is what the AppWorld LoRA was trained on.

## `data_v2/`: ACE-distilled, `bullet_ids` emptied (the "ACE-LoRA v2" data)

Built by `datagen/make_data_v2.py`. The rows match data_v1 one to one; only `"bullet_ids": [...]` is replaced by `[]`.
Citations inside the reasoning text are **not** stripped (the selected sets do strip them).

| file | rows | LoRA test (empty context) |
|---|---|---|
| `formula.jsonl` | 803 | 75.0 (base 69.0) |
| `finer.jsonl` | 277 | 68.5 (base 61.8) |

## `baselines/`: SFT controls on the same train data

Built by `datagen/build_sft_baselines.py` (job `finetune/job_sft_baselines.sbatch`).

| file | rows | target | LoRA test |
|---|---|---|---|
| `formula_gold_all.jsonl` | 500 | gold answer only (empty reasoning), every train item | 50.0 (stops reasoning) |
| `formula_gold_matched.jsonl` | 417 | gold answer only, the same items as data_v2 | 51.0 |
| `formula_self_all.jsonl` | 344 | the base model's own correct answers (344/500 correct) | 67.5 |
| `formula_self_matched.jsonl` | 335 | self-distillation on the data_v2 items | 65.5 |
| `finer_gold_all.jsonl` | 1000 | gold labels only, every train item | **73.8** |
| `finer_gold_matched.jsonl` | 207 | gold labels only, the data_v2 items | 64.8 |
| `finer_self_all.jsonl` | 268 | the base model's own correct answers (268/1000 correct) | 64.1 |
| `finer_self_matched.jsonl` | 100 | self-distillation on the data_v2 items | 62.6 |

The "matched" sets use the same items as data_v2, so their difference from data_v2 measures what the ACE traces add per item.
On those items data_v2 is better: Formula vs self p=0.002, FiNER vs self p=0.006, FiNER vs gold p=0.019.
