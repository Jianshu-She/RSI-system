# Selected LoRA SFT data (data selection, ds_r1)

> New data. The data used before selection (ACE-distilled v1/v2, gold and self-distillation baselines) is kept in
> [`../original/`](../original/README.md). See [`../README.md`](../README.md) for an overview.

Ready-to-train SFT sets for **Formula** and **FiNER**, built only from **train** items (no val/test). Each row is the
chat JSONL format of [`rsi/finetune/DATA_FORMAT.md`](../../../../rsi/finetune/DATA_FORMAT.md):

```json
{"messages": [{"role": "user", "content": "<ACE generator prompt with an EMPTY playbook>"},
              {"role": "assistant", "content": "{\"reasoning\": ..., \"bullet_ids\": [], \"final_answer\": ...}"}],
 "meta": {"task": "formula", "train_index": 12, "source": "teacher", "leak": false, "base_pass": 0.25, "nll": 0.31, "judge": 9}}
```

Train with thinking off, loss on the last turn:
```bash
python rsi/finetune/train_lora.py --base $MODEL_DIR --data benchmarks/ace/data/selected/formula_rand_hardup.jsonl \
  --out ckpt/formula_rand_hardup --thinking off --loss_on last --epochs 2 --lr 1e-4 --rank 16 --alpha 32 --batch 16 --max_len 8192
```
Or all 10 sets plus evaluation on one node: `sbatch finetune/job_ds_train_eval.sbatch` (cwd `benchmarks/ace`).

## How the data was made

1. **Candidate pool** (`datagen/ds_pool.py`, job `finetune/job_ds_pool.sbatch`), on every train item:
   - `teacher`: Qwen3-32B **with** the learned ACE offline+GT playbook, 4 samples at temperature 0.7;
   - `ace:*`: the first-attempt generator calls of the ACE offline runs;
   - `base`: Qwen3-32B with an **empty** playbook, 4 samples. This is only a difficulty probe (`base_pass` = fraction correct) and is never used as a target.

   Only correct candidates are kept. Each one is scored with:
   - `nll`: the mean token NLL of the target under the base model;
   - `judge`: a 1–10 score from Qwen3.5-397B;
   - `leak`: whether the reasoning cites playbook bullet ids or mentions "playbook".
   The pools are shipped as `pool_<task>.jsonl.gz`.
2. **Selection** (`datagen/ds_select.py`, seed 0):
   - The SFT input is the empty-playbook prompt, so the target must not depend on the playbook.
   - `bullet_ids` are emptied and playbook citations are stripped from the reasoning. Traces that still leak afterwards are dropped.
   - Running the script on the shipped pools reproduces these files byte-for-byte.

| set | rule | Formula rows | FiNER rows |
|---|---|---|---|
| `rand` | every covered item, one random correct trace | 431 | 634 |
| `clean` | only items with a trace that never cited the playbook | 431 | 452 |
| `hard` | only items the base model does not always solve (`base_pass < 1`), lowest-NLL trace | 145 | 496 |
| `hard_ctrl` | size-matched random control for `hard` | 145 | 496 |
| `quality` | every item, best trace (uncited > judge ≥ 8 > lowest NLL) | 431 | 634 |
| **`rand_hardup`** | `rand` + up to 2 extra distinct traces per hard item | **706** | **1277** |

Items with no correct, clean trace are not covered: 69/500 Formula items and 366/1000 FiNER items. `<task>_summary.json`
gives the mean `base_pass`, mean NLL and source mix of each set.

## Results (test split, empty context, temperature 0, one seed)

| set | Formula (200) | FiNER (441) |
|---|---|---|
| base Qwen3-32B | 68.5 | 62.1 |
| previous ACE-LoRA v2 (data_v2) | 74.5 | 68.8 |
| gold-label LoRA (1000 FiNER items) | – | 73.6 |
| rand | 77.5 | 73.7 |
| clean | – | 72.2 |
| quality | – | 72.1 |
| hard | 79.5 | 73.9 |
| hard_ctrl | 67.0 | 72.3 |
| **rand_hardup** | **81.5** | **75.2** |
| *ACE playbook in context (no LoRA)* | *86.0* | *73.8* |

**Recommendation:** use `rand_hardup`. See [`results/RESULTS.md`](../../results/RESULTS.md#3b) §3b for the paired tests.
