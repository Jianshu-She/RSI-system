# ACE LoRA SFT data

All data comes from **train** items only. No val or test item is used, so nothing leaks into the evaluation splits.
Every file uses the chat JSONL format of [`rsi/finetune/DATA_FORMAT.md`](../../../rsi/finetune/DATA_FORMAT.md):
`{"messages": [...], "meta": {...}}`.

| folder | what | built by | README |
|---|---|---|---|
| [`original/`](original/) | the data used before selection: ACE-distilled v1/v2 plus the gold-label and self-distillation baselines | `datagen/build_sft_data.py`, `make_data_v2.py`, `build_sft_baselines.py` | [original/README.md](original/README.md) |
| [`selected/`](selected/) | **new**: a scored candidate pool plus 6 selected sets (difficulty-weighted). **Recommended: `selected/<task>_rand_hardup.jsonl`** | `datagen/ds_pool.py`, `ds_select.py` | [selected/README.md](selected/README.md) |

Test accuracy of the LoRA trained on each set (empty context, temperature 0):

| data | Formula | FiNER |
|---|---|---|
| base Qwen3-32B, no LoRA | 68.5–69.0 | 61.8–62.1 |
| `original/data_v2` (ACE-LoRA v2) | 75.0 | 68.5 |
| `original/baselines/finer_gold_all` | – | 73.8 |
| **`selected/*_rand_hardup`** | **81.5** | **75.2** |
| ACE playbook in context, no LoRA | 86.0–86.5 | 73.8 |
