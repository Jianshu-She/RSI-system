# SFT data format (shared by all benchmarks)

JSONL, one training example per line:
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "source": "teacher:run_name", "task_id": 123, "step": 4, "n_tokens": 9876}
```
- `messages` is exactly what the model saw at inference, ending with the target assistant turn. Only `messages` is used
  for training; other keys are provenance for filtering/analysis.
- Roles: `system`, `user`, `assistant`.
- `--loss_on last`: loss on the final assistant turn only (WebArena: one row = one agent step; the page/history is in the user turn).
- `--loss_on all`: loss on every assistant turn (AppWorld: one row = a whole multi-turn trajectory).
- `--thinking on`: assistant content contains its `<think>…</think>` block (Qwen3 thinking mode, WebArena).
  `--thinking off`: rendered with `enable_thinking=False`; each assistant turn is prefixed with the empty think block the
  model saw at inference (ACE finance / AppWorld).

Producers: `benchmarks/webarena_reasoningbank/datagen/build_sft_data.py`, `benchmarks/ace/datagen/build_sft_data.py`.
