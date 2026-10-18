# rsi.finetune — LoRA SFT on agent trajectories

`train_lora.py` trains a LoRA adapter on chat JSONL ([DATA_FORMAT.md](DATA_FORMAT.md)) on 1..N GPUs.
```bash
python rsi/finetune/train_lora.py --base Qwen/Qwen3-32B --data sft.jsonl --out ckpt/run1                 # 1 GPU
torchrun --nproc_per_node=4 rsi/finetune/train_lora.py --data sft.jsonl --out ckpt/run1 \
    --epochs 1 --lr 1e-4 --rank 16 --alpha 32 --batch 16 --loss_on last --thinking on                  # WebArena recipe
torchrun --nproc_per_node=4 rsi/finetune/train_lora.py --data appworld.jsonl --out ckpt/aw \
    --loss_on all --thinking off                                                                        # multi-turn, no thinking
```
- Loss only on assistant tokens; the LM head runs only at target positions (no full-vocab logits for 20–50k-token prompts).
- Data parallel: every rank holds the full bf16 model (Qwen3-32B: ~80–98 GB/GPU at 24k tokens with gradient checkpointing,
  fits one H200); gradients are all-reduced per optimizer step; `--batch` = sequences per step (global).
- Output: `<out>/final` and `<out>/stepN` PEFT adapters (vLLM: `--enable-lora --lora-modules name=<out>/final`),
  `train_log.jsonl`, `train_args.json`.
- Tested: 1 GPU and 2-GPU torchrun on WebArena steps (last/on) and AppWorld trajectories (all/off); the rendered prompt of
  the first target turn equals the model's chat template generation prompt token for token.
- Env: [requirements-train.txt](requirements-train.txt).
