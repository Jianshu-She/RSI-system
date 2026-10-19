# Copy of RSI-system benchmark/ACE/sft/train_lora.py with one change: --targets (default = the original 7 modules).
# Used as the attention-only control for the DeltaServe co-served LoRA (which trains q,k,v,o only).
"""LoRA SFT of Qwen3-32B on ACE context-distillation data (sft/data_v1/*.jsonl), single GPU.

Format = exactly what the model sees at inference (thinking OFF):
  prompt  = chat_template(messages[:i], add_generation_prompt=True, enable_thinking=False)   (ends with
            "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n")
  target  = assistant content + "<|im_end|>"
Single-turn finance examples match inference exactly. Multi-turn AppWorld trajectories are rendered once with every
assistant turn preceded by the empty think block (the conditioning the model had when it generated that turn) and the
loss on every assistant turn; earlier turns at inference appear without the block (small, documented mismatch).
Loss is computed only on target positions (lm_head applied to those hidden states only -> no 28k x 152k logits).

Usage (conda env verl):
  python train_lora.py --data sft/data_v1/formula.jsonl --out sft/runs/<id>/formula --epochs 2 --lr 1e-4
"""
import argparse
import json
import math
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

MODEL = "/mnt/weka/home/jianshu.she/RSI-system/benchmark/ACE/models/Qwen3-32B-yarn131k"  # same config as inference (YaRN x4)
THINK = "<think>\n\n</think>\n\n"

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--dropout", type=float, default=0.05)
ap.add_argument("--tokens_per_step", type=int, default=32768, help="target tokens (incl. prompt) per optimizer step")
ap.add_argument("--max_len", type=int, default=32768)
ap.add_argument("--heldout_frac", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                help="[DeltaServe control] LoRA target modules (attention-only = q_proj,k_proj,v_proj,o_proj)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
random.seed(args.seed)
torch.manual_seed(args.seed)
tok = AutoTokenizer.from_pretrained(MODEL)
END = tok.convert_tokens_to_ids("<|im_end|>")


def encode(messages):
    """-> (input_ids, labels); labels=-100 except assistant contents (+ their <|im_end|>).
    First prompt = official template up to the first assistant turn (identical to inference); every later turn is
    appended as '<|im_start|>{role}\\n{content}<|im_end|>\\n' followed by the assistant header + empty think block."""
    # AppWorld prompts contain fixed few-shot demo dialogues (assistant turns that are NOT the model's own work): the
    # real task starts at the last user message saying "solve the actual task"; everything up to it is prompt only.
    start = max([i for i, m in enumerate(messages) if m["role"] == "user" and "solve the actual task" in m["content"]] or [0])
    first = next(i for i, m in enumerate(messages) if m["role"] == "assistant" and i > start)
    ids = tok.encode(tok.apply_chat_template(messages[:first], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False), add_special_tokens=False)
    labels = [-100] * len(ids)
    nl = tok.encode("\n", add_special_tokens=False)
    for i in range(first, len(messages)):
        m = messages[i]
        if m["role"] == "assistant":
            if i != first:
                p = tok.encode("<|im_start|>assistant\n" + THINK, add_special_tokens=False)
                ids, labels = ids + p, labels + [-100] * len(p)
            t = tok.encode(m["content"], add_special_tokens=False) + [END]
            ids, labels = ids + t + nl, labels + t + [-100] * len(nl)
        else:
            p = tok.encode(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n", add_special_tokens=False)
            ids, labels = ids + p, labels + [-100] * len(p)
    return ids, labels


rows = [json.loads(l) for l in open(args.data)]
data = []
for r in rows:
    ids, lab = encode(r["messages"])
    if len(ids) > args.max_len:
        ids, lab = ids[: args.max_len], lab[: args.max_len]
    if any(x != -100 for x in lab):
        data.append((ids, lab))
random.shuffle(data)
n_dev = max(1, int(len(data) * args.heldout_frac))
dev, train = data[:n_dev], data[n_dev:]
n_tok = sum(len(x[0]) for x in train)
steps_per_epoch = max(1, math.ceil(n_tok / args.tokens_per_step))
total_steps = max(1, int(steps_per_epoch * args.epochs))
print(json.dumps({"train_ex": len(train), "dev_ex": len(dev), "train_tokens": n_tok, "target_tokens": sum(sum(1 for y in x[1] if y != -100) for x in train),
                  "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, **vars(args)}), flush=True)

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map={"": 0})
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, task_type="CAUSAL_LM",
                                         target_modules=args.targets.split(",")))
model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * total_steps)), total_steps)
base = model.base_model.model


def loss_of(ids, lab):
    x = torch.tensor([ids], device="cuda")
    y = torch.tensor([lab], device="cuda")
    h = base.model(input_ids=x).last_hidden_state[0, :-1]
    tgt = y[0, 1:]
    m = tgt != -100
    logits = base.lm_head(h[m]).float()
    return torch.nn.functional.cross_entropy(logits, tgt[m], reduction="sum"), int(m.sum())


@torch.no_grad()
def eval_dev():
    model.eval()
    tot, n = 0.0, 0
    for ids, lab in dev:
        l, k = loss_of(ids, lab)
        tot, n = tot + l.item(), n + k
    model.train()
    return tot / max(n, 1)


log = open(f"{args.out}/train_log.jsonl", "w")
print(json.dumps({"step": 0, "dev_loss": eval_dev()}), file=log, flush=True)
model.train()
step, t0, order = 0, time.time(), []
while step < total_steps:
    if not order:
        order = list(range(len(train)))
        random.shuffle(order)
    batch, btok = [], 0
    while order and btok < args.tokens_per_step:
        i = order.pop()
        batch.append(train[i])
        btok += len(train[i][0])
    n_tgt = sum(sum(1 for y in lab if y != -100) for _, lab in batch)
    tot = 0.0
    for ids, lab in batch:  # token-weighted mean over the batch
        l, _ = loss_of(ids, lab)
        (l / n_tgt).backward()
        tot += l.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    step += 1
    rec = {"step": step, "train_loss": tot / n_tgt, "lr": sched.get_last_lr()[0], "seqs": len(batch), "tokens": btok,
           "elapsed_s": round(time.time() - t0), "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
    if step % max(1, steps_per_epoch // 2) == 0 or step == total_steps:
        rec["dev_loss"] = eval_dev()
    print(json.dumps(rec), file=log, flush=True)
    print(json.dumps(rec), flush=True)
model.save_pretrained(args.out)
tok.save_pretrained(args.out)
print("SAVED", args.out, flush=True)
