"""Generic LoRA SFT for chat models on {"messages": [...]} JSONL (see DATA_FORMAT.md). 1..N GPUs via torchrun.

Loss is computed only on assistant tokens; the LM head is applied only at target positions, so long prompts (web pages,
tool outputs, 20-50k tokens) never materialize full-vocabulary logits.

  --loss_on last   train on the final assistant turn only (one row = one agent step; WebArena)
  --loss_on all    train on every assistant turn of a multi-turn conversation (AppWorld trajectories); Qwen chat
                   format built turn by turn, each turn conditioned exactly as when it was generated
  --thinking on    render with the model's thinking mode (Qwen3 default); targets keep their <think>...</think>
  --thinking off   render with enable_thinking=False (the model was run without thinking at inference)

Single GPU:   python rsi/finetune/train_lora.py --data train.jsonl --out ckpt/run1
Multi GPU:    torchrun --nproc_per_node=4 rsi/finetune/train_lora.py --data train.jsonl --out ckpt/run1
Output: <out>/final (PEFT adapter, loadable by vLLM --enable-lora), <out>/stepN (mid-run checkpoints), train_log.jsonl.
"""
import argparse, json, math, os, random, time

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--base", default="Qwen/Qwen3-32B", help="HF id or local dir of the base model")
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=1)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--dropout", type=float, default=0.05)
ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
ap.add_argument("--batch", type=int, default=16, help="sequences per optimizer step (global, across GPUs)")
ap.add_argument("--warmup", type=float, default=0.05)
ap.add_argument("--max_len", type=int, default=32768, help="drop longer sequences")
ap.add_argument("--loss_on", choices=["last", "all"], default="last")
ap.add_argument("--thinking", choices=["on", "off"], default="on")
ap.add_argument("--saves", type=int, default=2, help="number of evenly spaced checkpoints (besides final)")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

distributed = "RANK" in os.environ
if distributed:
    dist.init_process_group("nccl")
RANK, WORLD = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
LOCAL = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(LOCAL)
MAIN = RANK == 0
random.seed(a.seed); torch.manual_seed(a.seed)
if MAIN:
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(vars(a), world_size=WORLD), open(f"{a.out}/train_args.json", "w"), indent=1)

tok = AutoTokenizer.from_pretrained(a.base)
TMPL = {} if a.thinking == "on" else {"enable_thinking": False}


def render(msgs, gen=False):
    return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen, **TMPL)


THINK_OFF = "<think>\n\n</think>\n\n"


def encode(msgs):
    """Token ids of the whole chat + list of (start, end) target spans (assistant turns)."""
    if a.loss_on == "last":
        # single target turn: the official chat template, prompt = render(history, add_generation_prompt) exactly as at inference
        pre, full = render(msgs[:-1], gen=True), render(msgs)
        return (full, [(len(pre), len(full))]) if full[:len(pre)] == pre and len(full) > len(pre) else (full, [])
    # every assistant turn: Qwen chat format built turn by turn, each assistant turn prefixed exactly as the model was
    # conditioned when it generated that turn (thinking off: empty think block; thinking on: content keeps its <think>).
    # Templates re-render history differently (e.g. drop old think blocks), so a template-based multi-turn render would
    # not match what the model saw; this matches the generation prompt of every turn.
    ids, spans = [], []
    for m in msgs:
        if m["role"] not in ("system", "user", "assistant"):
            raise ValueError(f"--loss_on all supports system/user/assistant roles, got {m['role']}")
        if m["role"] == "assistant":
            ids += tok.encode("<|im_start|>assistant\n" + (THINK_OFF if a.thinking == "off" else ""), add_special_tokens=False)
            body = tok.encode(m["content"] + "<|im_end|>", add_special_tokens=False)
            spans.append((len(ids), len(ids) + len(body)))
            ids += body + tok.encode("\n", add_special_tokens=False)
        else:
            ids += tok.encode(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n", add_special_tokens=False)
    return ids, spans


data, skipped = [], 0
for line in open(a.data):
    ids, spans = encode(json.loads(line)["messages"])
    if not spans or len(ids) > a.max_len:
        skipped += 1; continue
    data.append((ids, spans))
if MAIN:
    print(f"{len(data)} sequences ({skipped} skipped: too long or no usable target), "
          f"{sum(len(i) for i, _ in data)} tokens, {sum(e - s for _, sp in data for s, e in sp)} target tokens, "
          f"world {WORLD}", flush=True)

model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map={"": LOCAL}, attn_implementation="sdpa")
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none",
                                         task_type="CAUSAL_LM", target_modules=a.target_modules.split(",")))
if MAIN:
    model.print_trainable_parameters()
backbone, lm_head = model.base_model.model.model, model.base_model.model.lm_head
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

# same global order on every rank; each optimizer step takes `batch` sequences, split round-robin over ranks
total = int(len(data) * a.epochs); order = []
while len(order) < total:
    ep = list(range(len(data))); random.shuffle(ep); order += ep
order = order[:total]
per_step = max(WORLD, a.batch - a.batch % WORLD)
steps = [order[i:i + per_step] for i in range(0, len(order), per_step)]
warm = max(1, int(len(steps) * a.warmup))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, len(steps) - warm))))
save_at = {round(len(steps) * k / (a.saves + 1)) for k in range(1, a.saves + 1)} - {0, len(steps)}

model.train(); t0 = time.time(); log = open(f"{a.out}/train_log.jsonl", "w") if MAIN else None
for step, batch in enumerate(steps, 1):
    loss_acc = torch.zeros((), device="cuda"); tok_acc = torch.zeros((), device="cuda")
    for idx in batch[RANK::WORLD]:
        ids, spans = data[idx]
        x = torch.tensor([ids], device="cuda")
        h = backbone(input_ids=x).last_hidden_state[0]
        pos = torch.cat([torch.arange(s - 1, e - 1, device="cuda") for s, e in spans])
        tgt = torch.cat([x[0, s:e] for s, e in spans])
        loss_sum = torch.nn.functional.cross_entropy(lm_head(h[pos]).float(), tgt, reduction="sum")
        (loss_sum / (tgt.numel() * len(batch))).backward()     # token-mean per sequence, mean over the global batch
        loss_acc += loss_sum.detach(); tok_acc += tgt.numel()
    if distributed:
        for p in params:
            if p.grad is None:                                   # a rank can get no sequence in the last batch
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad)
        dist.all_reduce(loss_acc); dist.all_reduce(tok_acc)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if MAIN:
        rec = {"step": step, "of": len(steps), "loss": (loss_acc / tok_acc).item(), "lr": sched.get_last_lr()[0],
               "elapsed_min": round((time.time() - t0) / 60, 2), "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if step in save_at:
            model.save_pretrained(f"{a.out}/step{step}")
if MAIN:
    model.save_pretrained(f"{a.out}/final")
    print("saved", f"{a.out}/final", flush=True)
if distributed:
    dist.barrier(); dist.destroy_process_group()
