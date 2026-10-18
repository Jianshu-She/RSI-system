"""Build a LoRA SFT set (chat JSONL) from saved WebArena agent runs.

Each output line = one agent step, in the format the agent saw at inference:
  {"messages": [system, user, assistant], "source", "task_id", "step", "n_tokens"}
The assistant turn is the training target (<think>...</think> + <action>...</action>); see rsi/finetune.

Rejection-sampling SFT: only steps from trajectories whose task succeeded (official reward) are kept, and only for tasks
in --split (default: train). Test tasks are never used (asserted).

Sources
  --teacher RUN [RUN ...]   runs of a (stronger) teacher agent, kept as is (e.g. Qwen3.5-397B no_memory runs)
  --self    RUN [RUN ...]   runs of the student model itself, kept only if the reply style is clean:
                            hedge markers ("We need", "maybe", ...) <= --max_hedge per 100 words and reply <= --max_reply tokens
Filters for all sources: no memory-augmented prompts, well-formed target, no parse retries, <= --max_len tokens, dedup.

Example (the lora_v2 recipe):
  python datagen/build_sft_data.py --out $WA_WORK/sft/v2.jsonl \
      --teacher $WA_RUNS/ref_397b_all $WA_RUNS/train_397_s1 $WA_RUNS/train_397_s2 $WA_RUNS/train_397_s3 \
      --self $WA_RUNS/base_q32_all
Lessons (see results/RESULTS.md): do not mix in traces with a telegraphic/hedging style (the v1 adapter copied it and looped).
"""
import argparse, collections, json, os, random, re, sys, tempfile

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HOME, "tools"))
from export_traces import export_run  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True)
ap.add_argument("--teacher", nargs="*", default=[])
ap.add_argument("--self", dest="self_runs", nargs="*", default=[])
ap.add_argument("--split", default="train", help="splits/<split>_ids.txt")
ap.add_argument("--tokenizer", default="Qwen/Qwen3-32B")
ap.add_argument("--max_len", type=int, default=24576)
ap.add_argument("--max_hedge", type=float, default=1.0)
ap.add_argument("--max_reply", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
if not a.teacher and not a.self_runs:
    ap.error("give at least one --teacher or --self run")

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(a.tokenizer)
keep_ids = {int(x) for x in open(os.path.join(HOME, "splits", f"{a.split}_ids.txt")).read().split()}
test_ids = {int(x) for x in open(os.path.join(HOME, "splits", "test_ids.txt")).read().split()}
assert a.split == "test" or not (keep_ids & test_ids), "training split overlaps the test split"
HEDGE = re.compile(r"\b(We need|We can maybe|Need maybe|maybe)\b")

stats, rows, seen = collections.Counter(), [], set()
with tempfile.TemporaryDirectory() as td:
    for kind, dirs in (("teacher", a.teacher), ("self", a.self_runs)):
        for rd in dirs:
            tmp = os.path.join(td, "x.jsonl")
            with open(tmp, "w") as f:
                export_run(rd, f)
            for line in open(tmp):
                r = json.loads(line); stats["seen"] += 1
                if r["task_id"] not in keep_ids: stats["drop_not_in_split"] += 1; continue
                if not (r["task_gt_reward"] or 0) > 0: stats["drop_failed_traj"] += 1; continue
                if r["has_memory_in_prompt"] or not r["response_format_ok"] or r["n_parse_retries"]:
                    stats["drop_format"] += 1; continue
                reply = r["response"]
                if kind == "self":
                    if len(HEDGE.findall(reply)) / max(len(reply.split()), 1) * 100 > a.max_hedge:
                        stats["drop_style_hedge"] += 1; continue
                    if len(tok(reply)["input_ids"]) > a.max_reply: stats["drop_style_long"] += 1; continue
                msgs = r["messages"] + [{"role": "assistant", "content": reply}]
                key = (msgs[1]["content"], reply)
                if key in seen: stats["drop_dup"] += 1; continue
                n = len(tok.apply_chat_template(msgs, tokenize=True))
                if n > a.max_len: stats["drop_too_long"] += 1; continue
                seen.add(key)
                rows.append({"messages": msgs, "source": f"{kind}:{os.path.basename(rd.rstrip('/'))}",
                             "task_id": r["task_id"], "step": r["step"], "n_tokens": n})
random.Random(a.seed).shuffle(rows)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, "w") as f:
    for x in rows:
        f.write(json.dumps(x) + "\n")
by = collections.Counter(x["source"] for x in rows); tasks = collections.defaultdict(set)
for x in rows: tasks[x["source"]].add(x["task_id"])
summary = {"out": a.out, "rows": len(rows), "tokens": sum(x["n_tokens"] for x in rows), "filters": dict(stats),
           "by_source": {k: {"steps": v, "tasks": len(tasks[k])} for k, v in by.items()},
           "distinct_tasks": len({x["task_id"] for x in rows}), "split_tasks": len(keep_ids)}
json.dump(summary, open(a.out + ".summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
