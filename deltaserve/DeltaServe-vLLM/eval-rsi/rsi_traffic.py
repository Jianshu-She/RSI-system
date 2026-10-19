"""ACE Finance "user traffic" against an OpenAI-compatible server, with traces.

Builds each request exactly like ACE's Generator (GENERATOR_PROMPT, temperature 0, thinking off),
streams it to measure TTFT / end-to-end latency, and scores it with ACE's own extract_answer +
DataProcessor.answer_is_correct. One JSONL record per request: the full prompt + response (the
trace), correctness, and timing. Run with the ACE venv python from the ACE upstream dir:

  cd $ACE/ace && .venv/bin/python /path/eval-rsi/rsi_traffic.py --url http://127.0.0.1:8000/v1 \
      --model llama3 --task formula --split train --context $ACE/results/.../best_playbook.txt \
      --concurrency 16 --out traces.jsonl
  # open-loop Poisson arrivals instead of a closed loop:  --rps 0.5 [--duration 600] [--loops 3]
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time

sys.path.insert(0, os.getcwd())
from openai import AsyncOpenAI  # noqa: E402

from ace.prompts.generator import GENERATOR_PROMPT  # noqa: E402
from eval.finance.data_processor import DataProcessor  # noqa: E402
from utils import extract_answer  # noqa: E402


def load_samples(task, split):
    cfg = json.load(open("./eval/finance/data/sample_config.json"))[task]
    key = {"train": "train_data", "val": "val_data", "test": "test_data"}[split]
    proc = DataProcessor(task)
    return proc, proc.process_task_data([json.loads(l) for l in open(cfg[key])])


async def one(client, args, proc, playbook, i, s, out, lock, stats):
    prompt = GENERATOR_PROMPT.format(playbook, "(empty)", s["question"], s["context"])
    t0 = time.time()
    ttft = None
    parts = []
    try:
        stream = await client.chat.completions.create(
            model=args.model, messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=args.max_tokens, stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        async for ch in stream:
            if ch.choices and ch.choices[0].delta.content:
                if ttft is None:
                    ttft = time.time() - t0
                parts.append(ch.choices[0].delta.content)
        err = None
    except Exception as e:  # keep the run alive; the record says what failed
        err = repr(e)
    e2e = time.time() - t0
    resp = "".join(parts)
    ans = extract_answer(resp) if resp else None
    correct = bool(resp) and proc.answer_is_correct(ans, s["target"])
    rec = {"index": i, "task": args.task, "split": args.split, "t_send": t0, "ttft": ttft,
           "e2e": e2e, "correct": correct, "error": err, "prompt": prompt, "response": resp,
           "question": s["question"], "context": s["context"], "target": s["target"]}
    async with lock:
        out.write(json.dumps(rec) + "\n")
        out.flush()
        stats["n"] += 1
        stats["ok"] += correct


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--context", default="", help="playbook file (empty = '(empty)')")
    ap.add_argument("--concurrency", type=int, default=0, help="closed loop with N workers")
    ap.add_argument("--rps", type=float, default=0.0, help="open-loop Poisson arrival rate")
    ap.add_argument("--loops", type=int, default=1, help="passes over the split (open loop)")
    ap.add_argument("--duration", type=float, default=0.0, help="stop sending after N s (open loop)")
    ap.add_argument("--stop-file", default="", help="stop sending once this file exists (open loop)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    proc, samples = load_samples(args.task, args.split)
    if args.limit:
        samples = samples[: args.limit]
    playbook = open(args.context).read() if args.context else "(empty)"
    client = AsyncOpenAI(base_url=args.url, api_key="x", timeout=3600)
    lock, stats = asyncio.Lock(), {"n": 0, "ok": 0}
    t_start = time.time()
    with open(args.out, "a") as out:
        if args.concurrency:
            sem = asyncio.Semaphore(args.concurrency)

            async def bounded(i, s):
                async with sem:
                    await one(client, args, proc, playbook, i, s, out, lock, stats)
            await asyncio.gather(*(bounded(i, s) for i, s in enumerate(samples)))
        else:
            rng = random.Random(args.seed)
            tasks = []
            for _ in range(max(1, args.loops)):
                for i, s in enumerate(samples):
                    if args.duration and time.time() - t_start > args.duration:
                        break
                    if args.stop_file and os.path.exists(args.stop_file):
                        break
                    tasks.append(asyncio.create_task(one(client, args, proc, playbook, i, s, out, lock, stats)))
                    await asyncio.sleep(rng.expovariate(args.rps))
            await asyncio.gather(*tasks)
    print(f"RESULT {args.task} {args.split}: {stats['ok']}/{stats['n']} correct "
          f"({100 * stats['ok'] / max(1, stats['n']):.1f}%), {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
