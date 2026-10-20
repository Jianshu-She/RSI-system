"""General-capability check outside the trained task: GSM8K (test, 1319) and MMLU (1000 items of `all/test`, seed 0).

Answers whether the distilled experience (the LoRA) or the skills in context (the playbook) cost general ability.
"Skills" = the playbook text put in a system message, the way an agent harness injects skills. Temperature 0,
thinking off. Writes the same JSON shape as ACE's eval_split.py, so compare.py pairs the items:
  {"results": {"accuracy", "correct", "total"}, "errors": {"errors": [{"index": i}, ...]}, "items": [...]}

  python eval_general.py --url http://127.0.0.1:9300/v1 --model qwen3-32b --bench gsm8k [--skills PLAYBOOK] --out x.json
"""
import argparse
import asyncio
import glob
import json
import os
import random
import re

import pyarrow.parquet as pq
from openai import AsyncOpenAI

HUB = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")


def parquet(pattern):
    f = sorted(glob.glob(os.path.join(HUB, pattern)))
    if not f:
        raise SystemExit(f"not in the HF cache: {pattern}")
    return pq.read_table(f[0]).to_pylist()


def load(bench):
    if bench == "gsm8k":
        rows = parquet("datasets--openai--gsm8k/snapshots/*/main/test-00000-of-00001.parquet")
        return [{"q": r["question"] + "\n\nSolve step by step, then give the final number on the last line as "
                 "'#### <number>'.", "a": r["answer"].split("####")[-1].strip().replace(",", "")} for r in rows]
    if bench == "mmlu":
        rows = parquet("datasets--cais--mmlu/snapshots/*/all/test-00000-of-00001.parquet")
        rows = random.Random(0).sample(rows, 1000)
        return [{"q": f"Subject: {r['subject'].replace('_', ' ')}\n\n{r['question']}\n"
                 + "".join(f"{'ABCD'[k]}. {c}\n" for k, c in enumerate(r["choices"]))
                 + "\nAnswer with the letter of the correct option only.", "a": "ABCD"[r["answer"]]} for r in rows]
    raise SystemExit(bench)


def num(s):
    s = s.replace(",", "").replace("$", "")
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", s) or None
    v = m.group(1) if m else (re.findall(r"-?\d+(?:\.\d+)?", s) or [None])[-1]
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def score(bench, resp, gold):
    if bench == "gsm8k":
        p = num(resp)
        return p is not None and abs(p - float(gold)) < 1e-6, p
    m = re.findall(r"\b([ABCD])\b", resp)
    p = m[0] if m else None
    return p == gold, p


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", required=True, choices=["gsm8k", "mmlu"])
    ap.add_argument("--skills", default="", help="playbook file put in the system message ('' = none)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="first N items only (smoke tests)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = load(a.bench)
    if a.limit:
        items = items[: a.limit]
    sys_msg = []
    if a.skills:
        sys_msg = [{"role": "system", "content": "You have the following playbook of skills learned from previous "
                    "tasks. Use it when it is relevant.\n\n" + open(a.skills).read()}]
    client = AsyncOpenAI(base_url=a.url, api_key="x", timeout=3600)
    sem = asyncio.Semaphore(a.workers)
    out = [None] * len(items)

    async def one(i, it):
        async with sem:
            r = await client.chat.completions.create(
                model=a.model, messages=sys_msg + [{"role": "user", "content": it["q"]}], temperature=0.0,
                max_tokens=1024 if a.bench == "gsm8k" else 32,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            resp = r.choices[0].message.content or ""
            ok, pred = score(a.bench, resp, it["a"])
            out[i] = {"index": i, "correct": ok, "pred": pred, "gold": it["a"], "response": resp,
                      "prompt_tokens": r.usage.prompt_tokens}

    await asyncio.gather(*(one(i, it) for i, it in enumerate(items)))
    c = sum(x["correct"] for x in out)
    json.dump({"task": a.bench, "context": a.skills or None, "model": a.model,
               "results": {"accuracy": c / len(out), "correct": c, "total": len(out)},
               "errors": {"errors": [{"index": x["index"]} for x in out if not x["correct"]]}, "items": out},
              open(a.out, "w"), indent=1)
    print(f"RESULT {a.bench} skills={bool(a.skills)} model={a.model}: accuracy={c / len(out):.4f} ({c}/{len(out)})")


if __name__ == "__main__":
    asyncio.run(main())
