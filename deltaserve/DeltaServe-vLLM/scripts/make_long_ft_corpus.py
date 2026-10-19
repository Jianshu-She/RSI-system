"""Build a mixed short/long FT corpus for the long-context bring-up.

One sample per line (the FinetuningStore format), newlines escaped. Long
samples are OpenR1-Math problem + reasoning trace pairs (real 1K-16K-token
trajectories); short ones come from an alpaca corpus. Lengths are measured
with the served model's tokenizer so the buckets are exact.

    python scripts/make_long_ft_corpus.py --model /path/Qwen3-32B \
        --short alpaca_1000.txt --out long_ft_corpus.txt
"""
import argparse
import glob
import os
import random

import pyarrow.parquet as pq
from transformers import AutoTokenizer

BUCKETS = [(1024, 2048), (2048, 4096), (4096, 8192), (8192, 12288), (12288, 16000)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--short", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-short", type=int, default=200)
    ap.add_argument("--per-bucket", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)

    files = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--open-r1--OpenR1-Math-220k/"
        "snapshots/*/all/*.parquet")))
    want = {b: [] for b in BUCKETS}
    for f in files:
        t = pq.read_table(f, columns=["problem", "generations"]).to_pylist()
        rng.shuffle(t)
        for row in t:
            for gen in row["generations"] or []:
                text = f"Problem: {row['problem']}\nSolution: {gen}"
                text = " ".join(text.split())          # one line, no newlines
                n = len(tok(text)["input_ids"])
                for b in BUCKETS:
                    if b[0] <= n < b[1] and len(want[b]) < args.per_bucket:
                        want[b].append((n, text))
            if all(len(v) >= args.per_bucket for v in want.values()):
                break
        if all(len(v) >= args.per_bucket for v in want.values()):
            break

    with open(args.short) as fh:
        short = [ln.strip() for ln in fh if ln.strip()][: args.n_short]
    lines = short + [t for v in want.values() for _, t in v]
    rng.shuffle(lines)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} samples to {args.out}: {len(short)} short")
    for b, v in want.items():
        print(f"  {b}: {len(v)} samples, lengths {sorted(n for n, _ in v)}")


if __name__ == "__main__":
    main()
