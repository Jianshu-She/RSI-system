"""Paired comparison of ACE Finance test evals (eval-rsi/eval/eval_<tag>.json).

Per-item correctness comes from each file's error list (indices of wrong items); the
test is RSI-system's: a two-sided sign test on the items where the two runs disagree.

    python eval-rsi/compare_ace.py formula base ctrl ds    # rows: tags; pairs vs the first tag and ds vs ctrl
"""
import json
import math
import os
import sys

EVAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval")


def load(task, tag):
    d = json.load(open(os.path.join(EVAL, f"eval_{tag}_{task}.json")))
    total = d["results"]["total"]
    wrong = {e["index"] for e in d["errors"]["errors"]}
    return total, [i not in wrong for i in range(total)], d["results"]["accuracy"]


def sign_test(a, b):
    win = sum(x and not y for x, y in zip(a, b))
    loss = sum(y and not x for x, y in zip(a, b))
    n = win + loss
    if n == 0:
        return win, loss, 1.0
    k = min(win, loss)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return win, loss, p


def main():
    task, tags = sys.argv[1], sys.argv[2:]
    runs = {t: load(task, t) for t in tags if os.path.exists(os.path.join(EVAL, f"eval_{t}_{task}.json"))}
    print(f"{task}: " + ", ".join(f"{t} {100 * acc:.1f} (n={tot})" for t, (tot, _, acc) in runs.items()))
    ref = tags[0]
    pairs = [(t, ref) for t in runs if t != ref] + ([("ds", "ctrl")] if {"ds", "ctrl"} <= set(runs) else [])
    for a, b in pairs:
        w, l, p = sign_test(runs[a][1], runs[b][1])
        print(f"  {a} vs {b}: +{w}/-{l}, sign-test p={p:.4f}")


if __name__ == "__main__":
    main()
