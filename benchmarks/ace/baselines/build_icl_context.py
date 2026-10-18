"""Build the ICL baseline context for the ACE finance tasks.

Paper (ACE, Sec. 4.2): ICL supplies "all training samples when they fit within the model's context window;
otherwise we fill the window with as many demonstrations as possible."

The demonstrations are written to a text file that is passed to `eval.finance.run --mode eval_only
--initial_playbook_path`, so ICL uses exactly ACE's generator prompt and evaluation code; only the
context slot differs (gold-labelled training samples instead of a learned playbook).

Usage (vLLM env python, needs transformers; paths from env.sh):
  $VLLM_PY baselines/build_icl_context.py <task> $ACE_ICL_DIR/<task>_icl_context.txt [--max_model_len 131072] [--max_tokens 4096] [--seed 42]
"""
import argparse
import json
import os
import random
import sys

from transformers import AutoTokenizer

_ROOT = os.environ.get("ACE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WORK = os.environ.get("ACE_WORK_DIR", f"{_ROOT}/work")
ACE_DIR = os.environ.get("ACE_UPSTREAM_DIR", f"{_WORK}/ace")
TOKENIZER = os.environ.get("MODEL_DIR", f"{_WORK}/models/Qwen3-32B-yarn131k")
DATA = {
    "finer": ("eval/finance/data/finer_train_batched_1000_samples.jsonl", "eval/finance/data/finer_test_subset_006_seed42.jsonl"),
    "formula": ("eval/finance/data/formula_train_subset_500.jsonl", "eval/finance/data/formula_test.jsonl"),
}
HEADER = ("Below are solved examples from the training set (question, context and the correct answer). "
          "Use them as demonstrations.\n\n")


def demo_text(i, s):
    ctx = f"Context: {s['context']}\n" if s["context"].strip() else ""
    return f"### Example {i}\nQuestion: {s['question']}\n{ctx}Answer: {s['target']}\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(DATA))
    ap.add_argument("out")
    ap.add_argument("--max_model_len", type=int, default=131072)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, ACE_DIR)
    from eval.finance.data_processor import DataProcessor
    from ace.prompts.generator import GENERATOR_PROMPT

    proc = DataProcessor(args.task)
    load = lambda p: proc.process_task_data([json.loads(l) for l in open(f"{ACE_DIR}/{p}")])
    train, test = load(DATA[args.task][0]), load(DATA[args.task][1])
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    n = lambda t: len(tok.encode(t))

    # Budget: the longest test prompt (with an empty context slot) + output tokens + chat-template margin must still fit.
    longest_test = max(n(GENERATOR_PROMPT.format("", "(empty)", s["question"], s["context"])) for s in test)
    budget = args.max_model_len - longest_test - args.max_tokens - 512
    random.Random(args.seed).shuffle(train)

    text, used = HEADER, n(HEADER)
    k = 0
    for s in train:
        d = demo_text(k + 1, s)
        c = n(d)
        if used + c > budget:
            break
        text, used, k = text + d, used + c, k + 1
    open(args.out, "w").write(text)
    print(json.dumps({"task": args.task, "train": len(train), "demos": k, "demo_tokens": used,
                      "budget": budget, "longest_test_prompt": longest_test, "out": args.out}))


if __name__ == "__main__":
    main()
