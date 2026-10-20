"""Data selection, step 3: build SFT sets from the scored pool ($DS_DIR/pool_<task>.jsonl[.gz]). One trace per item.

SFT row = {"messages": [user: ACE generator prompt with an EMPTY playbook, assistant: cleaned target]} (thinking off).
Cleaning: bullet_ids emptied; playbook citations removed from the reasoning ("[calc-00057]", "according to the playbook").

Strategies (seed 0):
  rand       every covered item, a random correct trace (citations stripped)
  clean      every item that has an uncited trace, a random uncited trace (no rewriting needed)
  hard       items the base model does not always solve (base_pass < 1), the most on-distribution trace (lowest nll,
             uncited preferred)
  hard_ctrl  size-matched random control for `hard`: random items (same count), random trace
  quality    every covered item, best trace: uncited > judge >= 8 > lowest nll
  rand_hardup  rand + up to 2 extra distinct traces per hard item
Usage (source env.sh): $ACE_PY datagen/ds_select.py <task>   -> $DS_DIR/<task>_<strategy>.jsonl (+ <task>_summary.json)
The shipped pools are gzipped; the script reads either form.
"""
import gzip, json, os, random, re, sys

_ROOT = os.environ.get("ACE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WORK = os.environ.get("ACE_WORK_DIR", f"{_ROOT}/work")
ACE = os.environ.get("ACE_UPSTREAM_DIR", f"{_WORK}/ace")
RESULTS = os.environ.get("ACE_RESULTS_DIR", f"{_WORK}/results")
DS_DIR = os.environ.get("DS_DIR", f"{_ROOT}/data/selected")   # pool_<task>.jsonl(.gz) + selected sets
os.chdir(ACE); sys.path.insert(0, ACE)
from ace.prompts.generator import GENERATOR_PROMPT  # noqa: E402
from eval.finance.data_processor import DataProcessor  # noqa: E402

TASK = sys.argv[1]
OUT = DS_DIR; os.makedirs(OUT, exist_ok=True)
TRAIN = {"formula": "eval/finance/data/formula_train_subset_500.jsonl",
         "finer": "eval/finance/data/finer_train_batched_1000_samples.jsonl"}[TASK]
proc = DataProcessor(TASK)
train = proc.process_task_data([json.loads(l) for l in open(TRAIN)])
BULLET_FIELD = re.compile(r'"bullet_ids"\s*:\s*\[[^\]]*\]')
CITE = [re.compile(p, re.I) for p in (
    r"\s*\((?:see |per |cf\. )?\[?[a-z]{2,8}-\d{5}\]?(?:\s*(?:,|and)\s*\[?[a-z]{2,8}-\d{5}\]?)*\)",   # "(calc-00057, ctx-00010)"
    r"(?:according to|per|as per|based on|following|using|from|as (?:stated|noted|described) in)\s+"
    r"\[?[a-z]{2,8}-\d{5}\]?(?:\s*(?:,|and)\s*\[?[a-z]{2,8}-\d{5}\]?)*\s*,?\s*",                         # "According to [calc-00057] and [ctx-00010], "
    r",?\s*(?:as|according to|per|in line with|following)\s+(?:the\s+)?(?:guidance\s+in\s+the\s+)?playbook(?:'s\s+\w+)?",
    r"\[?[a-z]{2,8}-\d{5}\]?")]                                                                            # any leftover id
LEAK = re.compile(r"\b[a-z]{2,8}-\d{5}\b|playbook", re.I)


def clean(resp):
    resp = BULLET_FIELD.sub('"bullet_ids": []', resp, count=1)
    try:
        d = json.loads(resp)
    except Exception:
        return None
    r = str(d.get("reasoning", ""))
    for p in CITE:
        r = p.sub(" " if p is CITE[-1] else "", r)
    r = re.sub(r"\s{2,}", " ", r).strip()
    if LEAK.search(r):
        return None
    d["reasoning"], d["bullet_ids"] = r, []
    return json.dumps(d, indent=2, ensure_ascii=False)


def row(c, cleaned):
    it = train[c["idx"]]
    p = GENERATOR_PROMPT.format("(empty)", "(empty)", it["question"], it["context"])
    return {"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": cleaned}],
            "meta": {"task": TASK, "train_index": c["idx"], "source": c["source"], "leak": c["leak"],
                     "base_pass": c["base_pass"], "nll": c["nll"], "judge": c.get("judge")}}


_pool = f"{DS_DIR}/pool_{TASK}.jsonl"
pool = [json.loads(l) for l in (open(_pool) if os.path.exists(_pool) else gzip.open(_pool + ".gz", "rt"))]
cands = {}
for c in pool:
    if c["correct"] and c["source"] != "base" and "nll" in c:
        cl = clean(c["response"])
        if cl is not None:
            c["clean"] = cl
            cands.setdefault(c["idx"], []).append(c)
rng = random.Random(0)
items = sorted(cands)
best = lambda v: min(v, key=lambda c: (c["leak"], (c.get("judge") or 0) < 8, c["nll"]))   # uncited, then judge>=8, then low nll
sets = {
    "rand": [rng.choice(cands[i]) for i in items],
    "clean": [rng.choice([c for c in cands[i] if not c["leak"]]) for i in items if any(not c["leak"] for c in cands[i])],
    "hard": [min(cands[i], key=lambda c: (c["leak"], c["nll"])) for i in items if cands[i][0]["base_pass"] < 1],
}
sets["hard_ctrl"] = [rng.choice(cands[i]) for i in rng.sample(items, len(sets["hard"]))]
sets["quality"] = [best(cands[i]) for i in items]
# rand + up to 2 extra distinct traces for every hard item (up-weights what the base model cannot do yet)
extra = []
for i in items:
    if cands[i][0]["base_pass"] < 1:
        rest = [c for c in cands[i] if c["clean"] not in {x["clean"] for x in sets["rand"] if x["idx"] == i}]
        extra += rng.sample(rest, min(2, len(rest)))
sets["rand_hardup"] = sets["rand"] + extra
summary = {"task": TASK, "items_with_clean_candidate": len(items)}
for name, sel in sets.items():
    with open(f"{OUT}/{TASK}_{name}.jsonl", "w") as f:
        for c in sel:
            f.write(json.dumps(row(c, c["clean"])) + "\n")
    summary[name] = {"rows": len(sel), "uncited_original": sum(not c["leak"] for c in sel),
                     "mean_base_pass": round(sum(c["base_pass"] for c in sel) / len(sel), 3),
                     "mean_nll": round(sum(c["nll"] for c in sel) / len(sel), 3),
                     "from_teacher": sum(c["source"] == "teacher" for c in sel)}
json.dump(summary, open(f"{OUT}/{TASK}_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
