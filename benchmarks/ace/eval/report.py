#!/usr/bin/env python3
"""ACE reproduction — one-shot progress + results report (read-only; stdlib only).

    python3 eval/report.py            # progress of every run + results table
    python3 eval/report.py --json     # same, machine-readable

Every number is read from result files written by the runs (never from memory/notes):
  finance ACE   results/finance/<task>_<mode>_<gt|nogt>[_resumeN]/<ts>/{final_results,initial_test_results,val_results}.json
  finance ICL   results/finance/<task>_icl/<ts>/final_results.json
  finance DC    results/finance_dc/<task>_<gt|nogt>/{steps.jsonl,final.json}
  AppWorld      $APPWORLD_DIR/experiments/outputs/<exp>/{tasks/*/dbs, evaluations/<split>.json}
Paired tests are two-sided sign tests on per-item correctness (same test items, same order).
"""
import glob
import json
import os
import re
import sys
from math import comb

_ROOT = os.environ.get("ACE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WORK = os.environ.get("ACE_WORK_DIR", f"{_ROOT}/work")
RESULTS = os.environ.get("ACE_RESULTS_DIR", f"{_WORK}/results")
LOGS = os.environ.get("ACE_LOGS_DIR", f"{_WORK}/logs")  # fin_<task>_<mode>_<lab>[_resumeN].log (step/window progress)
FIN, DC = f"{RESULTS}/finance", f"{RESULTS}/finance_dc"
AW = os.environ.get("APPWORLD_DIR", f"{os.environ.get('ACE_UPSTREAM_DIR', f'{_WORK}/ace')}/ace-appworld")
PAPER = {  # DeepSeek-V3.1, arXiv 2510.04618 Tables 1-2
    ("formula", "base"): 67.5, ("finer", "base"): 70.7, ("formula", "icl"): 67.0, ("finer", "icl"): 72.3,
    ("formula", "offline_gt"): 85.5, ("finer", "offline_gt"): 78.3, ("formula", "offline_nogt"): 83.0, ("finer", "offline_nogt"): 71.1,
    ("formula", "online_gt"): 76.5, ("finer", "online_gt"): 76.7, ("formula", "online_nogt"): 78.5, ("finer", "online_nogt"): 67.3,
    ("formula", "dc_gt"): 69.5, ("finer", "dc_gt"): 74.2, ("formula", "dc_nogt"): 62.5, ("finer", "dc_nogt"): 68.3,
}
PAPER_AW = {"react": 42.4, "off_gt": 59.4, "off_nogt": 57.2, "online": 59.5, "lora": None}
N_TEST = {"formula": 200, "finer": 441}
N_TRAIN = {"formula": 500, "finer": 1000}


def one(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def load(path):
    return json.load(open(path)) if path and os.path.exists(path) else None


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def wrong(result):  # set of item indices answered wrong, from an ACE *_results.json
    return {e["index"] for e in result["error_log"]["errors"]} if result else None


def last_match(path, regex):
    if not path or not os.path.exists(path):
        return None
    hits = re.findall(regex, open(path, errors="ignore").read())
    return hits[-1] if hits else None


def finance():
    out = {}
    for task in ("formula", "finer"):
        bases, rows = [], {}
        icl_dir = one(f"{FIN}/{task}_icl/*/")
        icl = load(one(f"{icl_dir}test_test_results.json")) if icl_dir else None
        for mode in ("offline", "online"):
            for lab in ("gt", "nogt"):
                key = f"{mode}_{lab}"
                d = one(f"{FIN}/{task}_{key}/*/")
                fr = load(f"{d}final_results.json") if d else None
                init = load(f"{d}initial_test_results.json") if d else None
                if init:
                    bases.append(init["test_results"]["accuracy"] if "test_results" in init else init.get("accuracy"))
                log = f"{LOGS}/fin_{task}_{key}.log"
                row = {"status": "running", "acc": None, "progress": None, "note": ""}
                if mode == "offline":
                    vals = load(f"{d}val_results.json") or [] if d else []
                    trace = {v["step"]: round(100 * v["val_results"]["accuracy"], 1) for v in vals}
                    step = last_match(log, r"Step (\d+)/\d+")
                    res = sorted(glob.glob(f"{FIN}/{task}_{key}_resume*/*/"),
                                 key=lambda p: int(re.search(r"_resume(\d+)", p).group(1)))
                    for seg in res:  # resumed after node losses: steps continue from each snapshot index
                        start = int(re.search(r"_resume(\d+)", seg).group(1))
                        rlog = f"{LOGS}/fin_{task}_{key}_resume{start}.log"
                        rstep = last_match(rlog, r"Step (\d+)/\d+")
                        step = start + int(rstep or 0)
                        for v in load(f"{seg}val_results.json") or []:
                            gstep = start + v["step"]
                            if gstep % 100 == 0:  # selection only on the original eval schedule
                                trace[gstep] = round(100 * v["val_results"]["accuracy"], 1)
                    if res:
                        row["note"] = "resumed@" + ",".join(re.search(r"_resume(\d+)", x).group(1) for x in res)
                    row["progress"] = f"{step or 0}/{N_TRAIN[task]} train steps"
                    row["val_trace"] = trace
                    if trace:
                        best = max(trace, key=lambda s: (trace[s], -s))
                        row["best_val"] = f"{trace[best]}@{best}"
                    if fr and "final_test_results" in fr and not res:
                        row.update(status="done", acc=100 * fr["final_test_results"]["accuracy"])
                        ft = load(f"{d}final_test_results.json")
                        row["vs_base"], row["vs_icl"] = paired(wrong(ft), wrong(init)), paired(wrong(ft), wrong(icl))
                    final_eval = one(f"{FIN}/{task}_{key}_finaltest/*/final_results.json")
                    if final_eval:  # eval_only of the overall best-val playbook (resumed runs)
                        row.update(status="done", acc=100 * load(final_eval)["test_results"]["accuracy"])
                else:
                    if fr and "online_test_results" in fr:
                        row.update(status="done", acc=100 * fr["online_test_results"]["accuracy"])
                    g = last_match(log, r"Global step (\d+)")
                    res = sorted(glob.glob(f"{FIN}/{task}_{key}_resume*/*/"))
                    if res:  # windows 1..k from the original log, k+1.. from the resumed run (rule fixed before results)
                        start = int(re.search(r"_resume(\d+)", res[-1]).group(1))
                        w0 = [float(x) for x in re.findall(r"Window \d+ test accuracy: ([0-9.]+)", open(log).read())][: start // 15]
                        rlog = f"{LOGS}/fin_{task}_{key}_resume{start}.log"
                        w1 = [float(x) for x in re.findall(r"Window \d+ test accuracy: ([0-9.]+)", open(rlog).read())] if os.path.exists(rlog) else []
                        sizes = [min(15, N_TEST[task] - 15 * i) for i in range(len(w0) + len(w1))]
                        done_n = sum(sizes)
                        g = done_n
                        if done_n == N_TEST[task]:
                            row.update(status="done", acc=100 * sum(a * s for a, s in zip(w0 + w1, sizes)) / done_n)
                        row["note"] = f"resumed@{start}"
                    row["progress"] = f"{g or 0}/{N_TEST[task]} test samples"
                rows[key] = row
        if icl:
            rows["icl"] = {"status": "done", "acc": 100 * icl["test_results"]["accuracy"] if "test_results" in icl else 100 * load(f"{icl_dir}final_results.json")["test_results"]["accuracy"], "progress": "eval_only"}
        for lab in ("gt", "nogt"):
            f, s = f"{DC}/{task}_{lab}/final.json", f"{DC}/{task}_{lab}/steps.jsonl"
            n = sum(1 for _ in open(s)) if os.path.exists(s) else 0
            fj = load(f)
            rows[f"dc_{lab}"] = {"status": "done" if fj else ("running" if n else "not started"),
                                 "acc": 100 * fj["accuracy"] if fj else None, "progress": f"{n}/{N_TEST[task]} test samples"}
        rows["base"] = {"status": "done", "acc": 100 * sum(bases) / len(bases) if bases else None,
                        "progress": f"mean of {len(bases)} runs' initial test", "spread": [round(100 * b, 1) for b in bases]}
        out[task] = rows
    return out


def paired(a_wrong, b_wrong):
    if a_wrong is None or b_wrong is None:
        return None
    only_a, only_b = len(b_wrong - a_wrong), len(a_wrong - b_wrong)  # items only A gets right / only B gets right
    return {"only_this_right": only_a, "only_other_right": only_b, "p": round(sign_p(only_a, only_b), 4)}


def appworld():
    split_ids = {s: open(f"{AW}/data/datasets/{s}.txt").read().split() for s in ("train", "test_normal", "test_challenge")}
    exps = {"react": ("qwen_react_eval", None), "off_gt": ("qwen_off_gt_eval", "qwen_off_gt_adapt"),
            "off_nogt": ("qwen_off_nogt_eval", "qwen_off_nogt_adapt"), "online": ("qwen_online_normal|qwen_online_challenge", None),
            "lora": ("qwen_lora_react_eval", None)}
    out = {}
    react = {s: load(f"{AW}/experiments/outputs/qwen_react_eval/evaluations/{s}.json") for s in ("test_normal", "test_challenge")}
    for key, (ev, ad) in exps.items():
        row = {"splits": {}}
        if ad:
            n = sum(os.path.exists(f"{AW}/experiments/outputs/{ad}/tasks/{t}/dbs") for t in split_ids["train"])
            row["adapt"] = f"{n}/{len(split_ids['train'])} train tasks"
        for split in ("test_normal", "test_challenge"):
            exp = ev if "|" not in ev else (ev.split("|")[0] if split == "test_normal" else ev.split("|")[1])
            n = sum(os.path.exists(f"{AW}/experiments/outputs/{exp}/tasks/{t}/dbs") for t in split_ids[split])
            e = load(f"{AW}/experiments/outputs/{exp}/evaluations/{split}.json")
            cell = {"progress": f"{n}/{len(split_ids[split])}"}
            if e and len(e["individual"]) == len(split_ids[split]):
                a = e["aggregate"]
                cell.update(tgc=a["task_goal_completion"], sgc=a["scenario_goal_completion"])
                if key != "react" and react[split]:
                    s1 = {t for t, v in e["individual"].items() if v["success"]}
                    s0 = {t for t, v in react[split]["individual"].items() if v["success"]}
                    cell["vs_react"] = {"only_this": len(s1 - s0), "only_react": len(s0 - s1), "p": round(sign_p(len(s1 - s0), len(s0 - s1)), 4)}
            row["splits"][split] = cell
        cells = [row["splits"][s].get(k) for s in ("test_normal", "test_challenge") for k in ("tgc", "sgc")]
        row["avg"] = round(sum(cells) / 4, 1) if None not in cells else None
        out[key] = row
    return out


def pfmt(p):
    return "p<1e-4" if p < 1e-4 else f"p={p:.3g}"


def fmt(x):
    return "—" if x is None else f"{x:.1f}"


def main():
    fin, aw = finance(), appworld()
    if "--json" in sys.argv:
        print(json.dumps({"finance": fin, "appworld": aw}, indent=1))
        return
    print("=" * 100 + "\nFINANCE (test accuracy %; FiNER per-tag, Formula per-question)   [ours | paper DeepSeek-V3.1]")
    order = ["base", "icl", "offline_gt", "offline_nogt", "online_gt", "online_nogt", "dc_gt", "dc_nogt"]
    for task, rows in fin.items():
        print(f"\n  {task.upper()}")
        for k in order:
            r = rows.get(k, {})
            extra = []
            if r.get("best_val"):
                extra.append(f"best val {r['best_val']}")
            for tag in ("vs_base", "vs_icl"):
                if r.get(tag):
                    extra.append(f"{tag}: +{r[tag]['only_this_right']}/-{r[tag]['only_other_right']} {pfmt(r[tag]['p'])}")
            if r.get("note"):
                extra.append(r["note"])
            print(f"   {k:13s} {fmt(r.get('acc')):>6s} | {fmt(PAPER.get((task, k))):>5s}   {r.get('status', '—'):8s} {r.get('progress') or '':28s} {'; '.join(extra)}")
    print("\n" + "=" * 100 + "\nAPPWORLD (TGC/SGC %, test_normal | test_challenge, avg of 4)   [paper avg]")
    for k, r in aw.items():
        n, c = r["splits"]["test_normal"], r["splits"]["test_challenge"]
        line = f"   {k:9s} {fmt(n.get('tgc'))}/{fmt(n.get('sgc'))} [{n['progress']}] | {fmt(c.get('tgc'))}/{fmt(c.get('sgc'))} [{c['progress']}]  avg {fmt(r['avg'])} | paper {PAPER_AW[k]}"
        if r.get("adapt"):
            line += f"  (adapt {r['adapt']})"
        for s, cell in (("normal", n), ("challenge", c)):
            if cell.get("vs_react"):
                v = cell["vs_react"]
                line += f"  {s} vs ReAct +{v['only_this']}/-{v['only_react']} {pfmt(v['p'])}"
        print(line)


if __name__ == "__main__":
    main()
