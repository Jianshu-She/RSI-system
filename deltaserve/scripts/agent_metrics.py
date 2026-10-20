"""Sequential vs co-serving completion times from agent_run.sh outputs (<run>/{inf,ft,co}).

Agent task = one user request + its tool loop: from sending its user round to the end of its last LLM round
(the real tool waits in between are the trace's, identical in every mode, so differences are LLM-side).
  sequential, inference first : tasks as in `inf`;  FT done at inf makespan + ft time
  sequential, training first  : FT done at ft time; a session arriving before then starts when FT ends, so its first
                                task completes max(0, ft - arrival) later (later tasks shift with it)
  co-serving                  : both measured on one server; makespan = max(replay end, FT end)
  python agent_metrics.py <run_dir>   -> <run_dir>/agent_metrics.{md,json}
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np


def pct(x, qs=(50, 90, 99)):
    return [round(float(np.percentile(x, q)), 3) for q in qs] if len(x) else None


def num(path):
    return float(open(path).read().split()[0]) if os.path.exists(path) else None


def tasks(rounds):
    t = defaultdict(list)
    for r in rounds:
        t[(r["session"], r["task"])].append(r)
    out = []
    for (s, k), rs in sorted(t.items()):
        rs.sort(key=lambda r: r["round"])
        out.append({"session": s, "task": k, "start": rs[0]["t_send"], "end": rs[-1]["t_done"],
                    "llm": sum(r["e2e"] for r in rs), "rounds": len(rs)})
    for x in out:
        x["time"] = x["end"] - x["start"]
    return out


def gpu_sm(path, gpus, t0, t1):
    if not path or not os.path.exists(path):
        return None
    v = []
    for ln in open(path):
        p = ln.split()
        if len(p) >= 9 and p[1] == "GPU" and p[2] in gpus and t0 <= float(p[0]) <= t1:
            try:
                v.append(float(p[6]))
            except ValueError:
                pass
    return round(100 * float(np.mean(v)), 1) if v else None


def mode(run, m):
    d = os.path.join(run, m)
    if not os.path.isdir(d):
        return None
    r = {"dir": d}
    kv = re.search(r"GPU KV cache size: ([0-9,]+) tokens", open(f"{d}/server.log").read())
    r["kv_tokens"] = int(kv.group(1).replace(",", "")) if kv else None
    gpus = re.search(r"on GPUs ([0-9,]+)", open(f"{d}/driver.log").read()).group(1).split(",")
    if os.path.exists(f"{d}/rounds.jsonl"):
        rounds = [json.loads(l) for l in open(f"{d}/rounds.jsonl")]
        ok = [x for x in rounds if not x["error"]]
        t0, t1 = num(f"{d}/t_replay_start"), num(f"{d}/t_replay_end")
        tk = tasks(ok)
        r.update({"rounds": len(rounds), "errors": len(rounds) - len(ok), "replay_s": t1 - t0, "t0": t0,
                  "compactions": sum(x["compacted"] for x in rounds),
                  "ttft": pct([x["ttft"] for x in ok if x["ttft"] is not None]),
                  "tbt_mean": pct([x["tbt_mean"] for x in ok if x["tbt_mean"] is not None]),
                  "tbt_p99_round": pct([x["tbt_p99"] for x in ok if x["tbt_p99"] is not None]),
                  "round_e2e": pct([x["e2e"] for x in ok]), "task_n": len(tk),
                  "task_time": pct([x["time"] for x in tk]), "task_llm": pct([x["llm"] for x in tk]),
                  "task_time_mean": round(float(np.mean([x["time"] for x in tk])), 2),
                  "sm_active": gpu_sm(f"{d}/gpu.txt", gpus, t0, t1), "_tasks": tk, "_rounds": ok})
    fs, fe = num(f"{d}/t_ft_start"), num(f"{d}/t_ft_end")
    if fs and fe:
        r["ft_s"] = fe - fs
        r["ft_start"] = fs
        if os.path.exists(f"{d}/bwd.csv"):
            rows = [l.split(",") for l in open(f"{d}/bwd.csv").read().splitlines()[1:] if l]
            r["ft_tokens"] = int(rows[-1][5]) if rows else None
            r["ft_cycles"] = len(rows)
            r["ft_loss_first_last"] = [float(rows[0][4]), float(rows[-1][4])] if rows else None
            r["ft_tok_s"] = round(r["ft_tokens"] / r["ft_s"], 1) if rows else None
    return r


def main():
    run = sys.argv[1]
    inf, ft, co = (mode(run, m) for m in ("inf", "ft", "co"))
    res = {"inf": inf, "ft": ft, "co": co}
    seq_if = inf["replay_s"] + ft["ft_s"]
    co_mk = max(co["t0"] + co["replay_s"], co["ft_start"] + co["ft_s"]) - min(co["t0"], co["ft_start"])
    # training first: each task of a session that arrived before FT finished shifts by max(0, ft - arrival)
    arr = {}
    for x in inf["_tasks"]:
        arr.setdefault(x["session"], x["start"] - inf["t0"]) if x["task"] == 0 else None
    shifted = [x["time"] + (max(0.0, ft["ft_s"] - arr.get(x["session"], 1e9)) if x["task"] == 0 else 0.0)
               for x in inf["_tasks"]]
    # (a delayed session's later tasks keep their own duration; only the first one waits)
    res["summary"] = {
        "sequential_inf_then_ft": {"makespan_s": seq_if, "ft_done_s": seq_if, "task_time": inf["task_time"]},
        "sequential_ft_then_inf": {"makespan_s": seq_if, "ft_done_s": ft["ft_s"], "task_time": pct(shifted),
                                   "tasks_delayed": sum(1 for x in inf["_tasks"] if x["task"] == 0
                                                        and arr.get(x["session"], 1e9) < ft["ft_s"])},
        "co_serving": {"makespan_s": co_mk, "ft_done_s": co["ft_start"] + co["ft_s"] - min(co["t0"], co["ft_start"]),
                       "task_time": co["task_time"]},
        "co_vs_seq_makespan": round(co_mk / seq_if, 3)}
    for m in (inf, co):
        m.pop("_tasks", None)
        m.pop("_rounds", None)
    json.dump(res, open(f"{run}/agent_metrics.json", "w"), indent=1)
    S = res["summary"]
    L = ["# agent-1: sequential vs co-serving (TraceLab agent workload, Qwen3-32B TP=4)", "",
         "| | makespan (both jobs) | FT job done at | agent task completion p50 / p90 / p99 (s) |", "|---|---|---|---|"]
    for k, n in (("sequential_inf_then_ft", "sequential: inference, then FT"),
                 ("sequential_ft_then_inf", "sequential: FT, then inference"), ("co_serving", "co-serving")):
        v = S[k]
        L.append(f"| {n} | {v['makespan_s']:.0f} s | {v['ft_done_s']:.0f} s | "
                 + " / ".join(f"{t:.1f}" for t in v["task_time"]) + " |")
    L += ["", f"co / sequential makespan = {S['co_vs_seq_makespan']}", "",
          "| | inf only | co-serving |", "|---|---|---|"]
    for k in ("kv_tokens", "rounds", "errors", "compactions", "replay_s", "task_n", "task_time", "task_time_mean",
              "task_llm", "round_e2e", "ttft", "tbt_mean", "tbt_p99_round", "sm_active"):
        L.append(f"| {k} | {inf.get(k)} | {co.get(k)} |")
    L += ["", "| FT | alone | co-serving |", "|---|---|---|"]
    for k in ("ft_s", "ft_tokens", "ft_cycles", "ft_tok_s", "ft_loss_first_last"):
        L.append(f"| {k} | {ft.get(k)} | {co.get(k)} |")
    open(f"{run}/agent_metrics.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
