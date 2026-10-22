"""Tables for rsi_poisson.sh runs: <run_dir>/<mode>_r<rate>/{traffic.jsonl, traffic.jsonl.ft.json}.

Request completion time = done - arrival (includes being held while a sequential FT job runs).
With --workload runs (TraceLab session timing) also task completion time: a task = a user round + its tool loop,
from the first request's arrival to the last request's done.
  python rsi_poisson_metrics.py <run_dir>   -> <run_dir>/rsi_poisson_metrics.{md,json}
"""
import glob
import gzip
import json
import os
import re
import sys

import numpy as np


def pct(x, qs=(50, 90, 99)):
    return [round(float(np.percentile(x, q)), 3) for q in qs]


def one(d):
    f = f"{d}/traffic.jsonl"
    rs = [json.loads(l) for l in (open(f) if os.path.exists(f) else gzip.open(f + ".gz", "rt"))]
    ok = [r for r in rs if not r["error"]]
    ft = json.load(open(f"{d}/traffic.jsonl.ft.json"))
    t0 = ft["t_start"]
    held = [r for r in ok if r["t_send"] - r["t_arrive"] > 0.5]
    tasks = {}
    for r in ok:
        if "session" in r:
            a, b = tasks.get((r["session"], r["task"]), (1e18, 0))
            tasks[(r["session"], r["task"])] = (min(a, r["t_arrive"]), max(b, r["t_arrive"] + r["latency"]))
    task = {"tasks": len(tasks), "task_time": pct([b - a for a, b in tasks.values()])} if tasks else {}
    return {**task, "n": len(rs), "errors": len(rs) - len(ok), "correct": sum(r["correct"] for r in ok),
            "latency": pct([r["latency"] for r in ok]), "latency_mean": round(float(np.mean([r["latency"] for r in ok])), 3),
            "ttft": pct([r["ttft"] for r in ok if r["ttft"] is not None]),
            "tbt_mean": pct([r["tbt_mean"] for r in ok if r["tbt_mean"] is not None]),
            "held": len(held), "held_wait_mean": round(float(np.mean([r["t_send"] - r["t_arrive"] for r in held])), 1) if held else 0,
            "ft_s": round(ft["t_ft_end"] - ft["t_ft_start"], 1),
            "ft_done_at": round(ft["t_ft_end"] - t0, 1),
            "traffic_done_at": round(ft["t_traffic_end"] - t0, 1),
            "makespan": round(max(ft["t_ft_end"], ft["t_traffic_end"]) - t0, 1),
            "drain_s": round(ft["t_drained"] - ft["t_trigger"], 2) if "t_drained" in ft else None}


def main():
    run = sys.argv[1]
    res = {}
    for d in sorted(glob.glob(f"{run}/*_r*")):
        m = re.match(r"(co|seq)_r([0-9.]+)$", os.path.basename(d))
        if m and os.path.exists(f"{d}/traffic.jsonl.ft.json"):
            res.setdefault(m.group(2), {})[m.group(1)] = one(d)
    json.dump(res, open(f"{run}/rsi_poisson_metrics.json", "w"), indent=1)
    L = ["# RSI traffic (ACE formula) + the RSI FT job: sequential vs co-serving", "",
         "| rate | mode | request completion p50 / p90 / p99 (s) | mean | held (mean wait) | FT time | FT done at | "
         "all done at | TTFT p50 / p99 (ms) | correct |", "|---|---|---|---|---|---|---|---|---|---|"]
    for rate in sorted(res, key=float):
        for mode in ("seq", "co"):
            v = res[rate].get(mode)
            if not v:
                continue
            L.append(f"| {rate} | {mode} | {' / '.join(f'{x:.2f}' for x in v['latency'])} | {v['latency_mean']:.2f} | "
                     f"{v['held']} ({v['held_wait_mean']} s) | {v['ft_s']} s | {v['ft_done_at']} s | {v['makespan']} s | "
                     f"{v['ttft'][0] * 1e3:.0f} / {v['ttft'][2] * 1e3:.0f} | {v['correct']}/{v['n']} |")
    if any("task_time" in v for r in res.values() for v in r.values()):
        L += ["", "| rate | mode | tasks | task completion p50 / p90 / p99 (s) |", "|---|---|---|---|"]
        for rate in sorted(res, key=float):
            for mode in ("seq", "co"):
                v = res[rate].get(mode)
                if v and "task_time" in v:
                    L.append(f"| {rate} | {mode} | {v['tasks']} | {' / '.join(f'{x:.1f}' for x in v['task_time'])} |")
    open(f"{run}/rsi_poisson_metrics.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
