"""Figures for rsi_poisson.sh runs: <run_dir>/<mode>_r<rate>/{traffic.jsonl[.gz], traffic.jsonl.ft.json, bwd.csv}.

  python plot_rsi_poisson.py <run_dir>   -> <run_dir>/fig_{timeline,cdf,summary}.png
"""
import csv
import datetime
import glob
import gzip
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLOR = {"seq": "#eb6834", "co": "#2a78d6"}
LABEL = {"seq": "sequential (stop serving to train)", "co": "co-serving (DeltaServe)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})


def load(d):
    f = f"{d}/traffic.jsonl"
    rs = [json.loads(l) for l in (open(f) if os.path.exists(f) else gzip.open(f + ".gz", "rt"))]
    ft = json.load(open(f"{d}/traffic.jsonl.ft.json"))
    t0 = ft["t_start"]
    rs = [r for r in rs if not r["error"]]
    bwd = list(csv.DictReader(open(f"{d}/bwd.csv")))
    tasks = {}
    for r in rs:
        if "session" in r:
            a, b = tasks.get((r["session"], r["task"]), (1e18, 0))
            tasks[(r["session"], r["task"])] = (min(a, r["t_arrive"]), max(b, r["t_arrive"] + r["latency"]))
    bt = [datetime.datetime.fromisoformat(b["timestamp"]).timestamp() - t0 for b in bwd]
    btok = [int(b["total_processed_tokens"]) for b in bwd]
    return {"arr": np.array([r["t_arrive"] - t0 for r in rs]), "lat": np.array([r["latency"] for r in rs]),
            "bt": np.array(bt), "btok": np.array(btok) / 1e3,
            "task": np.array([b - a for a, b in tasks.values()]), "ft": {k: v - t0 for k, v in ft.items() if k.startswith("t_")}}


def main():
    run = sys.argv[1]
    res = {}
    for d in sorted(glob.glob(f"{run}/*_r*")):
        m = re.match(r"(co|seq)_r([0-9.]+)$", os.path.basename(d))
        if m and os.path.exists(f"{d}/traffic.jsonl.ft.json"):
            res.setdefault(m.group(2), {})[m.group(1)] = load(d)
    rates = sorted(res, key=float)

    # 1. timeline: request completion time by arrival (top), FT tokens trained (bottom)
    fig, ax = plt.subplots(2, len(rates), figsize=(4.2 * len(rates), 5.6), sharex="col",
                           gridspec_kw={"height_ratios": [1.4, 1]})
    for j, rate in enumerate(rates):
        for mode in ("seq", "co"):
            v = res[rate][mode]
            ax[0, j].scatter(v["arr"], v["lat"], s=9, color=COLOR[mode], alpha=0.75, linewidths=0, label=LABEL[mode])
            ax[1, j].plot(np.r_[v["ft"]["t_ft_start"], v["bt"]], np.r_[0, v["btok"]], color=COLOR[mode], lw=2,
                           drawstyle="steps-post")
            ax[1, j].axvline(v["ft"]["t_ft_end"], color=COLOR[mode], lw=1, ls=":")
        s = res[rate]["seq"]["ft"]
        for a in ax[:, j]:
            a.axvspan(s["t_trigger"], s["t_ft_end"], color=COLOR["seq"], alpha=0.08, lw=0)
        traffic_end = max(res[rate][m]["ft"]["t_traffic_end"] for m in res[rate])
        ax[1, j].axvline(traffic_end, color=MUTED, lw=1, ls="--")
        ax[1, j].text(traffic_end, ax[1, j].get_ylim()[1] * 0.04, " traffic ends", color=MUTED, fontsize=7.5)
        ax[0, j].set_yscale("log")
        ax[0, j].set_title(f"{rate} rps", color=INK, fontsize=10)
        ax[1, j].set_xlabel("time since first arrival (s)")
    ax[0, 0].set_ylabel("request completion time (s)\narrival → done, log")
    ax[1, 0].set_ylabel("FT tokens trained (K)")
    ax[0, 0].text(res[rates[0]]["seq"]["ft"]["t_trigger"] + 3, 150, "seq: serving stopped\nfor the FT job",
                  color=MUTED, fontsize=7.5, va="top")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False, markerscale=2)
    fig.suptitle("ACE formula traffic (Poisson) + the RSI FT job (526K tok) at t=120 s, Qwen3-32B TP=4",
                 y=0.995, fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(f"{run}/fig_timeline.png", dpi=160)

    # 2. CDF of request completion time
    fig, ax = plt.subplots(1, len(rates), figsize=(4.2 * len(rates), 3.2), sharey=True)
    for j, rate in enumerate(rates):
        for mode in ("seq", "co"):
            x = np.sort(res[rate][mode]["lat"])
            ax[j].plot(x, np.arange(1, len(x) + 1) / len(x), color=COLOR[mode], lw=2, label=LABEL[mode])
            p90 = np.percentile(x, 90)
            ax[j].plot([p90], [0.9], "o", ms=5, color=COLOR[mode], mec="white", mew=1.5)
            ax[j].annotate(f"p90 {p90:.1f}s", (p90, 0.9), textcoords="offset points",
                           xytext=(6, -12 if mode == "seq" else 4), fontsize=7.5, color=INK)
        ax[j].set_xscale("log")
        ax[j].set_title(f"{rate} rps", fontsize=10, color=INK)
        ax[j].set_xlabel("request completion time (s), log")
    ax[0].set_ylabel("fraction of requests")
    ax[0].legend(frameon=False, loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(f"{run}/fig_cdf.png", dpi=160)

    # 3. summary: the trade — p90 completion time vs when the adapter is ready
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0))
    x, w = np.arange(len(rates)), 0.36
    for k, mode in enumerate(("seq", "co")):
        p90 = [np.percentile(res[r][mode]["lat"], 90) for r in rates]
        ready = [res[r][mode]["ft"]["t_ft_end"] for r in rates]
        for a, vals, fmt in ((ax[0], p90, "{:.1f}"), (ax[1], ready, "{:.0f}")):
            b = a.bar(x + (k - 0.5) * (w + 0.02), vals, w, color=COLOR[mode], label=LABEL[mode])
            a.bar_label(b, [fmt.format(v) for v in vals], fontsize=7.5, color=INK, padding=2)
    ax[0].set_title("request completion p90 (s) — lower is better", fontsize=9.5, color=INK)
    ax[1].set_title("adapter ready at (s since first arrival)", fontsize=9.5, color=INK)
    for a in ax:
        a.set_xticks(x, [f"{r} rps" for r in rates])
        a.grid(axis="x", visible=False)
    ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{run}/fig_summary.png", dpi=160)
    names = ["timeline", "cdf", "summary"]

    # 4. agent tasks (TraceLab-timed runs only): a user round + its tool loop, first arrival -> last answer
    if all(len(res[r][m]["task"]) for r in rates for m in ("seq", "co")):
        fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
        for a, q in zip(ax, (50, 90)):
            for k, mode in enumerate(("seq", "co")):
                vals = [np.percentile(res[r][mode]["task"], q) for r in rates]
                b = a.bar(x + (k - 0.5) * (w + 0.02), vals, w, color=COLOR[mode], label=LABEL[mode])
                a.bar_label(b, [f"{v:.0f}" for v in vals], fontsize=7.5, color=INK, padding=2)
            a.set_title(f"agent task completion p{q} (s): lower is better", fontsize=9.5, color=INK)
            a.set_xticks(x, [f"{r} rps" for r in rates])
            a.grid(axis="x", visible=False)
        ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")
        fig.tight_layout()
        fig.savefig(f"{run}/fig_tasks.png", dpi=160)
        names.append("tasks")
    print("wrote", *(f"{run}/fig_{n}.png" for n in names))


if __name__ == "__main__":
    main()
