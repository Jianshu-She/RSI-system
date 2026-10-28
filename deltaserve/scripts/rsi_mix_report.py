"""Tables + figures for rsi_mix.sh runs: <run_dir>/<cell>_r<rate>/{traffic.jsonl[.gz], traffic.jsonl.ft.json}.

  python rsi_mix_report.py <run_dir>   -> <run_dir>/rsi_mix.{md,json}, <run_dir>/fig_mix_r<rate>.png
  An optional <run_dir>/labels.json {"<rate>": "<label>"} names the load in tables and titles (default "<rate> rps").

Request completion time = done - arrival (incl. a seq hold). Agent task = a user round + its tool loop (first
arrival -> last done). Round wait = round k trained - round k started (the RSI loop's turnaround: how long the
new data waits before it is in the adapter). FT share = tokens trained while traffic ran / (that time x the
seq cell's standalone rate).
"""
import glob
import gzip
import json
import os
import re
import sys

import numpy as np

CELLS = {"seq": ("sequential (stop serving to train)", "#eb6834"),
         "co": ("DeltaServe co-serving", "#2a78d6"),
         "costrict": ("DeltaServe co-serving, strict SLO", "#8fb8e8"),
         "cofix": ("DeltaServe co-serving, scheduler fixes", "#0b3d91"),
         "coloose": ("DeltaServe co-serving, fixes + loose SLO", "#1a9e77"),
         "cofast": ("DeltaServe co-serving, fixes + loose SLO + fast backward", "#c2185b"),
         "coguard": ("DeltaServe co-serving, fast backward + slow-step guard", "#e6a100"),
         "cofastb": ("DeltaServe co-serving, fast backward (repeat)", "#e88bb3"),
         "seqfast": ("sequential, fast backward", "#f2a07b"),
         "tt": ("vLLM + torchtune (same GPUs, time-sliced)", "#6f4bb8")}


def pct(x, qs=(50, 90, 99)):
    return [round(float(np.percentile(x, q)), 3) for q in qs] if len(x) else [None] * len(qs)


def load(d):
    f = f"{d}/traffic.jsonl"
    rs = [json.loads(l) for l in (open(f) if os.path.exists(f) else gzip.open(f + ".gz", "rt"))]
    ft = json.load(open(f"{d}/traffic.jsonl.ft.json"))
    t0 = ft["t_start"]
    ok = [r for r in rs if not r["error"]]
    tasks = {}
    for r in ok:
        a, b = tasks.get((r["session"], r["agent_task"]), (1e18, 0))
        tasks[(r["session"], r["agent_task"])] = (min(a, r["t_arrive"]), max(b, r["t_arrive"] + r["latency"]))
    rounds = [{**rd, **{k: rd[k] - t0 for k in rd if k.startswith("t_")}} for rd in ft["rounds"]]
    prog = []  # (t, cumulative tokens)
    if os.path.exists(f"{d}/bwd.csv"):
        import csv
        import datetime
        for b in csv.DictReader(open(f"{d}/bwd.csv")):
            prog.append((datetime.datetime.fromisoformat(b["timestamp"]).timestamp() - t0,
                         int(b["total_processed_tokens"])))
    elif os.path.exists(f"{d}/progress.txt"):
        prog = [(float(l.split()[0]) - t0, int(l.split()[1])) for l in open(f"{d}/progress.txt")]
    return {"ok": ok, "n": len(rs), "t0": t0, "tasks": [b - a for a, b in tasks.values()], "rounds": rounds,
            "prog": prog, "traffic_end": ft["t_traffic_end"] - t0}


def summarize(v, standalone):
    ok = v["ok"]
    lat = [r["latency"] for r in ok]
    held = [r for r in ok if r["t_send"] - r["t_arrive"] > 0.5]
    waits = [rd["t_done"] - rd["t_trigger"] for rd in v["rounds"]]
    t_end = v["traffic_end"]
    first = v["rounds"][0]["t_trigger"]
    during = max([n for t, n in v["prog"] if t <= t_end] or [0])
    out = {"requests": v["n"], "ok": len(ok),
           "latency_p50_p90_p99": pct(lat), "latency_mean": round(float(np.mean(lat)), 3),
           "held": len(held), "held_wait_mean": round(float(np.mean([r["t_send"] - r["t_arrive"] for r in held])), 1) if held else 0,
           "ttft_ms_p50_p99": [round(1e3 * x, 1) for x in pct([r["ttft"] for r in ok if r["ttft"]], (50, 99))],
           "tbt_max_ms_p50_p99": [round(1e3 * x, 1) for x in pct([r["tbt_max"] for r in ok if r["tbt_max"]], (50, 99))],
           "tbt_mean_ms_p50": round(1e3 * float(np.median([r["tbt_mean"] for r in ok if r["tbt_mean"]])), 1),
           "task_p50_p90": pct(v["tasks"], (50, 90)),
           "round_wait_s": [round(w, 1) for w in waits], "round_wait_mean": round(float(np.mean(waits)), 1),
           "all_rounds_done_at": round(v["rounds"][-1]["t_done"], 1),
           "ft_tokens_during_traffic": during,
           "correct": {t: f"{sum(r['correct'] for r in ok if r['task'] == t)}/{sum(r['task'] == t for r in ok)}"
                       for t in sorted({r["task"] for r in ok})}}
    if standalone:
        out["ft_share_during_traffic"] = round(during / ((t_end - first) * standalone), 3)
    return out


def label(run, rate):
    f = f"{run}/labels.json"
    return json.load(open(f)).get(rate, f"{rate} rps") if os.path.exists(f) else f"{rate} rps"


def main():
    run = sys.argv[1]
    res = {}
    for d in sorted(glob.glob(f"{run}/*_r*")):
        m = re.match(r"([a-z]+)_r([0-9.]+)$", os.path.basename(d))
        if m and m.group(1) in CELLS and os.path.exists(f"{d}/traffic.jsonl.ft.json"):
            res.setdefault(m.group(2), {})[m.group(1)] = load(d)
    table = {}
    for rate, cells in sorted(res.items(), key=lambda x: float(x[0])):
        sa = None
        if "seq" in cells:  # standalone FT rate = tokens / (sum of the seq rounds' training time)
            s = cells["seq"]
            sa = s["rounds"][-1]["target_cum"] / sum(rd["t_done"] - rd["t_ft_start"] for rd in s["rounds"])
        table[rate] = {c: summarize(v, sa) for c, v in cells.items()}
        if sa:
            table[rate]["standalone_ft_tok_s"] = round(sa)
    json.dump(table, open(f"{run}/rsi_mix.json", "w"), indent=1)
    L = ["# Mixed RSI traffic (ACE formula + finer, TraceLab timing) + 4 FT rounds", "",
         "| load | system | request p50 / p90 / p99 (s) | held | TTFT p50 / p99 (ms) | max-TBT p50 / p99 (ms) | "
         "agent task p50 / p90 (s) | round wait (s) per round | mean | all rounds done (s) | FT share | correct formula / finer |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rate, cells in table.items():
        for c in [c for c in CELLS if c in cells]:
            s = cells[c]
            L.append(f"| {label(run, rate)} | {c} | {' / '.join(map(str, s['latency_p50_p90_p99']))} | {s['held']} | "
                     f"{' / '.join(map(str, s['ttft_ms_p50_p99']))} | {' / '.join(map(str, s['tbt_max_ms_p50_p99']))} | "
                     f"{' / '.join(map(str, s['task_p50_p90']))} | {', '.join(map(str, s['round_wait_s']))} | "
                     f"{s['round_wait_mean']} | {s['all_rounds_done_at']} | {s.get('ft_share_during_traffic', '-')} | "
                     f"{s['correct'].get('formula', '-')} / {s['correct'].get('finer', '-')} |")
        if "standalone_ft_tok_s" in cells:
            L.append(f"| {label(run, rate)} | (standalone FT, from seq) | {cells['standalone_ft_tok_s']} tok/s |||||||||||")
    open(f"{run}/rsi_mix.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    plot(run, res, ["seq", "cofast", "coguard", "tt"], "fig_mix")       # headline: final DeltaServe vs the two baselines
    plot(run, res, list(CELLS), "fig_mix_all")               # every variant


def plot(run, res, want, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})
    for rate, cells in res.items():
        order = [c for c in want if c in cells]
        if not order:
            continue
        fig, ax = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
        for c in order:
            v, (lab, col) = cells[c], CELLS[c]
            ax[0].scatter([r["t_arrive"] - v["t0"] for r in v["ok"]], [r["latency"] for r in v["ok"]], s=7,
                          color=col, alpha=0.6, linewidths=0, label=lab)
            if v["prog"]:
                t, n = zip(*([(v["rounds"][0]["t_ft_start"], 0)] + v["prog"]))
                ax[1].plot(t, np.array(n) / 1e3, color=col, lw=2, drawstyle="steps-post", label=lab)
        any_cell = cells[order[0]]
        for rd in any_cell["rounds"]:
            for a in ax:
                a.axvline(rd["t_trigger"], color=MUTED, lw=0.8, ls=":")
            ax[1].axhline(rd["target_cum"] / 1e3, color=GRID, lw=1, zorder=0)
            ax[1].text(rd["t_trigger"] + 4, rd["target_cum"] / 1e3,
                       f"round {rd['k'] + 1} data", color=MUTED, fontsize=7.5, va="bottom")
        if "seq" in order:
            for rd in cells["seq"]["rounds"]:
                for a in ax:
                    a.axvspan(rd["t_trigger"], rd["t_done"], color=CELLS["seq"][1], alpha=0.08, lw=0)
        ax[0].set_yscale("log")
        ax[0].set_ylabel("request completion time (s)\narrival -> done, log")
        ax[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), markerscale=2.5, fontsize=8)
        ax[0].set_title(f"{label(run, rate)}, mixed traffic (formula + finer); dotted lines = a new FT round's data arrives",
                        fontsize=9.5, color=INK, loc="left")
        ax[1].set_ylabel("FT tokens trained (K)")
        ax[1].set_xlabel("time since first arrival (s)")
        ax[1].set_title("training progress: a round is done when the curve reaches its gray line",
                        fontsize=9.5, color=INK, loc="left")
        fig.tight_layout()
        fig.savefig(f"{run}/{name}_r{rate}.png", dpi=160)
        print("wrote", f"{run}/{name}_r{rate}.png")


if __name__ == "__main__":
    main()
