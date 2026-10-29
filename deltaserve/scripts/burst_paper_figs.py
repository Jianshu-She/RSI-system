"""Paper figures for RESULTS.md §8 (rsi_burst): request-latency CDF and the training-vs-latency trade-off.

  python burst_paper_figs.py results/rsi_burst_2026-10-28   -> fig_cdf.{png,pdf}, fig_tradeoff.{png,pdf}

Reads <run>/<cell>_r<k>/traffic.jsonl.gz (latency = arrival -> done, incl. a seq hold) and traffic.jsonl.ft.json
(round wait = data arrives -> trained).
"""
import gzip
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CELLS = [("seq", "Stop serving to train", "#d1495b", "--", "s"),
         ("coguard", "DeltaServe", "#0f7b6c", "-", "o"),
         ("tt", "vLLM + torchtune", "#6f4bb8", ":", "^")]
LOADS = [("1", "r1: 24 h window, 1.15 rps"), ("2", "r2: 12 h window, 1.78 rps")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(run, cell, k):
    d = f"{run}/{cell}_r{k}"
    lat = np.array([json.loads(l)["latency"] for l in gzip.open(f"{d}/traffic.jsonl.gz", "rt")])
    ft = json.load(open(f"{d}/traffic.jsonl.ft.json"))
    waits = [r["t_done"] - r["t_trigger"] for r in ft["rounds"] if "t_done" in r]
    return lat, float(np.mean(waits))


def style():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "pdf.fonttype": 42})


def cdf(run, data):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (k, title) in zip(axes, LOADS):
        for cell, label, color, ls, _ in CELLS:
            lat = np.sort(data[k, cell][0])
            y = np.arange(1, len(lat) + 1) / len(lat)
            ax.plot(lat, y, color=color, ls=ls, lw=2, label=label)
            p99 = np.percentile(lat, 99)
            ax.plot([p99], [0.99], marker="o", ms=5, color=color, mec="white", mew=1.2, zorder=3)
            ax.annotate(f"p99 {p99:.0f} s", (p99, 0.99), xytext=(4, -12), textcoords="offset points",
                        fontsize=7.5, color=MUTED)
        ax.set_xscale("log")
        ax.set_xlim(0.8, 200)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("request completion time (s, log)")
    axes[0].set_ylabel("fraction of requests")
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{run}/fig_cdf.{ext}", dpi=200)


def tradeoff(run, data):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (k, title) in zip(axes, LOADS):
        for cell, label, color, _, marker in CELLS:
            lat, wait = data[k, cell]
            p99 = np.percentile(lat, 99)
            ax.scatter([wait], [p99], s=90, color=color, marker=marker, edgecolors="white", linewidths=1.5,
                       zorder=3, label=label)
            ax.annotate(label, (wait, p99), xytext=(8, 4), textcoords="offset points", fontsize=8, color=INK)
        ax.set_xlim(0, 125)
        ax.set_ylim(0, 145)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("mean round wait: new data -> trained (s)")
        ax.annotate("better", xy=(4, 4), xytext=(14, 24), fontsize=8, color=MUTED,
                    arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1})
    axes[0].set_ylabel("request p99 (s)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{run}/fig_tradeoff.{ext}", dpi=200)


def main():
    run = sys.argv[1]
    style()
    data = {(k, c[0]): load(run, c[0], k) for k, _ in LOADS for c in CELLS}
    for (k, c), (lat, w) in sorted(data.items()):
        print(f"r{k} {c:8s} n={len(lat)} p50={np.percentile(lat, 50):.2f} p99={np.percentile(lat, 99):.1f} "
              f"round_wait={w:.1f}")
    cdf(run, data)
    tradeoff(run, data)


if __name__ == "__main__":
    main()
