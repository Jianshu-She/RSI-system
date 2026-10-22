"""One-load, two-panel version of fig_timeline for slides: users' wait (top) and training progress (bottom).

  python plot_rsi_poisson_simple.py <run_dir> [rate]   -> <run_dir>/fig_simple_r<rate>.png
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from plot_rsi_poisson import COLOR, INK, MUTED, load

run, rate = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "1")
v = {m: load(f"{run}/{m}_r{rate}") for m in ("seq", "co")}
name = {"seq": "Sequential: pause serving, train, resume", "co": "Co-serving (DeltaServe): train while serving"}
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for m in ("seq", "co"):
    ax[0].scatter(v[m]["arr"], v[m]["lat"], s=10, color=COLOR[m], alpha=0.8, linewidths=0, label=name[m])
    ax[1].plot(np.r_[v[m]["ft"]["t_ft_start"], v[m]["bt"]], np.r_[0, 100 * v[m]["btok"] / v[m]["btok"][-1]],
               color=COLOR[m], lw=2.5, drawstyle="steps-post", label=name[m])
    t = v[m]["ft"]["t_ft_end"]
    ax[1].annotate(f"done at {t:.0f} s", (t, 100), textcoords="offset points", xytext=(4, -14),
                   color=COLOR[m], fontsize=9, fontweight="bold")
s = v["seq"]["ft"]
for a in ax:
    a.axvspan(s["t_trigger"], s["t_ft_end"], color=COLOR["seq"], alpha=0.08, lw=0)
ax[0].annotate("Sequential: users who arrive while\nit trains wait up to 2 minutes",
               (135, 120), xytext=(260, 95), color=COLOR["seq"], fontsize=9,
               arrowprops={"arrowstyle": "->", "color": COLOR["seq"]})
ax[0].text(330, 18, "Co-serving: every user done in a few seconds", color=COLOR["co"], fontsize=9)
ax[0].set_ylabel("how long each user waited (s)")
ax[0].set_title(f"{rate} request/s; a training job starts at t = 120 s (Qwen3-32B, 4 GPUs)", color=INK, fontsize=10.5)
ax[0].legend(frameon=False, loc="upper right", fontsize=8.5, markerscale=2)
ax[1].set_ylabel("training progress (%)")
ax[1].set_xlabel("time (s)")
ax[1].text(290, 3, "Co-serving trains slowly while users are being served\n(our scheduler is too conservative)",
           color=MUTED, fontsize=9)
fig.tight_layout()
fig.savefig(f"{run}/fig_simple_r{rate}.png", dpi=160)
print(f"{run}/fig_simple_r{rate}.png")
