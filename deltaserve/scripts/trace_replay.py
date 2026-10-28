"""Replay the TraceLab trace's REAL request timeline (arXiv:2606.30560): every Claude round's real start time in one
window of --hours of wall clock, compressed by --hours*3600/--duration into a --duration s run.

agent_trace.py keeps each session's real gaps but starts sessions as a Poisson process; at serving rates (1-2 rps)
that stacks dozens of sessions and smooths the traffic to Poisson-like (10 s-window CV 0.3-0.5, no idle windows),
while the trace itself is very bursty (busy hours: CV 1.8-2.7, 70-84 % idle 10 s windows, peak 10-16x the mean).
This keeps the real arrival process instead: which sessions run, when each round starts, and how rounds cluster.

Window: among the windows (stepped by 30 min) holding --requests +-25 % rounds, the one whose compressed 10 s-window
CV is the median (a typical window, not the burstiest). Each round carries "t" = its compressed start (s from the
run start); rsi_traffic.py sends it at t, or right after the session's previous answer if that is later (closed
loop within a session), and "wait" = the compressed gap, for drivers that only know waits.

  python trace_replay.py --db syfi_coding_trace.duckdb --hours 24 --requests 1125 --duration 750 --out wl.json
"""
import argparse
import json

import duckdb
import numpy as np

Q = """
with r as (select round_pk, session_id, round_index, first_input_event_type fit from rounds where provider = 'claude'),
     t as (select round_pk, min(epoch_us(timestamp)) / 1e6 t0 from timing_events group by 1)
select r.*, t.t0 from r join t using (round_pk) order by t0
"""


def cv10(y, duration):
    h = np.bincount((y // 10).astype(int), minlength=int(duration // 10))[:int(duration // 10)] / 10
    return h.std() / h.mean(), (h == 0).mean(), h.max() / h.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--requests", type=int, default=1125)
    ap.add_argument("--duration", type=float, default=750)
    ap.add_argument("--rank", type=float, default=0.5, help="pick the window at this CV quantile (0.5 = median)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = duckdb.connect(a.db, read_only=True).execute(Q).df()
    t = df.t0.to_numpy()
    W, k = a.hours * 3600, a.hours * 3600 / a.duration
    cands = []
    for s in np.arange(t[0], t[-1] - W, 1800):
        lo, hi = np.searchsorted(t, [s, s + W])
        if abs((hi - lo) - a.requests) <= 0.25 * a.requests:
            cands.append((cv10((t[lo:hi] - s) / k, a.duration)[0], s))
    cands.sort()
    cv, s = cands[int(a.rank * (len(cands) - 1))]
    w = df[(df.t0 >= s) & (df.t0 < s + W)].copy()
    w["t"] = (w.t0 - s) / k
    sessions = []
    for sid, g in w.groupby("session_id", sort=False):
        g = g.sort_values("t")
        ts = g.t.to_numpy()
        rounds = [{"kind": "user" if f == "user_message" else "tool", "t": round(float(x), 3),
                   "wait": round(float(x - (ts[i - 1] if i else x)), 3)}
                  for i, (x, f) in enumerate(zip(ts, g.fit))]
        rounds[0]["kind"] = "user"   # a window can open mid-task; its first round starts the session's first task
        sessions.append({"trace_session": str(sid), "arrival": round(float(ts[0]), 3), "rounds": rounds})
    sessions.sort(key=lambda x: x["arrival"])
    for i, x in enumerate(sessions):
        x["id"] = i
    cvv, idle, pm = cv10(w.t.to_numpy(), a.duration)
    meta = {"source": "TraceLab v0.0.1 (claude rounds), real timeline replay", "args": vars(a),
            "window_start_utc": str(np.datetime64(int(s), "s")), "compression": round(k, 1),
            "candidate_windows": len(cands), "requests": len(w), "sessions": len(sessions),
            "mean_rps": round(len(w) / a.duration, 2), "cv_10s": round(cvv, 2), "idle_10s_windows": round(idle, 2),
            "peak_over_mean_10s": round(pm, 1)}
    json.dump({"meta": meta, "sessions": sessions}, open(a.out, "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
