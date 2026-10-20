"""Build an agentic (coding-agent) workload from the TraceLab trace (arXiv:2606.30560, github.com/uw-syfi/TraceLab).

Samples real Claude Code sessions from the released DuckDB and keeps, per LLM round, its real shape:
  append  = newly appended input tokens (tool results / user message)    -> scaled by --scale
  output  = output tokens                                                -> kept (clipped to --max-output)
  prefix0 = the context the session starts with (first round's prefix)   -> scaled by --scale
  wait    = idle time before this round: the real tool latency (tool-initiated round) or the real human think time
            (user-initiated round), measured from the round timestamps; capped at --human-cap / --tool-cap s (a
            "tool" gap of an hour is a person away from a pending permission prompt, not a tool)
  kind    = "tool" | "user"; a *task* is one user request = a user round + the tool loop that follows it
Sessions arrive as a Poisson process (the paper releases no arrival process; csv_export does the same).
Each session is cut at its first --max-tasks user tasks and --max-rounds rounds (whole tasks only).

  python agent_trace.py --db syfi_coding_trace.duckdb --sessions 32 --rate 0.035 --out agent_workload.json
"""
import argparse
import json
import random

import duckdb
import numpy as np

Q = """
with r as (select round_pk, session_id, round_index, first_input_event_type fit, prefix_tokens p,
                  newly_append_tokens a, output_tokens o from rounds where provider = 'claude'),
     t as (select round_pk, min(epoch_us(timestamp)) t0, max(epoch_us(timestamp)) t1 from timing_events group by 1)
select r.*, t.t0, t.t1 from r join t using (round_pk) order by session_id, round_index, round_pk
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--sessions", type=int, default=32)
    ap.add_argument("--rate", type=float, default=0.035, help="session arrivals per second (Poisson)")
    ap.add_argument("--scale", type=float, default=0.125, help="context-token scale (TraceLab contexts are 100K+)")
    ap.add_argument("--human-cap", type=float, default=300.0, help="cap on one human think gap, s")
    ap.add_argument("--tool-cap", type=float, default=300.0, help="cap on one tool gap, s (paper P99: 3.2 min)")
    ap.add_argument("--max-tasks", type=int, default=4)
    ap.add_argument("--max-rounds", type=int, default=80)
    ap.add_argument("--max-output", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = duckdb.connect(a.db, read_only=True).execute(Q).df()
    df["gap"] = ((df.t0 - df.groupby("session_id").t1.shift()) / 1e6).fillna(0).clip(lower=0)
    sess = []
    for sid, g in df.groupby("session_id", sort=True):
        g = g[g.fit.isin(["user_message", "tool_result"])]
        if g.empty or g.fit.iloc[0] != "user_message":
            continue
        rounds, tasks = [], 0
        for r in g.itertuples():
            user = r.fit == "user_message"
            if user:
                tasks += 1
                if tasks > a.max_tasks:
                    break
            if len(rounds) >= a.max_rounds and user:
                break
            rounds.append({"kind": "user" if user else "tool",
                           "wait": 0.0 if not rounds else round(min(r.gap, a.human_cap if user else a.tool_cap), 3),
                           "append": max(1, int(r.a * a.scale)), "output": int(min(max(r.o, 1), a.max_output))})
        if len(rounds) >= a.max_rounds and rounds[-1]["kind"] == "tool":  # keep whole tasks only
            while rounds and rounds[-1]["kind"] == "tool":
                rounds.pop()
            rounds = rounds[:-1] if rounds else rounds
        if rounds:
            sess.append({"trace_session": str(sid), "prefix0": int(g.p.iloc[0] * a.scale), "rounds": rounds})
    rng = random.Random(a.seed)
    pick = rng.sample(sess, a.sessions)
    t = 0.0
    for i, s in enumerate(pick):
        s["id"] = i
        s["arrival"] = round(t, 3)
        t += rng.expovariate(a.rate)
    rs = [r for s in pick for r in s["rounds"]]
    hw = [r["wait"] for r in rs if r["kind"] == "user" and r["wait"] > 0]
    tw = [r["wait"] for r in rs if r["kind"] == "tool"]
    meta = {"source": "TraceLab v0.0.1 (claude rounds)", "args": vars(a), "pool_sessions": len(sess),
            "sessions": len(pick), "rounds": len(rs), "tasks": sum(r["kind"] == "user" for r in rs),
            "append_p50_p90": [int(np.percentile([r["append"] for r in rs], q)) for q in (50, 90)],
            "output_p50_p90": [int(np.percentile([r["output"] for r in rs], q)) for q in (50, 90)],
            "tool_wait_p50_p90_p99": [round(float(np.percentile(tw, q)), 2) for q in (50, 90, 99)] if tw else None,
            "human_wait_p50_p90": [round(float(np.percentile(hw, q)), 1) for q in (50, 90)] if hw else None,
            "last_arrival_s": pick[-1]["arrival"]}
    json.dump({"meta": meta, "sessions": pick}, open(a.out, "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
