"""Score evaluation runs and compare them on the same tasks.

Usage:
  python tools/report.py runs/A runs/B ...                      # per-run table
  python tools/report.py --baseline runs/A runs/B runs/C        # + paired comparison of each run vs the baseline
  python tools/report.py --split test runs/A ...                # restrict to splits/test_ids.txt
  python tools/report.py --json out.json ...                    # also write the numbers

Success = official WebArena reward (summary_info.json cum_reward > 0). Tasks that crashed count as failures.
Paired comparison uses only tasks both runs finished; p = one-sided exact sign test on the discordant tasks.
"""
import argparse, glob, json, math, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(run_dir):
    rows = {}
    for p in glob.glob(os.path.join(run_dir, "webarena.*", "summary_info.json")):
        d = os.path.dirname(p)
        s = json.load(open(p))
        judge = None
        for j in glob.glob(os.path.join(d, "*_autoeval.json")):
            judge = json.load(open(j))[0].get("rm")
        rows[int(d.rsplit(".", 1)[1])] = {
            "success": (s.get("cum_reward") or 0) > 0, "steps": s.get("n_steps") or 0,
            "error": bool(s.get("err_msg")), "judge": judge}
    return rows


def sign_test(wins, losses):
    n = wins + losses
    return sum(math.comb(n, i) for i in range(wins, n + 1)) / 2 ** n if n else 1.0


def summarize(name, rows, expected):
    n = len(rows)
    if not n:
        return {"run": name, "done": 0, "expected": expected}
    succ = [r for r in rows.values() if r["success"]]
    judged = [r for r in rows.values() if r["judge"] is not None]
    return {"run": name, "done": n, "expected": expected,
            "success_rate": 100 * len(succ) / n,
            "avg_steps": sum(r["steps"] for r in rows.values()) / n,
            "avg_steps_success": sum(r["steps"] for r in succ) / max(len(succ), 1),
            "errors": sum(r["error"] for r in rows.values()),
            "self_judge_acc": 100 * sum((r["judge"] is True) == r["success"] for r in judged) / len(judged) if judged else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--baseline", default=None, help="run dir to compare every other run against")
    ap.add_argument("--split", default=None, help="restrict to splits/<split>_ids.txt")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    keep = None
    if a.split:
        keep = {int(x) for x in open(os.path.join(ROOT, "splits", f"{a.split}_ids.txt")).read().split()}
    runs = ([a.baseline] if a.baseline else []) + [r for r in a.runs if r != a.baseline]
    data, out = {}, {"runs": [], "paired": []}
    for r in runs:
        rows = load(r)
        meta_p = os.path.join(r, "meta.json")
        exp = json.load(open(meta_p)).get("n_tasks") if os.path.exists(meta_p) else None
        if keep is not None:
            rows = {t: v for t, v in rows.items() if t in keep}
            exp = len(keep)
        data[r] = rows
        out["runs"].append(summarize(os.path.basename(r.rstrip("/")), rows, exp))
    print(f"{'run':28s} {'done':>9s} {'success%':>9s} {'steps':>6s} {'steps(succ)':>11s} {'errors':>6s} {'self-judge acc':>14s}")
    for s in out["runs"]:
        if not s["done"]:
            print(f"{s['run']:28s} {0:>4d}/{s['expected']}"); continue
        ja = f"{s['self_judge_acc']:.1f}" if s["self_judge_acc"] is not None else "-"
        print(f"{s['run']:28s} {s['done']:>4d}/{str(s['expected'] or '?'):<4s} {s['success_rate']:9.1f} {s['avg_steps']:6.2f} "
              f"{s['avg_steps_success']:11.2f} {s['errors']:6d} {ja:>14s}")
    if a.baseline:
        b = data[a.baseline]
        print(f"\npaired vs {os.path.basename(a.baseline.rstrip('/'))} (same tasks only):")
        for r in runs[1:]:
            x = data[r]; common = sorted(set(b) & set(x)); n = len(common)
            if not n:
                continue
            w = sum(x[t]["success"] and not b[t]["success"] for t in common)
            l = sum(b[t]["success"] and not x[t]["success"] for t in common)
            sb = 100 * sum(b[t]["success"] for t in common) / n; sx = 100 * sum(x[t]["success"] for t in common) / n
            p = {"run": os.path.basename(r.rstrip("/")), "shared": n, "baseline_sr": sb, "run_sr": sx, "delta": sx - sb,
                 "wins": w, "losses": l, "p_one_sided": sign_test(w, l),
                 "baseline_steps": sum(b[t]["steps"] for t in common) / n, "run_steps": sum(x[t]["steps"] for t in common) / n}
            out["paired"].append(p)
            print(f"  {p['run']:28s} n={n:3d}  {sb:5.1f} -> {sx:5.1f}  ({p['delta']:+5.1f})  wins/losses {w}/{l}  "
                  f"p={p['p_one_sided']:.3f}  steps {p['baseline_steps']:.2f} -> {p['run_steps']:.2f}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
