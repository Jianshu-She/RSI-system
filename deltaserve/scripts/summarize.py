"""Summarize a run_e2e.sh run directory: latency of phase A vs C, trace yield (B), training (bwd.csv), eval accuracy.

    python3 deltaserve/scripts/summarize.py <run_dir>        # prints a table, writes <run_dir>/summary.json
"""
import csv
import json
import os
import sys


def q(a, p):
    return a[min(len(a) - 1, int(p * len(a)))] if a else float("nan")


def traffic(path, slo=0.3):
    if not os.path.exists(path):
        return None
    r = [json.loads(l) for l in open(path)]
    t = sorted(x["ttft"] for x in r if x["ttft"] is not None)
    e = sorted(x["e2e"] for x in r)
    return {"n": len(r), "errors": sum(1 for x in r if x["error"]), "correct": sum(x["correct"] for x in r),
            "ttft_p50": q(t, .5), "ttft_p90": q(t, .9), "ttft_p99": q(t, .99), "ttft_max": t[-1] if t else None,
            "ttft_slo_frac": sum(x < slo for x in t) / max(1, len(t)),
            "e2e_p50": q(e, .5), "e2e_p90": q(e, .9), "e2e_p99": q(e, .99)}


def main():
    run = sys.argv[1]
    out = {"A_inference_only": traffic(f"{run}/A.jsonl"), "C_coserving": traffic(f"{run}/C.jsonl"),
           "B_traces": traffic(f"{run}/B.jsonl")}
    if os.path.exists(f"{run}/bwd.csv"):
        b = list(csv.DictReader(open(f"{run}/bwd.csv")))
        ep = {}
        for x in b:
            ep.setdefault(x["epoch"], []).append(float(x["batch_loss"]))
        out["training"] = {"cycles": len(b), "tokens": int(b[-1]["total_processed_tokens"]) if b else 0,
                           "start": b[0]["timestamp"] if b else None, "end": b[-1]["timestamp"] if b else None,
                           "epoch_mean_loss": {k: sum(v) / len(v) for k, v in ep.items()}}
    for tag in ("base", "rsi"):
        f = f"{run}/eval/eval_{tag}_formula.json"
        if os.path.exists(f):
            out[f"eval_{tag}"] = json.load(open(f))["results"]["accuracy"]
    json.dump(out, open(f"{run}/summary.json", "w"), indent=1)
    for k in ("A_inference_only", "C_coserving"):
        v = out[k]
        if v:
            print(f"{k:17s} n={v['n']:4d} TTFT p50/p90/p99 = {1e3 * v['ttft_p50']:.0f}/{1e3 * v['ttft_p90']:.0f}/"
                  f"{1e3 * v['ttft_p99']:.0f} ms  SLO<0.3s {100 * v['ttft_slo_frac']:.1f}%  E2E p50/p99 = "
                  f"{v['e2e_p50']:.2f}/{v['e2e_p99']:.2f} s  errors {v['errors']}")
    if out.get("B_traces"):
        print(f"B traces: {out['B_traces']['correct']}/{out['B_traces']['n']} correct")
    if out.get("training"):
        tr = out["training"]
        print(f"training: {tr['cycles']} cycles, {tr['tokens']} tokens, {tr['start']} -> {tr['end']}, "
              f"epoch mean loss {tr['epoch_mean_loss']}")
    for tag in ("base", "rsi"):
        if f"eval_{tag}" in out:
            print(f"test accuracy [{tag}]: {100 * out[f'eval_{tag}']:.1f}")


if __name__ == "__main__":
    main()
