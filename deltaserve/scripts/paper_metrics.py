"""Aggregate a paper_run.sh run directory into paper_metrics.json + a markdown report (stdout).

  python3 deltaserve/scripts/paper_metrics.py <run_dir>

Sections: (1) accuracy grid {base, rsi} x {empty, skills} on formula / finer / gsm8k / mmlu with paired sign tests;
(2) serving cost of skills vs LoRA (stock vLLM, TP=4): prompt tokens, TTFT, TBT, E2E, KV-cache usage, prefix hits;
(3) DeltaServe co-serving: latency + GPU utilization (DCGM SM_ACTIVE) in phase A (inference only) vs C (training);
(4) prompt-extraction leakage.
"""
import json
import math
import os
import re
import sys
import time

KV_BYTES_PER_TOKEN = 2 * 64 * 8 * 128 * 2  # K+V x layers x kv heads x head dim x bf16 (Qwen3-32B), whole model
BENCHES = ("formula", "finer", "gsm8k", "mmlu")
CELLS = ("base_empty", "base_skills", "rsi_empty", "rsi_skills")
PAIRS = (("base_skills", "base_empty"), ("rsi_empty", "base_empty"), ("rsi_skills", "base_skills"),
         ("rsi_skills", "rsi_empty"), ("rsi_empty", "base_skills"))


def q(a, p):
    a = sorted(x for x in a if x is not None)
    return a[min(len(a) - 1, int(p * len(a)))] if a else None


def mean(a):
    a = [x for x in a if x is not None]
    return sum(a) / len(a) if a else None


def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def sign_test(a, b):
    w = sum(x and not y for x, y in zip(a, b))
    l = sum(y and not x for x, y in zip(a, b))
    n, k = w + l, min(w, l)
    return w, l, (1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))


def accuracy(run):
    out = {}
    for b in BENCHES:
        cells = {}
        for c in CELLS:
            f = f"{run}/eval/eval_{c}_{b}.json"
            if os.path.exists(f):
                d = json.load(open(f))
                tot = d["results"]["total"]
                wrong = {e["index"] for e in d["errors"]["errors"]}
                cells[c] = {"acc": d["results"]["accuracy"], "n": tot, "items": [i not in wrong for i in range(tot)]}
        tests = {}
        for a, r in PAIRS:
            if a in cells and r in cells and cells[a]["n"] == cells[r]["n"]:
                w, l, p = sign_test(cells[a]["items"], cells[r]["items"])
                tests[f"{a} vs {r}"] = {"win": w, "loss": l, "p": p, "delta": cells[a]["acc"] - cells[r]["acc"]}
        out[b] = {"acc": {c: v["acc"] for c, v in cells.items()}, "n": {c: v["n"] for c, v in cells.items()},
                  "tests": tests}
    return out


def window(metrics, t0, t1):
    return [m for m in metrics if t0 <= m["t"] <= t1]


def traffic_stats(recs, metrics=()):
    if not recs:
        return None
    t0 = min(r["t_send"] for r in recs)
    t1 = max(r["t_send"] + r["e2e"] for r in recs)
    gaps = [g for r in recs for g in (r.get("tbt_gaps") or [])]
    out = {"n": len(recs), "errors": sum(1 for r in recs if r["error"]), "correct": sum(r["correct"] for r in recs),
           "prompt_tokens_mean": mean([r.get("prompt_tokens") for r in recs]),
           "completion_tokens_mean": mean([r.get("completion_tokens") for r in recs]),
           "ttft_p50": q([r["ttft"] for r in recs], .5), "ttft_p90": q([r["ttft"] for r in recs], .9),
           "ttft_p99": q([r["ttft"] for r in recs], .99),
           "ttft_slo_0.3s": sum((r["ttft"] or 9e9) < 0.3 for r in recs) / len(recs),
           "tbt_p50": q(gaps, .5), "tbt_p90": q(gaps, .9), "tbt_p99": q(gaps, .99),
           "tbt_req_mean_p50": q([r.get("tbt_mean") for r in recs], .5),
           "e2e_p50": q([r["e2e"] for r in recs], .5), "e2e_p99": q([r["e2e"] for r in recs], .99),
           "wall_s": t1 - t0,
           "output_tok_per_s": sum(r.get("completion_tokens") or 0 for r in recs) / max(1e-9, t1 - t0)}
    w = window(metrics, t0, t1)
    if w:
        kv = [m.get("vllm:kv_cache_usage_perc") for m in w]
        run_ = [m.get("vllm:num_requests_running") for m in w]
        hq = w[-1].get("vllm:prefix_cache_queries_total", 0) - w[0].get("vllm:prefix_cache_queries_total", 0)
        hh = w[-1].get("vllm:prefix_cache_hits_total", 0) - w[0].get("vllm:prefix_cache_hits_total", 0)
        out.update({"kv_usage_mean": mean(kv), "kv_usage_peak": max(x for x in kv if x is not None),
                    "running_mean": mean(run_), "running_peak": max(x for x in run_ if x is not None),
                    "waiting_peak": max(m.get("vllm:num_requests_waiting", 0) for m in w),
                    "prefix_hit_rate": hh / hq if hq else None,
                    "preemptions": w[-1].get("vllm:num_preemptions_total", 0) - w[0].get("vllm:num_preemptions_total", 0)})
    return out


def kv_pool(log):
    if not os.path.exists(log):
        return {}
    s = open(log).read()
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", s)
    c = re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", s)
    return {"kv_pool_tokens": int(m.group(1).replace(",", "")) if m else None,
            "max_concurrency_at_max_len": float(c.group(2)) if c else None}


def serving_cost(run, sub="cost"):
    d = f"{run}/{sub}"
    if not os.path.isdir(d):
        return None
    mP, mL = jl(f"{d}/metrics_P.jsonl"), jl(f"{d}/metrics_L.jsonl")
    out = {"pool_P": kv_pool(f"{d}/logs/vllm_P.log"), "pool_L": kv_pool(f"{d}/logs/vllm_L.log"),
           "pool_M": kv_pool(f"{d}/logs/vllm_M.log"), "conditions": {}}
    for f in sorted(os.listdir(d)):
        m = re.match(r"traffic_(.+)_r([\d.]+)\.jsonl$", f)
        if m:
            cond, rate = m.groups()
            met = mL if cond.startswith("lora") else jl(f"{d}/metrics_M.jsonl") if cond.startswith("merged") else mP
            out["conditions"][f"{cond}@{rate}"] = traffic_stats(jl(f"{d}/{f}"), met)
    out["leak"] = {}
    for c in ("skills", "base", "lora"):
        f = f"{d}/leak_{c}.json"
        if os.path.exists(f):
            x = json.load(open(f))
            out["leak"][c] = {k: x[k] for k in ("n_bullets", "n_attacks", "id_leak", "verbatim_leak", "ids_leaked",
                                                "bullets_verbatim")}
    return out


def phases(driver):
    """phase name -> [t_start, t_end] from run_e2e.sh's driver.log ('=== YYYY-mm-dd HH:MM:SS <msg>')."""
    ev = []
    for ln in open(driver):
        m = re.search(r"=== (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*)", ln)
        if m:
            ev.append((time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")), m.group(2)))
    t = {}
    for ts, msg in ev:
        for key, pat in (("A", "A: "), ("B", "B: "), ("I", "I: "), ("C", "C: "), ("C_end", "final adapter"),
                         ("stop", "server stopped"), ("up", "server up")):
            if msg.startswith(pat) and key not in t:
                t[key] = ts
    return {"A": [t.get("A"), t.get("B")], "B": [t.get("B"), t.get("I")],
            "C": [t.get("C"), t.get("C_end") or t.get("stop")]}


def gpu_util(path, gpus, wins):
    rows = []
    for ln in open(path) if os.path.exists(path) else []:
        p = ln.split()
        if len(p) >= 9 and p[1] == "GPU" and int(p[2]) in gpus:
            f = [None if v == "N/A" else float(v) for v in p[3:9]]
            rows.append((float(p[0]), int(p[2]), *f))
    out = {}
    for name, (t0, t1) in wins.items():
        if t0 is None or t1 is None:
            continue
        w = [r for r in rows if t0 <= r[0] <= t1]
        if w:
            out[name] = {"seconds": t1 - t0, "power_w": mean([r[2] for r in w]), "gputl": mean([r[3] for r in w]),
                         "sm_active": mean([r[5] for r in w]), "sm_occupancy": mean([r[6] for r in w]),
                         "tensor_active": mean([r[7] for r in w])}
    return out


def coserving(run):
    import csv
    out = {"A_inference_only": traffic_stats(jl(f"{run}/A.jsonl")), "C_coserving": traffic_stats(jl(f"{run}/C.jsonl")),
           "B_traces": {"n": len(jl(f"{run}/B.jsonl")), "correct": sum(r["correct"] for r in jl(f"{run}/B.jsonl"))}}
    if os.path.exists(f"{run}/sft_from_traces.jsonl"):
        out["sft_rows"] = sum(1 for _ in open(f"{run}/sft_from_traces.jsonl"))
    if os.path.exists(f"{run}/bwd.csv"):
        b = list(csv.DictReader(open(f"{run}/bwd.csv")))
        out["training"] = {"cycles": len(b), "tokens": int(b[-1]["total_processed_tokens"]) if b else 0}
    wins = phases(f"{run}/driver.log") if os.path.exists(f"{run}/driver.log") else {}
    gpus = list(range(len(os.environ.get("DS_GPUS", "0,1,2,3").split(","))))
    out["gpu"] = gpu_util(f"{run}/gpu.txt", set(gpus), wins)
    if wins.get("C") and wins["C"][0] and wins["C"][1] and out.get("training"):
        out["training"]["seconds"] = wins["C"][1] - wins["C"][0]
        out["training"]["ft_tok_per_s"] = out["training"]["tokens"] / max(1, out["training"]["seconds"])
    for k in ("A_inference_only", "C_coserving"):
        v = out[k]
        if v and wins.get(k[0]) and None not in wins[k[0]]:
            v["output_tok_per_s_phase"] = sum(r.get("completion_tokens") or 0 for r in jl(f"{run}/{k[0]}.jsonl")) / \
                (wins[k[0]][1] - wins[k[0]][0])
    return out


def f(x, s=1, d=1, unit=""):
    return "–" if x is None else f"{x * s:.{d}f}{unit}"


def cost_table(c, title):
    L = ["", f"## 2. Serving cost: skills in context vs LoRA (stock vLLM 0.31, TP=4) — {title}", "",
         f"KV pool: {c['pool_P'].get('kv_pool_tokens')} tokens (plain), {c['pool_L'].get('kv_pool_tokens')} "
         f"(LoRA server); KV per token {KV_BYTES_PER_TOKEN / 1024:.0f} KiB", "",
         "| condition @ rps | prompt tok | out tok | KV/request (no sharing) | TTFT p50 / p99 (ms) | TBT p50 / p99 (ms) | "
         "E2E p50 / p99 (s) | KV usage mean / peak | running peak | waiting peak | prefix hit | out tok/s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(c["conditions"].items()):
        if not v:
            continue
        kvreq = (v["prompt_tokens_mean"] or 0) * KV_BYTES_PER_TOKEN / 2 ** 30
        L.append(f"| {k} | {f(v['prompt_tokens_mean'], d=0)} | {f(v['completion_tokens_mean'], d=0)} | {kvreq:.2f} GiB | "
                 f"{f(v['ttft_p50'], 1e3, 0)} / {f(v['ttft_p99'], 1e3, 0)} | {f(v['tbt_p50'], 1e3, 1)} / "
                 f"{f(v['tbt_p99'], 1e3, 1)} | {f(v['e2e_p50'], d=2)} / {f(v['e2e_p99'], d=2)} | "
                 f"{f(v.get('kv_usage_mean'), 100)} / {f(v.get('kv_usage_peak'), 100)} % | "
                 f"{f(v.get('running_peak'), d=0)} | {f(v.get('waiting_peak'), d=0)} | "
                 f"{f(v.get('prefix_hit_rate'), 100)} % | {f(v['output_tok_per_s'], d=0)} |")
    return L


def report(m):
    L = ["# Paper metrics", ""]
    L += ["## 1. Accuracy: {base, LoRA (rsi)} × {no skills, skills in context}", "",
          "| bench | n | base | base + skills | LoRA | LoRA + skills |", "|---|---|---|---|---|---|"]
    for b, v in m["accuracy"].items():
        if v["acc"]:
            n = next(iter(v["n"].values()))
            L.append(f"| {b} | {n} | " + " | ".join(f(v['acc'].get(c), 100) for c in CELLS) + " |")
    L += ["", "| bench | comparison | Δ (pts) | +/− | sign-test p |", "|---|---|---|---|---|"]
    for b, v in m["accuracy"].items():
        for k, t in v["tests"].items():
            L.append(f"| {b} | {k} | {t['delta'] * 100:+.1f} | +{t['win']}/−{t['loss']} | {t['p']:.4f} |")
    c = m.get("serving_cost")
    for key, title in (("serving_cost", "prefix cache on; the 4-rps rows repeat the 1-rps prompts (full-prompt cache hits)"),
                       ("serving_cost_fresh", "prefix cache on, 4 rps, formula TRAIN split: every prompt new to the server"),
                       ("serving_cost_nopc", "prefix cache OFF: skills not shared between requests (per-user skills)")):
        if m.get(key):
            L += cost_table(m[key], title)
    LK = []
    if c and c.get("leak"):
            LK += ["", "## 4. Prompt-extraction leakage (12 attacks, union over attacks)", "",
                  "| deployment | bullet ids leaked | bullets reproduced verbatim |", "|---|---|---|"]
            for k, v in c["leak"].items():
                LK.append(f"| {k} | {v['ids_leaked']}/{v['n_bullets']} ({100 * v['id_leak']:.1f} %) | "
                         f"{v['bullets_verbatim']}/{v['n_bullets']} ({100 * v['verbatim_leak']:.1f} %) |")
    s = m["coserving"]
    L += ["", "## 3. DeltaServe co-serving: same traffic, inference only (A) vs training in the gaps (C)", "",
          "| phase | n | TTFT p50 / p99 (ms) | TTFT < 0.3 s | TBT p50 / p99 (ms) | E2E p50 / p99 (s) | SM active | "
          "GPU util (nvidia-smi) | power (W/GPU) |", "|---|---|---|---|---|---|---|---|---|"]
    for k, ph in (("A_inference_only", "A"), ("C_coserving", "C")):
        v, g = s.get(k), s["gpu"].get(ph, {})
        if v:
            L.append(f"| {ph} | {v['n']} | {f(v['ttft_p50'], 1e3, 0)} / {f(v['ttft_p99'], 1e3, 0)} | "
                     f"{f(v['ttft_slo_0.3s'], 100)} % | {f(v['tbt_p50'], 1e3, 1)} / {f(v['tbt_p99'], 1e3, 1)} | "
                     f"{f(v['e2e_p50'], d=2)} / {f(v['e2e_p99'], d=2)} | {f(g.get('sm_active'), 100)} % | "
                     f"{f(g.get('gputl'), d=0)} % | {f(g.get('power_w'), d=0)} |")
    if s.get("training"):
        t = s["training"]
        L += ["", f"Training in C: {t['cycles']} cycles, {t['tokens']} tokens in {f(t.get('seconds'), d=0)} s "
              f"({f(t.get('ft_tok_per_s'), d=0)} FT tok/s) from {s.get('sft_rows')} self-generated rows "
              f"({s['B_traces']['correct']}/{s['B_traces']['n']} correct traces)."]
    return "\n".join(L + LK)


def main():
    run = sys.argv[1]
    m = {"accuracy": accuracy(run), "serving_cost": serving_cost(run), "coserving": coserving(run),
         "serving_cost_fresh": serving_cost(run, "cost_fresh"), "serving_cost_nopc": serving_cost(run, "cost_nopc")}
    json.dump(m, open(f"{run}/paper_metrics.json", "w"), indent=1)
    print(report(m))


if __name__ == "__main__":
    main()
