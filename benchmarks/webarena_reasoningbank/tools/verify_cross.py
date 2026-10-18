"""Cross-check verifier: does the 32B agent's outcome agree with an independent 397B agent run on the same task?
No GT is used to make the decision; GT is only used to score the verifier.
Usage: python verify_cross.py --ref results/shop_nomem_397 --out results/cross_<name>.jsonl <run_dirs...>"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("WA_CODE", os.path.join(os.path.dirname(TOOLS_DIR), "work", "reasoning-bank", "WebArena")))
from utils.local_llm import chat, GT_JUDGE_URL, GT_JUDGE_MODEL  # noqa: E402

PROMPT = """Two independent web agents performed the same task on the same shopping website.

Task: {intent}

Agent A final answer: {a}
Agent A final page URL: {ua}

Agent B final answer: {b}
Agent B final page URL: {ub}

Do the two agents reach the same outcome? For information questions compare the substance of the answers (same numbers,
same items, same conclusion; wording may differ). For navigation tasks (the user wants to be shown a page) compare the final
pages: same product/category/filter. Answer with exactly one word: SAME or DIFFERENT."""


def outcome(row):
    return (row.get("final_msg") or "(no answer)")[:3000], urlparse(row.get("last_url") or "").path + (
        "?" + urlparse(row.get("last_url") or "").query if urlparse(row.get("last_url") or "").query else "")


def audit(run_dir):
    import subprocess
    out = subprocess.run([sys.executable, os.path.join(TOOLS_DIR, "audit_answers.py"), run_dir],
                         capture_output=True, text=True, check=True).stdout
    return {r["tid"]: r for r in map(json.loads, out.splitlines())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("run_dirs", nargs="+")
    a = ap.parse_args()
    ref = audit(a.ref)
    jobs = []
    for rd in a.run_dirs:
        for tid, r in audit(rd).items():
            if tid in ref:
                jobs.append((rd, r, ref[tid]))

    def one(job):
        rd, r, q = job
        (ans_a, url_a), (ans_b, url_b) = outcome(r), outcome(q)
        msg = PROMPT.format(intent=r["intent"], a=ans_a, ua=url_a, b=ans_b, ub=url_b)
        txt, _ = chat([{"role": "user", "content": msg}], base_url=GT_JUDGE_URL, model=GT_JUDGE_MODEL,
                      temperature=0.0, enable_thinking=False)
        same = "SAME" in txt.upper() and "DIFFERENT" not in txt.upper()
        return {"run": os.path.basename(rd.rstrip("/")), "tid": r["tid"], "gt": (r["reward"] or 0) > 0,
                "ref_gt": (q["reward"] or 0) > 0, "same": same}

    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(one, jobs))
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = len(rows)
    tp = sum(r["same"] and r["gt"] for r in rows); fp = sum(r["same"] and not r["gt"] for r in rows)
    fn = sum(not r["same"] and r["gt"] for r in rows); tn = n - tp - fp - fn
    print(f"cross-check (SAME => success): n={n} acc={100*(tp+tn)/n:.1f} TP={tp} FP={fp} FN={fn} TN={tn} "
          f"precision={100*tp/max(tp+fp,1):.1f} recall={100*tp/max(tp+fn,1):.1f}")
