"""Data selection, step 1+2: build a scored candidate pool of SFT traces on the TRAIN split (Formula / FiNER).

Candidates (all on TRAIN items, never val/test):
  teacher : Qwen3-32B WITH the learned ACE offline+GT playbook in context, K samples @ temp 0.7   (skills -> traces)
  ace     : the ACE offline runs' first-attempt generator calls (data_v1 source), already on disk
Per item, a difficulty probe:
  base    : Qwen3-32B with an EMPTY playbook, K samples @ temp 0.7 -> base_pass = fraction correct
Scores for every correct candidate (SFT target = response with bullet_ids emptied; input = empty-playbook prompt):
  nll     : mean per-token NLL of the target under the BASE model given the empty-playbook prompt (vLLM prompt_logprobs)
  judge   : Qwen3.5-397B 1-10 score of the reasoning (correct, general, self-contained, concise)
  leak    : reasoning cites playbook bullet ids / "playbook" (would be unanswerable without the context)
Output: $DS_DIR/pool_<task>.jsonl (default benchmarks/ace/data/selected) (one row per candidate, correct or not) — phases are cached, rerun resumes.

Usage (source env.sh; Qwen3-32B replicas from scripts/serve_qwen.sh, judge from webarena_reasoningbank/scripts/serve_aux.sh):
  GEN_URLS=http://127.0.0.1:9400/v1,... JUDGE_URL=http://127.0.0.1:8002/v1 $ACE_PY datagen/ds_pool.py formula
Full job: finetune/job_ds_pool.sbatch
"""
import glob, json, os, random, re, sys, threading
from concurrent.futures import ThreadPoolExecutor

_ROOT = os.environ.get("ACE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WORK = os.environ.get("ACE_WORK_DIR", f"{_ROOT}/work")
ACE = os.environ.get("ACE_UPSTREAM_DIR", f"{_WORK}/ace")
RESULTS = os.environ.get("ACE_RESULTS_DIR", f"{_WORK}/results")
DS_DIR = os.environ.get("DS_DIR", f"{_ROOT}/data/selected")   # pool_<task>.jsonl(.gz) + selected sets
os.chdir(ACE); sys.path.insert(0, ACE)
from openai import OpenAI  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from ace.prompts.generator import GENERATOR_PROMPT  # noqa: E402
from eval.finance.data_processor import DataProcessor  # noqa: E402
from utils import extract_answer  # noqa: E402

TASK = sys.argv[1]
K = int(os.environ.get("K", 4))
OUT = os.environ.get("POOL_OUT", f"{DS_DIR}/pool_{TASK}.jsonl")
TRAIN = {"formula": "eval/finance/data/formula_train_subset_500.jsonl",
         "finer": "eval/finance/data/finer_train_batched_1000_samples.jsonl"}[TASK]
PLAYBOOK = {"formula": glob.glob(f"{RESULTS}/finance/formula_offline_gt/*/best_playbook.txt"),
            "finer": glob.glob(f"{RESULTS}/finance/finer_offline_gt/ace_run_*/intermediate_playbooks/epoch_1_step_300_playbook.txt")}[TASK][0]
GEN_URLS = os.environ["GEN_URLS"].split(",")
JUDGE_URL, JUDGE_MODEL = os.environ.get("JUDGE_URL"), os.environ.get("JUDGE_MODEL", "qwen3.5-397b")
MODEL_DIR = os.environ.get("MODEL_DIR", f"{_WORK}/models/Qwen3-32B-yarn131k")
BULLET = re.compile(r'"bullet_ids"\s*:\s*\[[^\]]*\]')
LEAK = re.compile(r"\b[a-z]{2,8}-\d{5}\b|playbook", re.I)

proc = DataProcessor(TASK)
train = proc.process_task_data([json.loads(l) for l in open(TRAIN)])
playbook = open(PLAYBOOK).read()
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
clients = [OpenAI(base_url=u, api_key="x", timeout=3600) for u in GEN_URLS]
rr = [0]; lock = threading.Lock()


def client():
    with lock:
        rr[0] += 1
        return clients[rr[0] % len(clients)]


def prompt(item, pb):
    return GENERATOR_PROMPT.format(pb, "(empty)", item["question"], item["context"])


def sample(idx, source):
    pb = playbook if source == "teacher" else "(empty)"
    r = client().chat.completions.create(
        model="qwen3-32b", messages=[{"role": "user", "content": prompt(train[idx], pb)}], n=K, temperature=0.7,
        max_tokens=4096, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    out = []
    for c in r.choices:
        resp = c.message.content or ""
        out.append({"task": TASK, "idx": idx, "source": source, "response": resp,
                    "correct": bool(proc.answer_is_correct(extract_answer(resp), train[idx]["target"]))})
    return out


def ace_traces():
    """first-attempt generator calls of the ACE offline runs on train items (as sft/build_sft_data.py)."""
    runs = {"formula": [("formula_offline_gt", 0), ("formula_offline_nogt", 0)],
            "finer": [("finer_offline_gt", 0), ("finer_offline_gt_resume450", 450),
                      ("finer_offline_nogt", 0), ("finer_offline_nogt_resume300", 300)]}[TASK]
    call = re.compile(r"generator_train_e_1_s_(\d+)_gen_initial_")
    rows = []
    for run, off in runs:
        for f in glob.glob(f"{RESULTS}/finance/{run}/*/detailed_llm_logs/generator_train_e_1_s_*_gen_initial_*.json"):
            idx = off + int(call.search(os.path.basename(f)).group(1)) - 1
            if (run == "finer_offline_gt" and idx >= 450) or (run == "finer_offline_nogt" and idx >= 300):
                continue
            resp = json.load(open(f))["response"]
            rows.append({"task": TASK, "idx": idx, "source": f"ace:{run}", "response": resp,
                         "correct": bool(proc.answer_is_correct(extract_answer(resp), train[idx]["target"]))})
    return rows


def reasoning_of(resp):
    try:
        return str(json.loads(resp).get("reasoning", ""))
    except Exception:
        return resp


def target(resp):
    return BULLET.sub('"bullet_ids": []', resp, count=1)


def nll(row):
    """mean NLL of the SFT target under the base model, given the empty-playbook prompt (thinking off)."""
    msgs = [{"role": "user", "content": prompt(train[row["idx"]], "(empty)")}]
    p_ids = tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False),
                       add_special_tokens=False)
    t_ids = tok.encode(target(row["response"]) + "<|im_end|>", add_special_tokens=False)
    r = client().completions.create(model="qwen3-32b", prompt=p_ids + t_ids, max_tokens=1, temperature=0,
                                    extra_body={"prompt_logprobs": 0})
    lps = r.choices[0].prompt_logprobs[len(p_ids):]
    vals = [list(d.values())[0]["logprob"] if isinstance(list(d.values())[0], dict) else list(d.values())[0].logprob
            for d in lps if d]
    return -sum(vals) / max(len(vals), 1)


JUDGE_PROMPT = """You grade a worked solution that will be used as a training example for a smaller model.
Question and context:
{q}

Solution (JSON with reasoning and final_answer; the final answer is known to be correct):
{s}

Score 1-10 how good this solution is as a TEACHING example: the reasoning is correct and complete, every step follows
from the question/context alone (no references to external notes, playbooks or ids), it uses general reusable methods,
and it is concise. Reply with only the integer."""


def judge(row):
    it = train[row["idx"]]
    q = f"{it['question']}\n\n{it['context']}"[:12000]
    r = OpenAI(base_url=JUDGE_URL, api_key="x", timeout=600).chat.completions.create(
        model=JUDGE_MODEL, messages=[{"role": "user", "content": JUDGE_PROMPT.format(q=q, s=target(row["response"])[:6000])}],
        temperature=0, max_tokens=8, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    m = re.search(r"\d+", r.choices[0].message.content or "")
    return int(m.group(0)) if m else None


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(OUT)] if os.path.exists(OUT) else []
    done = {(r["idx"], r["source"]) for r in rows}
    n_items = min(len(train), int(os.environ.get("LIMIT", len(train))))   # LIMIT: smoke tests
    todo = [(i, s) for i in range(n_items) for s in ("teacher", "base") if (i, s) not in done]
    print(f"[{TASK}] {len(train)} train items, {len(todo)} (item, source) sampling jobs left", flush=True)
    if not any(r["source"].startswith("ace:") for r in rows):
        rows += [r for r in ace_traces() if r["idx"] < n_items]
    with ThreadPoolExecutor(64 * len(clients)) as ex:
        for out in ex.map(lambda j: sample(*j), todo):
            rows += out
    with open(OUT, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    base_pass = {}
    for r in rows:
        if r["source"] == "base": base_pass.setdefault(r["idx"], []).append(r["correct"])
    cands = [r for r in rows if r["correct"] and r["source"] != "base"]
    for r in cands:
        r["base_pass"] = sum(base_pass.get(r["idx"], [0])) / max(len(base_pass.get(r["idx"], [0])), 1)
        r["leak"] = bool(LEAK.search(reasoning_of(r["response"])))
    need = [r for r in cands if "nll" not in r]
    with ThreadPoolExecutor(32 * len(clients)) as ex:
        for r, v in zip(need, ex.map(nll, need)): r["nll"] = v
    print(f"[{TASK}] nll done for {len(need)}", flush=True)
    if JUDGE_URL:
        need = [r for r in cands if "judge" not in r]
        with ThreadPoolExecutor(128) as ex:
            for r, v in zip(need, ex.map(judge, need)): r["judge"] = v
        print(f"[{TASK}] judge done for {len(need)}", flush=True)
    with open(OUT, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    stat = {"items": len(train), "candidates_correct": len(cands),
            "items_with_candidate": len({r["idx"] for r in cands}),
            "by_source": {s: sum(r["source"] == s for r in cands) for s in sorted({r["source"] for r in cands})},
            "base_pass_mean": sum(sum(v) / len(v) for v in base_pass.values()) / max(len(base_pass), 1),
            "leak_rate": sum(r["leak"] for r in cands) / max(len(cands), 1)}
    json.dump(stat, open(OUT.replace(".jsonl", ".summary.json"), "w"), indent=1)
    print(json.dumps(stat, indent=1))
