#!/usr/bin/env python3
"""Collect every LoRA training dataset + its adapter, training stats and test results into sft/lora_catalog/.

  python3 finetune/make_lora_catalog.py      (stdlib only; re-runnable; paths from env.sh)

The run directories below (lora_r1_2667873, lora_r2_fin_2668622, sft_baselines_2673885, base eval evalfin_2668076)
are the SLURM job ids of the source project; edit them to point at your own runs.

Output:
  $ACE_SFT_DIR/lora_catalog/datasets/<name>.jsonl   symlinks to the exact files each adapter was trained on
  $ACE_SFT_DIR/lora_catalog/MANIFEST.json           one record per dataset/adapter (machine-readable)
  $ACE_SFT_DIR/lora_catalog/README.md               the same as tables
Every number is read from files written by the training/eval jobs.
"""
import glob
import json
import os
from math import comb

_ROOT = os.environ.get("ACE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WORK = os.environ.get("ACE_WORK_DIR", f"{_ROOT}/work")
ROOT = _WORK  # paths in MANIFEST.json are written relative to the work dir
S = os.environ.get("ACE_SFT_DIR", f"{_WORK}/sft")
OUT = f"{S}/lora_catalog"
BASE = f"{S}/runs/lora_r1_2667873/evalfin_2668076"   # same-prompt base ("(empty)" context) on test
AW = os.environ.get("APPWORLD_DIR", f"{os.environ.get('ACE_UPSTREAM_DIR', f'{_WORK}/ace')}/ace-appworld") + "/experiments/outputs"

RECIPE = "LoRA r=16 alpha=32 dropout=0.05 on q,k,v,o,gate,up,down (134M params, 0.41%); lr 1e-4 cosine, 5% warmup; ~32k tokens/step; 2 epochs; 5% of rows held out (seed 42) for dev loss; loss on assistant tokens only; Qwen3-32B (YaRN x4 config), bf16"

# name -> (task, data file, adapter dir, eval files {tag: path}, source, target, note)
D = [
    ("formula_ace_v1", "formula", f"{S}/data_v1/formula.jsonl", f"{S}/runs/lora_r1_2667873/adapter_formula",
     {"lora_empty": f"{BASE}/eval_formula_lora_empty.json", "lora+ACE_playbook": f"{BASE}/eval_formula_lora_aceplaybook.json"},
     "ACE offline (+GT & noGT runs) first-attempt generator answers on TRAIN items, generated WITH the learned playbook; correct only",
     "full ACE JSON answer (reasoning, bullet_ids citing playbook bullets, final_answer)",
     "BUGGY: targets cite playbook bullet ids absent from the input -> adapter hallucinates endless id lists (superseded by v2)"),
    ("finer_ace_v1", "finer", f"{S}/data_v1/finer.jsonl", f"{S}/runs/lora_r1_2667873/adapter_finer",
     {"lora_empty": f"{BASE}/eval_finer_lora_empty.json"},
     "as formula_ace_v1 (FiNER, only items with all 4 tags correct)", "full ACE JSON answer incl. bullet_ids", "superseded by v2"),
    ("appworld_ace_v1", "appworld", f"{S}/data_v1/appworld.jsonl", f"{S}/runs/lora_r1_2667873/adapter_appworld",
     {"appworld": "qwen_lora_react_eval"},
     "ACE offline (+GT & noGT) adaptation trajectories on TRAIN tasks that pass the AppWorld evaluator; +GT retry turns dropped; "
     "learned playbook swapped for the shipped initial playbook", "multi-turn agent trajectory (loss on the model's own turns after the demo prefix)", ""),
    ("formula_ace_v2", "formula", f"{S}/data_v2/formula.jsonl", f"{S}/runs/lora_r2_fin_2668622/adapter_formula",
     {"lora_empty": f"{S}/runs/lora_r2_fin_2668622/eval_formula_lora_empty.json",
      "lora+ACE_playbook": f"{S}/runs/lora_r2_fin_2668622/eval_formula_lora_aceplaybook.json"},
     "= formula_ace_v1 with bullet_ids emptied", "ACE JSON answer (reasoning + final_answer, bullet_ids [])", "main ACE-distilled adapter"),
    ("finer_ace_v2", "finer", f"{S}/data_v2/finer.jsonl", f"{S}/runs/lora_r2_fin_2668622/adapter_finer",
     {"lora_empty": f"{S}/runs/lora_r2_fin_2668622/eval_finer_lora_empty.json"},
     "= finer_ace_v1 with bullet_ids emptied", "ACE JSON answer (bullet_ids [])", "main ACE-distilled adapter"),
]
SB = f"{S}/runs/sft_baselines_2673885"
for task, n_all in (("formula", 500), ("finer", 1000)):
    for kind, src, tgt in (("gold", "gold labels of TRAIN items", "answer only: {reasoning: '', bullet_ids: [], final_answer: gold}"),
                           ("self", "base model (no playbook, temp 0) answers on TRAIN items, correct only (rejection sampling)",
                            "base model's own JSON answer (bullet_ids [])")):
        for subset, sdesc in (("all", f"all {n_all} train items" if kind == "gold" else "all correctly answered train items"),
                              ("matched", "restricted to the train items of the ACE-distilled set")):
            n = f"{task}_{kind}_{subset}"
            D.append((n, task, f"{SB}/data/{n}.jsonl", f"{SB}/adapter_{n}", {"lora_empty": f"{SB}/eval_{n}.json"},
                      f"{src}; {sdesc}", tgt, "SFT baseline (also LoRA)"))


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return round(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n), 4) if n else 1.0


def wrong(p):
    d = json.load(open(p))
    e = d.get("errors") or d.get("error_log")
    return {x["index"] for x in e["errors"]}, round(100 * d["results"]["accuracy"], 1)


def data_stats(p):
    n, items, chars = 0, set(), 0
    for line in open(p):
        r = json.loads(line)
        n += 1
        m = r.get("meta", {})
        items.add(m.get("train_index", m.get("task_id")))
        chars += sum(len(x["content"]) for x in r["messages"])
    return n, len(items), round(os.path.getsize(p) / 1e6, 2)


def train_stats(adapter, run_logs):
    out = {}
    name = os.path.basename(adapter).replace("adapter_", "")
    log = f"{run_logs}/train_{name}.log"
    if os.path.exists(log):
        for line in open(log):
            if line.startswith('{"train_ex"'):
                h = json.loads(line)
                out.update(train_rows=h["train_ex"], dev_rows=h["dev_ex"], train_tokens=h["train_tokens"],
                           target_tokens=h["target_tokens"], steps=h["total_steps"])
                break
    tl = f"{adapter}/train_log.jsonl"
    if os.path.exists(tl):
        rows = [json.loads(l) for l in open(tl)]
        dev = [r["dev_loss"] for r in rows if "dev_loss" in r]
        out.update(dev_loss_first=round(dev[0], 3) if dev else None, dev_loss_last=round(dev[-1], 3) if dev else None,
                   train_loss_last=round(rows[-1].get("train_loss", float("nan")), 3) if rows else None,
                   train_minutes=round(rows[-1].get("elapsed_s", 0) / 60, 1) if rows else None)
    return out


os.makedirs(f"{OUT}/datasets", exist_ok=True)
manifest = []
for name, task, data, adapter, evals, source, target, note in D:
    if not os.path.exists(data):
        continue
    link = f"{OUT}/datasets/{name}.jsonl"
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.relpath(data, f"{OUT}/datasets"), link)
    n, uniq, mb = data_stats(data)
    rec = {"name": name, "task": task, "file": os.path.relpath(data, ROOT), "examples": n, "unique_items": uniq, "size_mb": mb,
           "source": source, "target": target, "note": note, "adapter": os.path.relpath(adapter, ROOT),
           "adapter_exists": os.path.exists(f"{adapter}/adapter_model.safetensors"), "recipe": RECIPE,
           **train_stats(adapter, f"{os.path.dirname(adapter)}/logs"), "test": {}}
    for tag, p in evals.items():
        if tag == "appworld":
            agg = {}
            for split in ("test_normal", "test_challenge"):
                e = f"{AW}/{p}/evaluations/{split}.json"
                if os.path.exists(e):
                    a = json.load(open(e))["aggregate"]
                    agg[split] = [a["task_goal_completion"], a["scenario_goal_completion"]]
            if len(agg) == 2:
                agg["avg"] = round(sum(agg["test_normal"] + agg["test_challenge"]) / 4, 1)
            rec["test"]["appworld"] = agg
        elif os.path.exists(p):
            w, acc = wrong(p)
            bw, bacc = wrong(f"{BASE}/eval_{task}_base_empty.json")
            rec["test"][tag] = {"acc": acc, "base_acc": bacc, "fixes_vs_base": len(bw - w), "breaks_vs_base": len(w - bw),
                                "p_vs_base": sign_p(len(bw - w), len(w - bw)), "eval_file": os.path.relpath(p, ROOT)}
    manifest.append(rec)
json.dump(manifest, open(f"{OUT}/MANIFEST.json", "w"), indent=1)

L = ["# LoRA training data catalog (ACE reproduction, Qwen3-32B)", "",
     "Generated by `python3 finetune/make_lora_catalog.py` from the job outputs. All data from TRAIN splits only (leak audits in",
     "EXPERIMENTS.md). Every adapter uses the same recipe:", "", f"> {RECIPE}", "",
     "Inputs are always the ACE generator prompt with an EMPTY playbook (`(empty)`), i.e. the base prompt; chat template with",
     "thinking off. Test eval: vLLM 0.31 `--enable-lora`, temp 0, same `(empty)` prompt; base = same prompt without adapter",
     "(Formula 69.0, FiNER 61.8). p = two-sided sign test on per-item correctness vs base.", "",
     "| dataset | task | examples | unique items | train/dev rows | dev loss first→last | test acc (lora, empty ctx) | vs base (+/−, p) | note |",
     "|---|---|---|---|---|---|---|---|---|"]
for r in manifest:
    t = r["test"].get("lora_empty")
    if t:
        pv = "<1e-4" if t["p_vs_base"] < 1e-4 else t["p_vs_base"]
        acc, vb = f"{t['acc']}", f"+{t['fixes_vs_base']}/−{t['breaks_vs_base']}, p={pv}"
    elif "appworld" in r["test"]:
        a = r["test"]["appworld"]
        acc, vb = f"avg {a.get('avg')} (normal {a['test_normal'][0]}/{a['test_normal'][1]}, challenge {a['test_challenge'][0]}/{a['test_challenge'][1]})", "ReAct avg 27.2"
    else:
        acc, vb = "—", "—"
    L.append(f"| `{r['name']}` | {r['task']} | {r['examples']} | {r['unique_items']} | {r.get('train_rows', '—')}/{r.get('dev_rows', '—')} | "
             f"{r.get('dev_loss_first')}→{r.get('dev_loss_last')} | {acc} | {vb} | {r['note']} |")
L += ["", "## Dataset definitions", ""]
for r in manifest:
    L.append(f"- **`{r['name']}`** — source: {r['source']}. Target: {r['target']}. File: `{r['file']}`. Adapter: `{r['adapter']}`.")
extra = [r for r in manifest if "lora+ACE_playbook" in r["test"]]
if extra:
    L += ["", "## LoRA + ACE playbook in context (Formula)", ""]
    for r in extra:
        t = r["test"]["lora+ACE_playbook"]
        L.append(f"- `{r['name']}` adapter + ACE offline +GT best playbook: {t['acc']} (ACE playbook alone: 86.5)")
open(f"{OUT}/README.md", "w").write("\n".join(L) + "\n")
print(f"wrote {len(manifest)} entries to {OUT}")
