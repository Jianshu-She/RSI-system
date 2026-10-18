"""Build SFT-baseline datasets (finance) for comparison with the ACE-distilled LoRA (sft/data_v2).

Same input format as data_v2 (ACE generator prompt with empty playbook/reflection = the base prompt); TRAIN split only.
  gold : target = {"reasoning": "", "bullet_ids": [], "final_answer": <gold>}            (plain label SFT, answer-only)
  self : target = the BASE model's own response (no playbook, temp 0) on that train item, kept only if correct
         (rejection sampling / STaR-style self-distillation); bullet_ids emptied as in data_v2
Each in two item sets:
  all     : every train item (gold) / every correctly answered train item (self)
  matched : restricted to the train items present in the ACE-distilled data_v2 (same items as the ACE LoRA)

Usage (ace/.venv python, cwd = $ACE_UPSTREAM_DIR):
  $ACE_PY $ACE_ROOT/datagen/build_sft_baselines.py <out_dir> <self_raw_dir_formula> <self_raw_dir_finer>
  (raw dirs = EVAL_LOG_DIR of `eval_split.py <task> train <empty_context> ...` run with the base model)
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, ".")
from ace.prompts.generator import GENERATOR_PROMPT  # noqa: E402
from eval.finance.data_processor import DataProcessor  # noqa: E402
from utils import extract_answer  # noqa: E402

_ROOT = os.environ.get("ACE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SFT = os.environ.get("ACE_SFT_DIR", f"{os.environ.get('ACE_WORK_DIR', f'{_ROOT}/work')}/sft")
out_dir, raw = sys.argv[1], {"formula": sys.argv[2], "finer": sys.argv[3]}
os.makedirs(out_dir, exist_ok=True)
cfg = json.load(open("./eval/finance/data/sample_config.json"))
summary = {}
for task in ("formula", "finer"):
    proc = DataProcessor(task)
    train = proc.process_task_data([json.loads(l) for l in open(cfg[task]["train_data"])])
    matched = {json.loads(l)["meta"]["train_index"] for l in open(f"{SFT}/data_v2/{task}.jsonl")}
    prompt = lambda s: GENERATOR_PROMPT.format("(empty)", "(empty)", s["question"], s["context"])
    row = lambda i, tgt, kind: {"messages": [{"role": "user", "content": prompt(train[i])},
                                             {"role": "assistant", "content": tgt}],
                                "meta": {"task": task, "train_index": i, "kind": kind}}
    gold = {i: row(i, json.dumps({"reasoning": "", "bullet_ids": [], "final_answer": s["target"]}, indent=2), "gold")
            for i, s in enumerate(train)}
    selfd = {}
    for f in glob.glob(f"{raw[task]}/generator_test_eval_*.json"):
        i = int(re.search(r"generator_test_eval_(\d+)_", os.path.basename(f)).group(1))
        resp = json.load(open(f)).get("response", "")
        if resp and proc.answer_is_correct(extract_answer(resp), train[i]["target"]):
            resp = re.sub(r'"bullet_ids"\s*:\s*\[[^\]]*\]', '"bullet_ids": []', resp, count=1)
            selfd[i] = row(i, resp, "self")
    for name, d in (("gold", gold), ("self", selfd)):
        for subset, keep in (("all", lambda i: True), ("matched", lambda i: i in matched)):
            rows = [d[i] for i in sorted(d) if keep(i)]
            with open(f"{out_dir}/{task}_{name}_{subset}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            summary[f"{task}_{name}_{subset}"] = len(rows)
    summary[f"{task}_base_train_correct"] = f"{len(selfd)}/{len(train)}"
    summary[f"{task}_ace_items"] = len(matched)
json.dump(summary, open(f"{out_dir}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
