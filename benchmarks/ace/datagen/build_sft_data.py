"""Build LoRA SFT data from ACE traces on TRAIN splits only (context distillation: playbook -> weights).

Finance (FiNER, Formula): every first-attempt generator call on a training sample (call id
`train_e_1_s_<k>_gen_initial`), kept only if its extracted answer is correct. Input = the ACE generator prompt
re-rendered with an EMPTY playbook and EMPTY reflection (exactly the base-LLM prompt); target = the response that
was generated WITH the learned playbook. Post-reflection / post-curation calls are excluded: in +GT runs they saw
the ground truth.

AppWorld: the last generator call of each training task whose final state passes AppWorld's evaluator
(evaluations/train.json). Its message list + output is the whole trajectory. The playbook block between
"### PLAYBOOK BEGIN/END" is replaced by the shipped initial playbook (what the ReAct baseline sees). Trajectories
that contain the +GT run's ground-truth retry turns are dropped.

Never uses test/val traces (online runs, evaluation runs, DC) -> no leakage into the evaluation splits.

Usage: $VLLM_PY datagen/build_sft_data.py $ACE_SFT_DIR/data_v1   (needs transformers for token stats; paths from env.sh)
NOTE: FIN_RUNS lists the source project's resume segments (finer *_resume450 / *_resume300, i.e. the state on
2026-10-13 ~11:20 when data_v1 was built). Adjust it to the run dirs you actually have.
"""
import glob
import json
import os
import re
import sys

_ROOT = os.environ.get("ACE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WORK = os.environ.get("ACE_WORK_DIR", f"{_ROOT}/work")
ACE = os.environ.get("ACE_UPSTREAM_DIR", f"{_WORK}/ace")
AW = os.environ.get("APPWORLD_DIR", f"{ACE}/ace-appworld")
RESULTS = os.environ.get("ACE_RESULTS_DIR", f"{_WORK}/results")
MODEL_DIR = os.environ.get("MODEL_DIR", f"{_WORK}/models/Qwen3-32B-yarn131k")
sys.path.insert(0, ACE)
from ace.prompts.generator import GENERATOR_PROMPT  # noqa: E402
from eval.finance.data_processor import DataProcessor  # noqa: E402
from utils import extract_answer  # noqa: E402

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
CALL = re.compile(r"generator_train_e_1_s_(\d+)_gen_initial_")

# (run dir glob, offset of step 1 into the train file)
FIN_RUNS = {
    "formula": [("formula_offline_gt", 0), ("formula_offline_nogt", 0)],
    "finer": [("finer_offline_gt", 0), ("finer_offline_gt_resume450", 450),
              ("finer_offline_nogt", 0), ("finer_offline_nogt_resume300", 300)],
}
TRAIN = {"formula": "eval/finance/data/formula_train_subset_500.jsonl",
         "finer": "eval/finance/data/finer_train_batched_1000_samples.jsonl"}


def finance(task):
    proc = DataProcessor(task)
    train = proc.process_task_data([json.loads(l) for l in open(f"{ACE}/{TRAIN[task]}")])
    rows, stats = [], {}
    for run, offset in FIN_RUNS[task]:
        kept = seen = 0
        for f in glob.glob(f"{RESULTS}/finance/{run}/*/detailed_llm_logs/generator_train_e_1_s_*_gen_initial_*.json"):
            m = CALL.search(os.path.basename(f))
            idx = offset + int(m.group(1)) - 1
            # resumed segments start mid-file; the pre-loss segment of a run stops where its snapshot was taken
            if run in ("finer_offline_gt",) and idx >= 450 or run in ("finer_offline_nogt",) and idx >= 300:
                continue
            seen += 1
            call = json.load(open(f))
            s = train[idx]
            if not proc.answer_is_correct(extract_answer(call["response"]), s["target"]):
                continue
            kept += 1
            prompt = GENERATOR_PROMPT.format("(empty)", "(empty)", s["question"], s["context"])
            rows.append({"messages": [{"role": "user", "content": prompt},
                                      {"role": "assistant", "content": call["response"]}],
                         "meta": {"task": task, "run": run, "train_index": idx}})
        stats[run] = {"first_attempts": seen, "correct_kept": kept}
    return rows, stats


def appworld():
    initial = open(f"{AW}/experiments/playbooks/appworld_initial_playbook.txt").read()
    rows, stats = [], {}
    for exp in ["qwen_off_gt_adapt", "qwen_off_nogt_adapt"]:
        ind = json.load(open(f"{AW}/experiments/outputs/{exp}/evaluations/train.json"))["individual"]
        kept = dropped_gt = 0
        for task_id, v in ind.items():
            if not v["success"]:
                continue
            calls = [json.loads(l) for l in open(f"{AW}/experiments/outputs/{exp}/tasks/{task_id}/logs/lm_calls.jsonl")]
            gen = [c for c in calls if c["input"]["messages"][0]["content"].startswith("I am your supervisor")]
            last = gen[-1]
            msgs = [dict(role=m["role"], content=m["content"]) for m in last["input"]["messages"]]
            blob = json.dumps(msgs)
            if "failed to match the ground truth" in blob or "Use the reasoning above, along with the playbook" in blob:
                dropped_gt += 1
                continue
            msgs.append({"role": "assistant", "content": last["output"]["choices"][0]["message"]["content"]})
            msgs[0]["content"] = re.sub(r"### PLAYBOOK BEGIN\n.*?\n### PLAYBOOK END",
                                        lambda _: f"### PLAYBOOK BEGIN\n{initial}\n### PLAYBOOK END",
                                        msgs[0]["content"], count=1, flags=re.S)
            kept += 1
            rows.append({"messages": msgs, "meta": {"task": "appworld", "run": exp, "task_id": task_id}})
        stats[exp] = {"successful": sum(v["success"] for v in ind.values()), "kept": kept, "dropped_gt_retry": dropped_gt}
    return rows, stats


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    summary = {}
    for name, fn in [("formula", lambda: finance("formula")), ("finer", lambda: finance("finer")), ("appworld", appworld)]:
        rows, stats = fn()
        lens = sorted(len(tok.encode(tok.apply_chat_template(r["messages"], tokenize=False))) for r in rows)
        with open(f"{out_dir}/{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summary[name] = {"examples": len(rows), "unique_items": len({json.dumps(r["meta"].get("train_index", r["meta"].get("task_id"))) for r in rows}),
                         "tokens_total": sum(lens), "len_p50": lens[len(lens) // 2] if lens else 0,
                         "len_max": lens[-1] if lens else 0, "per_run": stats}
    json.dump(summary, open(f"{out_dir}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
