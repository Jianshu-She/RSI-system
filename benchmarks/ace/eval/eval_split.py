"""Score a fixed context (empty / ICL demos / any playbook file) on a finance split with ACE's own evaluator.

Uses ace.core.Generator + utils.evaluate_test_set exactly as ACE's eval_only/val passes do, so numbers are
comparable with ACE's validation accuracies.

Usage (ace/.venv python, cwd = $ACE_UPSTREAM_DIR):
  LOCAL_LLM_URL=http://127.0.0.1:8100/v1 $ACE_PY $ACE_ROOT/eval/eval_split.py <task> <train|val|test> <context_file|-> <out.json>
"""
import json
import os
import sys

sys.path.insert(0, ".")
from ace.core import Generator
from eval.finance.data_processor import DataProcessor
from utils import evaluate_test_set, initialize_clients

task, split, ctx_path, out = sys.argv[1:5]
cfg = json.load(open("./eval/finance/data/sample_config.json"))[task]
key = {"train": "train_data", "val": "val_data", "test": "test_data"}[split]
proc = DataProcessor(task)
samples = proc.process_task_data([json.loads(l) for l in open(cfg[key])])
playbook = "" if ctx_path == "-" else open(ctx_path).read()

gen_client, _, _ = initialize_clients("local")
gen = Generator(gen_client, "local", os.environ.get("EVAL_MODEL", "qwen3-32b"), max_tokens=4096)  # EVAL_MODEL: e.g. a vLLM LoRA name
log_dir = os.environ.get("EVAL_LOG_DIR")  # optional: save every raw generator call (prompt + response)
if log_dir:
    os.makedirs(log_dir, exist_ok=True)
results, errors = evaluate_test_set(proc, gen, playbook, samples, max_tokens=4096, log_dir=log_dir, max_workers=20)
json.dump({"task": task, "split": split, "context": ctx_path, "model": os.environ.get("EVAL_MODEL", "qwen3-32b"), "results": results, "errors": errors}, open(out, "w"), indent=1)
print(f"RESULT {task} {split} ctx={ctx_path}: accuracy={results['accuracy']:.4f} ({results.get('correct')}/{results.get('total')})")
