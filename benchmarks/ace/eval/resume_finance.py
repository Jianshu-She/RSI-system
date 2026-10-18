"""Resume an interrupted ACE finance run from a saved playbook snapshot (node loss on 2026-10-13).

ACE processes samples in file order (no shuffling), and the playbook is its only state, so continuing from a
snapshot taken after N samples with samples[N:] reproduces the uninterrupted run's trajectory.

- offline: train on train[N:] starting from the step-N playbook; val/test sets unchanged.
- online:  stream test[N:] starting from the playbook after the last completed window (N = windows_done * 15).
Everything else is eval/finance/run.py's configuration (defaults), via ACE.run.

Usage (ace/.venv python, cwd = $ACE_UPSTREAM_DIR):
  LOCAL_LLM_URL=... $ACE_PY $ACE_ROOT/eval/resume_finance.py <task> <offline|online> <gt|nogt> <N> <playbook> <save_path> [eval_steps]
"""
import json
import sys

sys.path.insert(0, ".")
from ace import ACE
from eval.finance.data_processor import DataProcessor
from playbook_utils import get_next_global_id

task, mode, label_mode, start, playbook_path, save_path = sys.argv[1:7]
eval_steps = int(sys.argv[7]) if len(sys.argv) > 7 else 100
start = int(start)
cfg = json.load(open("./eval/finance/data/sample_config.json"))[task]
proc = DataProcessor(task)
load = lambda p: proc.process_task_data([json.loads(l) for l in open(p)])
test = load(cfg["test_data"])
if mode == "offline":
    train, val = load(cfg["train_data"])[start:], load(cfg["val_data"])
else:
    train, val, test = None, None, test[start:]

ace = ACE(api_provider="local", generator_model="qwen3-32b", reflector_model="qwen3-32b", curator_model="qwen3-32b",
          max_tokens=4096, initial_playbook=open(playbook_path).read(), use_bulletpoint_analyzer=False,
          bulletpoint_analyzer_threshold=0.90)
# ACE.__init__ resets next_global_id to 1 even with an initial playbook; continue the snapshot's ID sequence instead
ace.next_global_id = get_next_global_id(ace.playbook)
print("next_global_id =", ace.next_global_id, flush=True)
config = {
    "num_epochs": 1, "max_num_rounds": 3, "curator_frequency": 1, "eval_steps": eval_steps, "online_eval_frequency": 15,
    "save_steps": 50, "playbook_token_budget": 80000, "task_name": task, "mode": mode, "json_mode": False,
    "no_ground_truth": label_mode == "nogt", "save_dir": save_path, "test_workers": 20,
    "initial_playbook_path": playbook_path, "use_bulletpoint_analyzer": False, "bulletpoint_analyzer_threshold": 0.90,
    "api_provider": "local", "batch_size": 1, "curator_batch_size": None, "curator_num_groups": None,
    "augmented_shuffling": True, "resume_start_index": start,
}
print(f"RESUME {task} {mode} {label_mode} from sample {start} with playbook {playbook_path}", flush=True)
ace.run(mode=mode, train_samples=train, val_samples=val, test_samples=test, data_processor=proc, config=config)
