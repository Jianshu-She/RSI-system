#!/usr/bin/env python3
"""Locate ACE finance offline snapshots across the original run and all resume segments (stdlib only).

  python3 eval/finance_snapshots.py latest <task> <gt|nogt>   -> "<global_step> <snapshot_path>"
  python3 eval/finance_snapshots.py best   <task> <gt|nogt>   -> "<global_step> <val_acc> <snapshot_path>"

Segments: $ACE_RESULTS_DIR/finance/<task>_offline_<lab>/<ts>/ (starts at 0) and .../<task>_offline_<lab>_resume<S>/<ts>/ (starts at S).
Snapshots are intermediate_playbooks/epoch_1_step_<k>_playbook.txt (global step S+k). Best = highest val accuracy on the
original eval schedule (global steps that are multiples of 100); ties -> earliest step.
"""
import glob
import json
import os
import re
import sys

_ROOT = os.environ.get("ACE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.environ.get("ACE_RESULTS_DIR", f"{os.environ.get('ACE_WORK_DIR', f'{_ROOT}/work')}/results")


def segments(task, lab):
    segs = []
    for d in glob.glob(f"{RESULTS}/finance/{task}_offline_{lab}/*/") + glob.glob(f"{RESULTS}/finance/{task}_offline_{lab}_resume*/*/"):
        m = re.search(r"_resume(\d+)/", d)
        segs.append((int(m.group(1)) if m else 0, d))
    return sorted(segs)


def snapshots(task, lab):
    out = {}
    for start, d in segments(task, lab):
        for f in glob.glob(f"{d}intermediate_playbooks/epoch_1_step_*_playbook.txt"):
            out[start + int(re.search(r"step_(\d+)_playbook", f).group(1))] = f
    return out


def vals(task, lab):
    out = {}
    for start, d in segments(task, lab):
        p = f"{d}val_results.json"
        if os.path.exists(p):
            for v in json.load(open(p)):
                g = start + v["step"]
                if g % 100 == 0:
                    out[g] = v["val_results"]["accuracy"]
    return out


if __name__ == "__main__":
    cmd, task, lab = sys.argv[1:4]
    snaps = snapshots(task, lab)
    if cmd == "latest":
        g = max(snaps)
        print(g, snaps[g])
    elif cmd == "best":
        v = vals(task, lab)
        g = max(v, key=lambda s: (v[s], -s))
        print(g, round(100 * v[g], 2), snaps[g])
