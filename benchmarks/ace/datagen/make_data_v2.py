#!/usr/bin/env python3
"""data_v1 -> data_v2: empty the "bullet_ids" list in every finance SFT target (stdlib only).

Why: data_v1 targets were generated WITH a learned playbook and cite its bullet ids, but the SFT input has no
playbook ("(empty)"). A LoRA trained on that learns to hallucinate endless id lists (Formula collapsed to 26.0 on
test). data_v2 = data_v1 with `"bullet_ids": []` in every assistant target; nothing else changes (same rows, same
order). Every row gets meta.v2 = "bullet_ids_emptied". AppWorld rows are not touched (no v2 for AppWorld).

Usage: python3 make_data_v2.py <data_v1_dir> <data_v2_dir>
"""
import json
import os
import re
import sys

src, dst = sys.argv[1:3]
os.makedirs(dst, exist_ok=True)
PAT = re.compile(r'"bullet_ids"\s*:\s*\[[^\]]*\]')
for task in ("formula", "finer"):
    n = changed = 0
    with open(f"{dst}/{task}.jsonl", "w") as out:
        for line in open(f"{src}/{task}.jsonl"):
            r = json.loads(line)
            for m in r["messages"]:
                if m["role"] == "assistant":
                    new = PAT.sub('"bullet_ids": []', m["content"], count=1)
                    changed += new != m["content"]
                    m["content"] = new
            r["meta"]["v2"] = "bullet_ids_emptied"
            out.write(json.dumps(r) + "\n")
            n += 1
    print(f"{task}: {n} rows, {changed} targets rewritten")
