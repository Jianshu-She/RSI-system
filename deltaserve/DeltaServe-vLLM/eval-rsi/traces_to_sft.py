"""Served ACE traces -> RSI-system SFT rows (their data_v2 recipe), and optionally ingest them.

Keeps the correct traces only (ACE's own answer_is_correct, recorded by rsi_traffic.py), re-renders the
prompt with an EMPTY playbook and reflection (context distillation: the playbook the traffic was served with
goes into the weights), and empties "bullet_ids" in the answer — exactly build_sft_data.py + make_data_v2.py.

  python3 eval-rsi/traces_to_sft.py traces.jsonl sft.jsonl [--ingest http://127.0.0.1:8000]
"""
import argparse
import json
import re
import sys
import urllib.request

PAT = re.compile(r'"bullet_ids"\s*:\s*\[[^\]]*\]')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces")
    ap.add_argument("out")
    ap.add_argument("--ace-dir", default="/mnt/weka/home/jianshu.she/RSI-system/benchmark/ACE/ace")
    ap.add_argument("--ingest", default="", help="server base URL: POST the rows to /ft/ingest")
    a = ap.parse_args()
    sys.path.insert(0, a.ace_dir)
    from ace.prompts.generator import GENERATOR_PROMPT

    seen, rows = set(), []
    n = 0
    for line in open(a.traces):
        t = json.loads(line)
        n += 1
        if not t["correct"] or t["index"] in seen:
            continue
        seen.add(t["index"])
        prompt = GENERATOR_PROMPT.format("(empty)", "(empty)", t["question"], t["context"])
        target = PAT.sub('"bullet_ids": []', t["response"], count=1)
        rows.append({"messages": [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": target}],
                     "meta": {"task": t["task"], "split": t["split"], "index": t["index"],
                              "source": "deltaserve-served-trace"}})
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{n} traces -> {len(rows)} SFT rows (correct, unique items) -> {a.out}")
    if a.ingest:
        req = urllib.request.Request(a.ingest.rstrip("/") + "/ft/ingest",
                                     data=json.dumps({"rows": rows}).encode(),
                                     headers={"Content-Type": "application/json"})
        print("ingest:", urllib.request.urlopen(req, timeout=600).read().decode())


if __name__ == "__main__":
    main()
