"""Split the RSI FT data into K rounds for rsi_mix.sh (ACE formula traces + ACE finer, interleaved).

Round k = chunk k of each source, so every round trains both domains (what one RSI loop iteration would
collect from mixed traffic). Each round's size is a multiple of --world (torchtune's FSDP ranks take
consecutive samples, so a step never straddles two rounds); DeltaServe trains the same samples.
Run with the DeltaServe env python (renders rows with DeltaServe's own chat_sft.encode_chat, i.e. the
exact input_len / loss spans the server uses):

  python rsi_mix_prep.py --tokenizer $MODEL_DIR --out <dir> --rounds 4 formula_traces.jsonl finer.jsonl
  -> <dir>/r<k>.jsonl (chat rows: DeltaServe corpus / POST /ft/ingest), <dir>/r<k>.tok.jsonl
     ({"tokens", "labels"} with -100 off the loss spans: torchtune), <dir>/meta.json (tokens per round)
"""
import argparse
import json
import os

from transformers import AutoTokenizer

from vllm.deltaserve.chat_sft import encode_chat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("sources", nargs="+")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    os.makedirs(args.out, exist_ok=True)
    chunks = [[] for _ in range(args.rounds)]
    for src in args.sources:
        rows = []
        for line in open(src):
            row = json.loads(line)
            ids, spans = encode_chat(tok, row["messages"], "ace", "off")
            if spans and 2 <= len(ids) <= args.max_tokens:
                rows.append((row, ids, spans, os.path.basename(src)))
        per = len(rows) // args.rounds
        for k in range(args.rounds):
            chunks[k] += rows[k * per:(k + 1) * per]
    meta = {"rounds": []}
    for k, rs in enumerate(chunks):
        rs = rs[: len(rs) - len(rs) % args.world]
        with open(f"{args.out}/r{k}.jsonl", "w") as f, open(f"{args.out}/r{k}.tok.jsonl", "w") as g:
            for row, ids, spans, _ in rs:
                f.write(json.dumps({"messages": row["messages"]}) + "\n")
                lab = [-100] * len(ids)
                for a, b in spans:
                    lab[a:b] = ids[a:b]
                g.write(json.dumps({"tokens": ids, "labels": lab}) + "\n")
        by_src = {}
        for _, ids, _, s in rs:
            by_src[s] = by_src.get(s, 0) + len(ids)
        meta["rounds"].append({"samples": len(rs), "tokens": sum(len(r[1]) for r in rs), "by_source": by_src})
    json.dump(meta, open(f"{args.out}/meta.json", "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
