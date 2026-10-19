"""Runtime FT ingest (POST /ft/ingest → FinetuningStore.add_rows) — run with `python`.

1. A server started with no corpus: admission may already have walked the epoch
   counter forward; the first ingest restarts it at 0 and the rows are selectable.
2. Chat rows get loss spans; plain-text rows don't; target-less / too-long rows skip.
3. Ingest after training has started appends to the CURRENT epoch's pool
   without resetting the epoch or the trained marks.

    python tests/test_ft_ingest.py [--tokenizer PATH]
"""

import argparse
import sys

from transformers import AutoTokenizer

from vllm.deltaserve.finetuning_store import FinetuningStore

PASS = FAIL = 0


def cond(name, ok, detail=""):
    global PASS, FAIL
    PASS += bool(ok)
    FAIL += not ok
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


def chat(q, a):
    return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}


def main(tok):
    st = FinetuningStore(None, tokenize=lambda t: tok(t)["input_ids"], total_epochs=2,
                         max_saved_finetuning_tokens=64, chat_loss_on="ace", tokenizer=tok)
    st.load()
    while st.advance_epoch():      # what an admission loop on an empty store does
        pass
    cond("empty store walked the epochs", st.current_epoch == 2)
    r = st.add_rows([chat("2+2?", "4"), chat("3+3?", "6"), {"text": "plain text row"},
                     {"messages": [{"role": "user", "content": "no answer"}]},
                     chat("long " * 80, "x")])
    cond("3 added, 1 without target, 1 too long",
         r["added"] == 3 and r["skipped_no_target"] == 1 and r["skipped_too_long"] == 1, str(r))
    cond("epoch restarted at 0 (nothing trained yet)", st.current_epoch == 0)
    cond("ingested rows are selectable", st.has_next() and st.pop_next() is not None)
    cond("chat rows carry spans, text rows don't",
         st.samples[0].loss_spans and st.samples[2].loss_spans is None)

    # train two samples, then ingest more: same epoch, marks kept
    a = st.pop_next()
    b = st.pop_next(exclude=[a])
    st.claim([a, b])
    st.commit_claimed([a, b])
    ep = st.current_epoch
    st.add_rows([chat("5+5?", "10")])
    cond("post-training ingest keeps the epoch", st.current_epoch == ep)
    cond("trained marks kept", st.trained[st.id2idx[a.request_id]] and st.trained[st.id2idx[b.request_id]])
    left = []
    while (s := st.pop_next(exclude=left)) is not None:
        left.append(s)
    cond("new row joins the current pool (2 untrained left)", len(left) == 2, str(len(left)))
    print(f"\n{PASS} passed, {FAIL} failed")
    return FAIL == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="/mnt/weka/home/jianshu.she/scratch/Qwen3-32B")
    a = ap.parse_args()
    sys.exit(0 if main(AutoTokenizer.from_pretrained(a.tokenizer)) else 1)
