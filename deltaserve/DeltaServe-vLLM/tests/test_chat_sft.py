"""Chat SFT (RSI-system data) gates — run with `python`, not pytest.

1. ``deltaserve.chat_sft.encode_chat`` == the offline trainers' encode, token
   for token, on real RSI-system rows (the reference functions below are copied
   verbatim from rsi/finetune/train_lora.py and the ACE train_lora.py).
2. ``head_backward(loss_spans=...)`` == an autograd reference with the loss on
   the span tokens only (mean over target tokens); no spans == the old path.
3. Coordinator: spans follow the runner's write order (request ids), not the
   scheduler's order.
4. ``FinetuningStore`` loads a .jsonl corpus into ids + spans.

    python tests/test_chat_sft.py [--data-dir DIR] [--tokenizer PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from vllm.deltaserve.bwd_services.common.head import head_backward
from vllm.deltaserve.bwd_services.common.ops import rmsnorm
from vllm.deltaserve.chat_sft import encode_chat
from vllm.deltaserve.coordinator import FinetuneCoordinator
from vllm.deltaserve.finetuning_store import FinetuningStore

PASS = FAIL = 0


def cond(name, ok, detail=""):
    global PASS, FAIL
    PASS += bool(ok)
    FAIL += not ok
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


# ── reference encoders (verbatim from the offline trainers) ──────────────────
THINK_OFF = "<think>\n\n</think>\n\n"


def ref_rsi(tok, msgs, loss_on, thinking):          # rsi/finetune/train_lora.py
    TMPL = {} if thinking == "on" else {"enable_thinking": False}

    def render(m, gen=False):
        out = tok.apply_chat_template(m, tokenize=True, add_generation_prompt=gen, **TMPL)
        return list(out["input_ids"]) if hasattr(out, "keys") else list(out)

    if loss_on == "last":
        pre, full = render(msgs[:-1], gen=True), render(msgs)
        return (full, [(len(pre), len(full))]) if full[:len(pre)] == pre and len(full) > len(pre) else (full, [])
    ids, spans = [], []
    for m in msgs:
        if m["role"] == "assistant":
            ids += tok.encode("<|im_start|>assistant\n" + (THINK_OFF if thinking == "off" else ""), add_special_tokens=False)
            body = tok.encode(m["content"] + "<|im_end|>", add_special_tokens=False)
            spans.append((len(ids), len(ids) + len(body)))
            ids += body + tok.encode("\n", add_special_tokens=False)
        else:
            ids += tok.encode(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n", add_special_tokens=False)
    return ids, spans


def ref_ace(tok, messages):                          # benchmarks/ace/.../train_lora.py
    END = tok.convert_tokens_to_ids("<|im_end|>")
    start = max([i for i, m in enumerate(messages) if m["role"] == "user" and "solve the actual task" in m["content"]] or [0])
    first = next(i for i, m in enumerate(messages) if m["role"] == "assistant" and i > start)
    ids = tok.encode(tok.apply_chat_template(messages[:first], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False), add_special_tokens=False)
    labels = [-100] * len(ids)
    nl = tok.encode("\n", add_special_tokens=False)
    for i in range(first, len(messages)):
        m = messages[i]
        if m["role"] == "assistant":
            if i != first:
                p = tok.encode("<|im_start|>assistant\n" + THINK_OFF, add_special_tokens=False)
                ids, labels = ids + p, labels + [-100] * len(p)
            t = tok.encode(m["content"], add_special_tokens=False) + [END]
            ids, labels = ids + t + nl, labels + t + [-100] * len(nl)
        else:
            p = tok.encode(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n", add_special_tokens=False)
            ids, labels = ids + p, labels + [-100] * len(p)
    return ids, labels


def spans_to_labels(ids, spans):
    lab = [-100] * len(ids)
    for s, e in spans:
        lab[s:e] = ids[s:e]
    return lab


def encoder_parity(tok, data_dir, n_rows):
    print("encode_chat == offline trainers")
    for name in ("formula", "finer", "appworld"):
        path = os.path.join(data_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            print(f"  (skip {name}: {path} missing)")
            continue
        rows = [json.loads(ln)["messages"] for ln in open(path) if ln.strip()][:n_rows]
        modes = [("ace", "off"), ("all", "off")] + ([("last", "off")] if name != "appworld" else [])
        for loss_on, thinking in modes:
            bad = 0
            for msgs in rows:
                got_ids, got_sp = encode_chat(tok, msgs, loss_on, thinking)
                if loss_on == "ace":
                    rid, rlab = ref_ace(tok, msgs)
                    ok = got_ids == rid and spans_to_labels(got_ids, got_sp) == rlab
                else:
                    rid, rsp = ref_rsi(tok, msgs, loss_on, thinking)
                    ok = got_ids == rid and [tuple(x) for x in got_sp] == [tuple(x) for x in rsp]
                bad += not ok
            cond(f"{name} loss_on={loss_on}: {len(rows)} rows identical", bad == 0, f"{bad} mismatches")


def head_parity():
    print("head_backward with loss spans")
    torch.manual_seed(0)
    D, vocab = 32, 500
    seq_lens = [12, 7, 9]
    b_start = [0, 12, 19]
    n = sum(seq_lens)
    final_in = torch.randn(n, D, dtype=torch.float32)
    lm_w = torch.randn(vocab, D) * 0.1
    norm_w = torch.rand(D) + 0.5
    ids = torch.randint(0, vocab, (n,))
    spans = [[(4, 9), (10, 12)], None, [(0, 3)]]      # (0, 3): position 0 is skipped

    # reference: autograd CE over the selected predicting rows, mean over them
    x = final_in.clone().requires_grad_(True)
    h = rmsnorm(x, norm_w, 1e-6).float()
    rows, tg = [], []
    for k, (st, ln) in enumerate(zip(b_start, seq_lens)):
        sel = range(1, ln) if spans[k] is None else [j for s, e in spans[k] for j in range(max(s, 1), e)]
        for j in sel:
            rows.append(st + j - 1)
            tg.append(int(ids[st + j]))
    logits = h[rows] @ lm_w.t()
    ref_loss = F.cross_entropy(logits, torch.tensor(tg), reduction="mean")
    ref_loss.backward()
    loss, n_valid, g = head_backward(final_in, lm_w, norm_w, 1e-6, ids, seq_lens, b_start,
                                     vocab, loss_spans=spans)
    cond("n_valid = number of target tokens", n_valid == len(rows), f"{n_valid} vs {len(rows)}")
    cond("loss matches", abs(loss - ref_loss.item()) < 1e-5, f"{loss:.6f} vs {ref_loss.item():.6f}")
    err = (g - x.grad).abs().max().item() / (x.grad.abs().max().item() + 1e-12)
    cond("grad_final_in matches", err < 1e-5, f"rel err {err:.1e}")
    l0, n0, g0 = head_backward(final_in, lm_w, norm_w, 1e-6, ids, seq_lens, b_start, vocab)
    l1, n1, g1 = head_backward(final_in, lm_w, norm_w, 1e-6, ids, seq_lens, b_start, vocab,
                               loss_spans=[None, None, None])
    cond("all-None spans == no spans (old path)", l0 == l1 and n0 == n1 and torch.equal(g0, g1))


def coordinator_order():
    print("coordinator: spans follow the runner's write order")
    c = FinetuneCoordinator(capacity=1024)
    sp_a, sp_b, sp_c = [(3, 5)], [(1, 4), (6, 8)], None
    # scheduler order a, b, c …
    c.reserve(30, samples=["A", "B", "C"], req_spans={"ft_a": sp_a, "ft_b": sp_b, "ft_c": sp_c})
    # … but the runner wrote b, c, a
    c.record_capture(30, [8, 12, 10], ["ft_b", "ft_c", "ft_a"])
    got = c._loss_spans()
    cond("spans in write order", got == [[[1, 4], [6, 8]], None, [[3, 5]]], str(got))
    c2 = FinetuneCoordinator(capacity=1024)
    c2.reserve(10, samples=["X"], req_spans={"x": None})
    c2.record_capture(10, [10], ["x"])
    cond("plain-text samples → None (loss on every token)", c2._loss_spans() is None)
    c3 = FinetuneCoordinator(capacity=1024)
    c3.reserve(10, samples=["X"], req_spans={"x": [(1, 2)]})
    c3.record_capture(10, [10])                     # no ids (old runner) → no masks
    cond("missing ids degrade to None, never misalign", c3._loss_spans() is None)


def store_load(tok):
    print("FinetuningStore loads .jsonl")
    rows = [{"messages": [{"role": "user", "content": "What is 2+2?"},
                          {"role": "assistant", "content": "4"}]},
            {"messages": [{"role": "user", "content": "no answer"}]}]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    st = FinetuningStore(f.name, tokenize=lambda t: tok(t)["input_ids"], max_saved_finetuning_tokens=4096,
                         chat_loss_on="ace", tokenizer=tok)
    n = st.load()
    os.unlink(f.name)
    s0 = st.samples[0] if st.samples else None
    cond("1 row loaded, the target-less row skipped", n == 1)
    ok = s0 is not None and tok.decode(s0.prompt_ids[s0.loss_spans[0][0]:s0.loss_spans[0][1]]) == "4<|im_end|>"
    cond("the span decodes to the assistant answer + <|im_end|>", ok)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/mnt/weka/home/jianshu.she/RSI-system/benchmark/ACE/sft/data_v1")
    ap.add_argument("--tokenizer", default="/mnt/weka/home/jianshu.she/scratch/Qwen3-32B")
    ap.add_argument("--rows", type=int, default=60)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    encoder_parity(tok, a.data_dir, a.rows)
    head_parity()
    coordinator_order()
    store_load(tok)
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
