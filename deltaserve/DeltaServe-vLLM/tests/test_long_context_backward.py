"""Long-context backward gates (run with `python`, not pytest).

1. Query-chunked attention forward/backward == the dense path (fp32).
2. Streaming LM head (online logsumexp) == the dense head (loss + grad).
3. Coordinator ``batch_tokens``: an empty buffer takes one long sample, a
   non-empty one only tops up to the batch size.
4. (GPU) one Qwen3-32B-TP4-shaped layer's attention backward at 16K tokens:
   peak memory and time, on the chunked path.

    python tests/test_long_context_backward.py
"""

from __future__ import annotations

import sys
import time

import torch

from vllm.deltaserve.bwd_services.common import attention as A
from vllm.deltaserve.bwd_services.common import head as Hd
from vllm.deltaserve.coordinator import FinetuneCoordinator

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PASS = FAIL = 0


def check(name, got, ref, tol):
    global PASS, FAIL
    err = (got.float() - ref.float()).abs().max().item()
    scale = ref.float().abs().max().item() + 1e-12
    ok = err / scale <= tol
    PASS += ok
    FAIL += not ok
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: rel max err {err / scale:.2e}")


def cond(name, ok):
    global PASS, FAIL
    PASS += bool(ok)
    FAIL += not ok
    print(f"  [{'ok' if ok else 'FAIL'}] {name}")


def attn_parity():
    print("attention: chunked vs dense")
    torch.manual_seed(0)
    Hq, Hkv, D = 8, 2, 64
    seq_lens = [37, 300, 1000, 17]
    b_start = [sum(seq_lens[:i]) for i in range(len(seq_lens))]
    n = sum(seq_lens)
    dims = (Hq, Hkv, D, Hkv * D)
    qh = torch.randn(n, Hq, D, device=DEV)
    kh = torch.randn(n, Hkv, D, device=DEV)
    vh = torch.randn(n, Hkv, D, device=DEV)
    g = torch.randn(n, Hq, D, device=DEV)
    big = A.SCORES_BYTES
    try:
        A.SCORES_BYTES = 1 << 62                       # dense everywhere
        ref_ctx = A.attn_forward_core(qh, kh, vh, seq_lens, b_start, dims,
                                      torch.float32)
        ref = A.attn_backward_core(qh, kh, vh, g, seq_lens, b_start, dims)
        for budget in (Hq * 1000 * 4 * 100, Hq * 1000 * 4 * 16, 4096):
            A.SCORES_BYTES = budget                    # chunk the long ones
            cs = [A._q_chunk(Hq, ln) for ln in seq_lens]
            ctx = A.attn_forward_core(qh, kh, vh, seq_lens, b_start, dims,
                                      torch.float32)
            got = A.attn_backward_core(qh, kh, vh, g, seq_lens, b_start, dims)
            cond(f"budget={budget} chunks={cs} takes the chunked path",
                 any(c < ln for c, ln in zip(cs, seq_lens)))
            check(f"budget={budget} ctx", ctx, ref_ctx, 1e-5)
            for nm, a, b in zip(("dq", "dk", "dv"), got, ref):
                check(f"budget={budget} {nm}", a, b, 1e-5)
    finally:
        A.SCORES_BYTES = big


def head_parity():
    print("head: streaming vs dense")
    torch.manual_seed(1)
    D, vocab, vpad = 64, 40000, 40064               # > 2 vocab chunks + padding
    seq_lens = [50, 3, 1, 200]
    b_start = [sum(seq_lens[:i]) for i in range(len(seq_lens))]
    n = sum(seq_lens)
    final_in = torch.randn(n, D, device=DEV, dtype=torch.bfloat16)
    lm_w = (torch.randn(vpad, D, device=DEV) * 0.05).to(torch.bfloat16)
    norm_w = torch.rand(D, device=DEV) + 0.5
    ids = torch.randint(0, vocab, (n,), device=DEV)
    ids[5] = vocab - 1                               # target in the last chunk
    ids[6] = 0
    big = Hd.LOGITS_BYTES
    try:
        Hd.LOGITS_BYTES = 1 << 62
        l_ref, nv_ref, g_ref = Hd.head_backward(final_in, lm_w, norm_w, 1e-6, ids,
                                                seq_lens, b_start, vocab)
        Hd.LOGITS_BYTES = 0
        l, nv, g = Hd.head_backward(final_in, lm_w, norm_w, 1e-6, ids,
                                    seq_lens, b_start, vocab)
    finally:
        Hd.LOGITS_BYTES = big
    cond(f"n_valid {nv} == {nv_ref}", nv == nv_ref)
    cond(f"loss {l:.6f} == {l_ref:.6f}", abs(l - l_ref) <= 1e-5 * abs(l_ref))
    check("grad_final_in", g, g_ref, 1e-5)


def coordinator_batching():
    print("coordinator: batch_tokens")
    c = FinetuneCoordinator(capacity=16384)
    cond("default batch_tokens == capacity", c.batch_tokens == 16384)
    c.batch_tokens = 512
    cond("empty buffer offers the full capacity", c.space_remaining() == 16384)
    c.fill_count = 300
    cond("non-empty buffer tops up to batch_tokens", c.space_remaining() == 212)
    c.fill_count = 10000
    cond("a long sample alone leaves no space", c.space_remaining() == 0)
    c.fill_count, c.reserved_fill = 0, 200
    cond("reservations count as used", c.space_remaining() == 312)
    c2 = FinetuneCoordinator(capacity=512)
    c2.fill_count = 100
    cond("batch_tokens == capacity keeps the original arithmetic",
         c2.space_remaining() == 412)


def long_attention_cost():
    if DEV != "cuda":
        print("long attention: skipped (no GPU)")
        return
    print("long attention: Qwen3-32B TP=4 shard (Hq=16, Hkv=2, Hd=128), L=16384")
    torch.manual_seed(2)
    Hq, Hkv, D, L = 16, 2, 128, 16384
    dims = (Hq, Hkv, D, Hkv * D)
    qh = torch.randn(L, Hq, D, device=DEV, dtype=torch.bfloat16)
    kh = torch.randn(L, Hkv, D, device=DEV, dtype=torch.bfloat16)
    vh = torch.randn(L, Hkv, D, device=DEV, dtype=torch.bfloat16)
    g = torch.randn(L, Hq, D, device=DEV, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    out = A.attn_backward_core(qh, kh, vh, g, [L], [0], dims, cdt=torch.bfloat16)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() - base) / 2**30
    dense = 4 * Hq * L * L * 4 / 2**30
    print(f"  chunk={A._q_chunk(Hq, L)} rows, {dt * 1e3:.0f} ms/layer, "
          f"peak +{peak:.2f} GiB (dense path would need ~{dense:.0f} GiB)")
    cond("finite grads", all(torch.isfinite(t).all().item() for t in out))
    cond("peak memory < 4 GiB", peak < 4)


if __name__ == "__main__":
    attn_parity()
    head_parity()
    coordinator_batching()
    long_attention_cost()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
