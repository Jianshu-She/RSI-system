#!/usr/bin/env python
"""backward_attn_sdpa: the fused SDPA attention backward == the fp32 per-sample core.

GPU. Packed batches of several causal samples with GQA (Hq=16, Hkv=2, Hd=128, the
Qwen3-32B TP=4 shard), bf16 inputs. The SDPA path runs bf16 q/k/v through the flash
kernel (fp32 accumulate), so it is compared with the fp32 core at bf16-level tolerance
(relative L2 per gradient), plus exact layout checks: per-sample isolation (a sample's
gradient does not depend on its neighbours) and the persistent-buffer path.

    python tests/test_attn_sdpa.py
"""
import torch

from vllm.deltaserve.bwd_services.common import attention as A

PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    PASS += bool(ok)
    FAIL += not ok
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {detail}")


def rel(a, b):
    return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-30)).item()


def run(seq_lens, Hq=16, Hkv=2, Hd=128, seed=0):
    torch.manual_seed(seed)
    n = sum(seq_lens)
    b_start = [sum(seq_lens[:i]) for i in range(len(seq_lens))]
    dims = (Hq, Hkv, Hd, Hkv * Hd)
    qh = torch.randn(n, Hq, Hd, device="cuda", dtype=torch.bfloat16)
    kh = torch.randn(n, Hkv, Hd, device="cuda", dtype=torch.bfloat16)
    vh = torch.randn(n, Hkv, Hd, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(n, Hq * Hd, device="cuda", dtype=torch.bfloat16)
    A.set_sdpa(False)
    ref = A.attn_backward_core(qh, kh, vh, g.view(n, Hq, Hd), seq_lens, b_start, dims,
                               cdt=torch.float32)
    A.set_sdpa(True)
    got = A.attn_backward_core(qh, kh, vh, g.view(n, Hq, Hd), seq_lens, b_start, dims,
                               cdt=torch.float32)
    A.set_sdpa(False)
    return ref, got, (qh, kh, vh, g, b_start, dims)


def main():
    print("SDPA vs fp32 core:")
    for lens in ([64], [600, 2400], [2400] * 3 + [600, 17], [1, 5, 300]):
        ref, got, _ = run(lens)
        for name, r, x in zip(("dq", "dk", "dv"), ref, got):
            e = rel(x, r)
            check(f"lens {lens} {name}", e < 2e-2, f"rel-L2 {e:.2e}")
    print("per-sample isolation (sample 1 of a batch == the same sample alone):")
    _, got_b, (qh, kh, vh, g, _, dims) = run([300, 500])
    A.set_sdpa(True)
    s1 = slice(300, 800)
    alone = A.attn_backward_core(qh[s1], kh[s1], vh[s1], g[s1].view(500, *qh.shape[1:]), [500], [0],
                                 dims, cdt=torch.float32)
    A.set_sdpa(False)
    for name, a1, b1 in zip(("dq", "dk", "dv"), alone, got_b):
        check(f"{name}", rel(b1[s1], a1) < 1e-6)
    print("persistent-buffer path:")
    _, got, (qh, kh, vh, g, b_start, dims) = run([200, 300])
    n = qh.shape[0]
    bufs = [torch.full((n + 7, *t.shape[1:]), 9.0, device="cuda") for t in (qh, kh, vh)]
    A.set_sdpa(True)
    out = A.attn_backward_core(qh, kh, vh, g.view(n, *qh.shape[1:]), [200, 300], b_start, dims,
                               cdt=torch.float32, grad_qh_buf=bufs[0], grad_kh_buf=bufs[1],
                               grad_vh_buf=bufs[2])
    A.set_sdpa(False)
    for name, a, b, buf in zip(("dq", "dk", "dv"), out, got, bufs):
        check(f"{name} view into the buffer", a.data_ptr() == buf.data_ptr() and torch.equal(a, b))
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
