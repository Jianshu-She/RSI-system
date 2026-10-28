# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-sample GQA causal attention: eager forward and backward cores.

Both loop over the samples of the packed batch (``seq_lens`` / ``b_start``),
so shapes vary per sample and neither is CUDA-graph-capturable; the graphed
runner (``common/graph.py``) has padded rewrites of the same math and falls
back to these when a batch overflows the padded bounds. Scores/softmax and
the dQ/dK/dV products always run in fp32 (the load-bearing GQA precision rule;
downgrading them plateaued Llama-3 loss in DeltaServe). ``dims`` is
``(Hq, Hkv, Hd, kv_size)``.

Long samples (the ``[Hq, L, L]`` fp32 scores would exceed ``SCORES_BYTES``)
take a query-chunked path: each block of query rows sees only the keys it can
attend to (``0 .. block_end``), so every intermediate is ``[Hq, C, block_end]``
instead of ``[Hq, L, L]``. The dropped keys are exactly the masked ones, so the
math is the dense math; only the GEMM/reduction tiling differs. Short samples
keep the dense path unchanged.

``set_sdpa(True)`` (meta ``backward_attn_sdpa``) swaps the eager backward core for
PyTorch's fused SDPA kernel (flash attention): bf16 q/k/v, fp32 accumulation and
softmax statistics inside the kernel, the per-sample causal attention recomputed
and differentiated by ``torch.autograd.grad``. It replaces the materialized fp32
``[Hq, L, L]`` scores, mask and softmax (most of a layer's backward at 16K tokens).
"""

import math
import os

import torch

# Per-intermediate fp32 budget for one sample's scores-shaped tensors. Above it
# the cores chunk over query rows (512 MB → C=512 rows at Hq=16, L=16K).
SCORES_BYTES = int(os.environ.get("DSERVE_ATTN_SCORES_BYTES", 512 << 20))

_SDPA = False


def set_sdpa(enabled: bool) -> None:
    """Route ``attn_backward_core`` through the fused SDPA kernel (see module doc)."""
    global _SDPA
    _SDPA = bool(enabled)


def _sdpa_backward(q, k, v, g, scale):
    """One sample's causal attention backward through the fused SDPA kernel.
    ``q`` [Hq, L, Hd], ``k``/``v`` [Hq, L, Hd] (GQA-repeated), ``g`` [Hq, L, Hd],
    all bf16/fp16. Returns ``(gq, gk, gv)`` in the same layout and dtype."""
    q, k, v = (t.detach().unsqueeze(0).requires_grad_() for t in (q, k, v))
    with torch.enable_grad():
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scale)
        gq, gk, gv = torch.autograd.grad(out, (q, k, v), g.unsqueeze(0))
    return gq[0], gk[0], gv[0]


def _q_chunk(hq: int, ln: int) -> int:
    """Query rows per chunk for a sample of length ``ln``; ``ln`` = dense."""
    if hq * ln * ln * 4 <= SCORES_BYTES:
        return ln
    return max(16, SCORES_BYTES // (hq * ln * 4))


def attn_forward_core(qh, kh, vh, seq_lens, b_start, dims, out_dtype):
    """Causal GQA attention over the packed batch. ``qh`` [n, Hq, Hd],
    ``kh``/``vh`` [n, Hkv, Hd] (post-RoPE). Returns the flat context
    ``[n, Hq*Hd]`` in ``out_dtype`` — the o_proj input."""
    Hq, Hkv, Hd, _ = dims
    n = qh.shape[0]
    scale = 1.0 / math.sqrt(Hd)
    kv_repeat = Hq // Hkv
    ctx_blocks = []
    for st, ln in zip(b_start, seq_lens):
        q_blk = qh[st:st + ln].transpose(0, 1)      # [Hq, L, Hd]
        k_blk = kh[st:st + ln].transpose(0, 1)      # [Hkv, L, Hd]
        v_blk = vh[st:st + ln].transpose(0, 1)
        if kv_repeat != 1:
            k_rep = k_blk.repeat_interleave(kv_repeat, 0)
            v_rep = v_blk.repeat_interleave(kv_repeat, 0)
        else:
            k_rep, v_rep = k_blk, v_blk
        C = _q_chunk(Hq, ln)
        if C < ln:
            ctx_blocks.append(_attn_forward_chunked(
                q_blk.float(), k_rep.float(), v_rep.float(), scale, C,
                out_dtype))
            continue
        scores = (q_blk.float() @ k_rep.float().transpose(-1, -2)) * scale
        mask = torch.triu(
            torch.ones(ln, ln, dtype=torch.bool, device=qh.device), 1)
        scores = scores.masked_fill(mask, -1e9)
        att = torch.softmax(scores, dim=-1)         # [Hq, L, L] fp32
        ctx_blk = (att @ v_rep.float()).to(out_dtype).transpose(0, 1)  # [L, Hq, Hd]
        ctx_blocks.append(ctx_blk)
    return torch.cat(ctx_blocks, 0).reshape(n, Hq * Hd)


def _causal_mask(qs: int, qe: int, kend: int, device) -> torch.Tensor:
    """True where key ``j`` is in the future of query ``qs + r`` (rows qs..qe)."""
    return torch.ones(qe - qs, kend, dtype=torch.bool, device=device).triu(qs + 1)


def _attn_forward_chunked(q, k, v, scale, C, out_dtype):
    """Query-chunked causal attention for one sample. ``q``/``k``/``v``
    ``[Hq, L, Hd]`` fp32 (k/v already GQA-repeated). Returns ``[L, Hq, Hd]``."""
    ln = q.shape[1]
    ctx = q.new_empty((ln, q.shape[0], q.shape[2]), dtype=out_dtype)
    for qs in range(0, ln, C):
        qe = min(qs + C, ln)
        scores = (q[:, qs:qe] @ k[:, :qe].transpose(-1, -2)) * scale
        scores.masked_fill_(_causal_mask(qs, qe, qe, q.device), -1e9)
        att = torch.softmax(scores, dim=-1)        # [Hq, c, qe] fp32
        ctx[qs:qe] = (att @ v[:, :qe]).to(out_dtype).transpose(0, 1)
    return ctx


def _attn_backward_chunked(q, k, v, g, scale, C):
    """Query-chunked attention backward for one sample. ``q``/``k``/``v``/``g``
    ``[Hq, L, Hd]`` fp32 (k/v GQA-repeated). Each query block recomputes its
    softmax rows over keys ``0 .. block_end`` and adds its dK/dV contribution
    to those keys. Returns ``(gq, gk, gv)``, each ``[Hq, L, Hd]`` fp32."""
    ln = q.shape[1]
    gq = torch.empty_like(q)
    gk = torch.zeros_like(k)
    gv = torch.zeros_like(v)
    for qs in range(0, ln, C):
        qe = min(qs + C, ln)
        q_c, k_c, v_c, g_c = q[:, qs:qe], k[:, :qe], v[:, :qe], g[:, qs:qe]
        mask = _causal_mask(qs, qe, qe, q.device)
        scores = (q_c @ k_c.transpose(-1, -2)) * scale
        scores.masked_fill_(mask, -1e9)
        att = torch.softmax(scores, dim=-1)        # [Hq, c, qe] fp32
        del scores
        grad_att = g_c @ v_c.transpose(-1, -2)
        gv[:, :qe] += att.transpose(-1, -2) @ g_c
        sm = (grad_att * att).sum(-1, keepdim=True)
        grad_scores = (att * (grad_att - sm)).masked_fill_(mask, 0.0)
        del att, grad_att
        gq[:, qs:qe] = (grad_scores @ k_c) * scale
        gk[:, :qe] += (grad_scores.transpose(-1, -2) @ q_c) * scale
    return gq, gk, gv


def attn_backward_core(qh, kh, vh, grad_ctx, seq_lens, b_start, dims, cdt=torch.float32,
                       *, grad_qh_buf=None, grad_kh_buf=None, grad_vh_buf=None):
    """Per-sample GQA attention backward (eager). Returns (grad_qh, grad_kh,
    grad_vh) in flat ``[n, H_*, Hd]`` layout, in ``cdt``. Per-sample shapes
    (variable seq_lens) prevent CUDA-graph capture; the runner uses a padded
    variant for the graphed fast path and silently falls back to this when a
    batch overflows the padded bounds. Scores/softmax/dQ/dK/dV always run in
    fp32 (the load-bearing GQA precision rule).

    If ``grad_qh_buf/kh_buf/vh_buf`` are provided (persistent buffers sized at
    s_max), the function zeroes their first n rows in-place and returns views
    into them — saves L=32 fresh-allocation zero-fills per backward. Falls
    back to fresh allocation when not provided (preserves the gradcheck path,
    which has no service to own persistent buffers)."""
    Hq, Hkv, Hd, _ = dims
    scale = 1.0 / math.sqrt(Hd)
    kv_repeat = Hq // Hkv
    n = grad_ctx.shape[0]
    device = grad_ctx.device
    # Use the persistent buffers only when they're large enough for n. They're
    # sized at s_max for the production path (n ≤ s_max always); the runner's
    # eager-fallback inside GraphedBackward.attn_backward also passes its own
    # static_grad_* (also s_max). The undersized branch fires only in contrived
    # parity tests that exceed s_max — we silently alloc fresh.
    if grad_qh_buf is not None and grad_qh_buf.shape[0] >= n:
        grad_qh_buf[:n].zero_()
        grad_kh_buf[:n].zero_()
        grad_vh_buf[:n].zero_()
        grad_qh = grad_qh_buf[:n]
        grad_kh = grad_kh_buf[:n]
        grad_vh = grad_vh_buf[:n]
    else:
        grad_qh = torch.zeros((n, Hq, Hd), dtype=cdt, device=device)
        grad_kh = torch.zeros((n, Hkv, Hd), dtype=cdt, device=device)
        grad_vh = torch.zeros((n, Hkv, Hd), dtype=cdt, device=device)
    if _SDPA:
        lp = qh.dtype if qh.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        for st, ln in zip(b_start, seq_lens):
            q_blk = qh[st:st + ln].transpose(0, 1).to(lp)       # [Hq, L, Hd]
            k_blk = kh[st:st + ln].transpose(0, 1).to(lp)       # [Hkv, L, Hd]
            v_blk = vh[st:st + ln].transpose(0, 1).to(lp)
            if kv_repeat != 1:
                k_blk = k_blk.repeat_interleave(kv_repeat, 0)
                v_blk = v_blk.repeat_interleave(kv_repeat, 0)
            g = grad_ctx[st:st + ln].transpose(0, 1).to(lp)
            gq, gk, gv = _sdpa_backward(q_blk, k_blk, v_blk, g, scale)
            if kv_repeat != 1:
                gk = gk.float().view(Hkv, kv_repeat, ln, Hd).sum(1)
                gv = gv.float().view(Hkv, kv_repeat, ln, Hd).sum(1)
            grad_qh[st:st + ln] = gq.transpose(0, 1).to(cdt)
            grad_kh[st:st + ln] = gk.transpose(0, 1).to(cdt)
            grad_vh[st:st + ln] = gv.transpose(0, 1).to(cdt)
        return grad_qh, grad_kh, grad_vh
    qh_f, kh_f, vh_f = qh.float(), kh.float(), vh.float()
    for st, ln in zip(b_start, seq_lens):
        q_blk = qh_f[st:st + ln].transpose(0, 1)     # [Hq, L, Hd] fp32
        k_blk = kh_f[st:st + ln].transpose(0, 1)     # [Hkv, L, Hd] fp32
        v_blk = vh_f[st:st + ln].transpose(0, 1)
        if kv_repeat != 1:
            k_rep = k_blk.repeat_interleave(kv_repeat, 0)
            v_rep = v_blk.repeat_interleave(kv_repeat, 0)
        else:
            k_rep, v_rep = k_blk, v_blk
        g = grad_ctx[st:st + ln].transpose(0, 1).float()     # fp32
        C = _q_chunk(Hq, ln)
        if C < ln:
            grad_q_blk, grad_k_rep, grad_v_rep = _attn_backward_chunked(
                q_blk, k_rep, v_rep, g, scale, C)
        else:
            mask = torch.triu(
                torch.ones(ln, ln, dtype=torch.bool, device=device), 1)
            scores = (q_blk @ k_rep.transpose(-1, -2)) * scale   # fp32
            scores = scores.masked_fill(mask, -1e9)
            att = torch.softmax(scores, dim=-1)          # [Hq, L, L] fp32
            grad_att = g @ v_rep.transpose(-1, -2)       # [Hq, L, L]
            grad_v_rep = att.transpose(-1, -2) @ g       # [Hq, L, Hd]
            sm = (grad_att * att).sum(-1, keepdim=True)
            grad_scores = (att * (grad_att - sm)).masked_fill(mask, 0.0)
            grad_q_blk = (grad_scores @ k_rep) * scale
            grad_k_rep = (grad_scores.transpose(-1, -2) @ q_blk) * scale
        if kv_repeat != 1:
            grad_k_kv = grad_k_rep.view(Hkv, kv_repeat, ln, Hd).sum(1)
            grad_v_kv = grad_v_rep.view(Hkv, kv_repeat, ln, Hd).sum(1)
        else:
            grad_k_kv, grad_v_kv = grad_k_rep, grad_v_rep
        grad_qh[st:st + ln] = grad_q_blk.transpose(0, 1).to(cdt)
        grad_kh[st:st + ln] = grad_k_kv.transpose(0, 1).to(cdt)
        grad_vh[st:st + ln] = grad_v_kv.transpose(0, 1).to(cdt)
    return grad_qh, grad_kh, grad_vh
