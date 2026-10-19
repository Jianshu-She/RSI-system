# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Final RMSNorm + LM head: per-sample shifted cross-entropy loss and the
gradient w.r.t. the pre-final-norm residual (``final_in``).

The LM head runs in fp32 (the precision contract): fp32 inputs, fp32
accumulation, fp32 softmax / cross-entropy. The head weight is shared from
vLLM in the model dtype (bf16), so each vocab chunk is converted to fp32 —
ONCE per pass per cycle — right before its GEMM. All samples' predicting
rows are batched into one GEMM per chunk: the loss and gradient are row-wise
independent, so the per-sample structure (shift-by-one targets, ``n_valid``
normalisation) is pure row indexing and batching changes no arithmetic
beyond the GEMM tile order. Two passes over the head per cycle (logits, then
``logit_grad @ W``), ~2 × the head's bytes of conversion traffic.

Before this restructure the head was computed per sample: 8 conversions of
the whole head + 8 GEMMs with ~31 rows each, which on Qwen3-14B cost ~63 ms
of a ~165 ms uncontended cycle (profiled 2026-09-08). The head is frozen;
only ``grad_final_in`` flows back into the layer stack.
"""

import os

import torch
import torch.nn.functional as F

from vllm.deltaserve.bwd_services.common.ops import rmsnorm, rmsnorm_backward

VOCAB_CHUNK = 16384
# Above this many bytes of fp32 logits ([n_valid, vocab]) the head streams:
# an online logsumexp over vocab chunks instead of materializing the logits
# (16K rows x 152K vocab would be ~10 GB, twice over for the softmax).
LOGITS_BYTES = int(os.environ.get("DSERVE_HEAD_LOGITS_BYTES", 2 << 30))


def logits_chunked(h: torch.Tensor, lm_w: torch.Tensor, vocab: int):
    """fp32 logits = h @ lm_w[:vocab].T, computed in vocab chunks (bounds the fp32
    LM-head temporary). h [m, D], lm_w [vocab_pad, D]. Returns [m, vocab] fp32."""
    hf = h.float()
    out = hf.new_empty((hf.shape[0], vocab))
    for c in range(0, vocab, VOCAB_CHUNK):
        e = min(c + VOCAB_CHUNK, vocab)
        out[:, c:e] = hf @ lm_w[c:e].float().t()
    return out


def head_backward(final_in, lm_w, norm_w, eps, ids, seq_lens, b_start, vocab,
                  pause_fn=None, loss_spans=None):
    """LM-head + final-norm: per-sample shift CE loss + grad w.r.t. final_in.
    Returns (loss: float, n_valid: int, grad_final_in [n,D] fp32).

    ``pause_fn`` (the service's ``_maybe_pause``) is called once per vocab
    chunk in both passes: the head is ~18 ms on Qwen3-14B and used to be one
    uninterruptible block at the very start of the cycle — exactly where the
    first inference prefill after the buffer-full trigger lands.

    Rows ``st .. st+ln-2`` of each sample predict ids ``st+1 .. st+ln-1``;
    samples with ``ln < 2`` contribute nothing. The loss is the SUM of the
    per-row cross-entropies divided by ``n_valid`` (the number of predicting
    rows), and the gradient is ``(softmax - onehot) / n_valid`` — identical to
    the original per-sample loop, computed for all rows at once.

    ``loss_spans`` (chat SFT): per sample, None or a list of target spans
    ``(s, e)`` relative to the sample start — only the rows predicting a token
    at positions ``s .. e-1`` carry loss (the row at ``j - 1`` predicts ``j``),
    and ``n_valid`` counts those rows. None for the whole call = every row
    (the original behaviour)."""
    # rmsnorm computes in fp32 but returns the input dtype (bf16 in prod);
    # the head contract is fp32 end to end, so upcast the rows once here.
    normed = rmsnorm(final_in, norm_w, eps)
    n, D = final_in.shape
    device = final_in.device
    rows, tgts = [], []
    for k, (st, ln) in enumerate(zip(b_start, seq_lens)):
        if ln < 2:
            continue
        spans = loss_spans[k] if loss_spans is not None else None
        if spans is None:
            rows.append(torch.arange(st, st + ln - 1, device=device))
            tgts.append(ids[st + 1:st + ln].long())
            continue
        for s, e in spans:
            s, e = max(int(s), 1), min(int(e), ln)    # position 0 has no predictor
            if e <= s:
                continue
            rows.append(torch.arange(st + s - 1, st + e - 1, device=device))
            tgts.append(ids[st + s:st + e].long())
    n_valid = sum(int(r.numel()) for r in rows)
    if n_valid == 0:
        return 0.0, 0, final_in.new_zeros((n, D), dtype=torch.float32)
    rows = torch.cat(rows)
    tgt = torch.cat(tgts)
    h = normed[rows].float()                          # [m, D] fp32, m = n_valid
    if n_valid * vocab * 4 > LOGITS_BYTES:
        loss_sum, grad_rows = _head_streaming(h, tgt, lm_w, vocab, pause_fn)
        grad_normed = grad_rows.new_zeros((n, D))
        grad_normed[rows] = grad_rows
        grad_final_in = rmsnorm_backward(final_in, grad_normed, norm_w, eps)
        return float(loss_sum / n_valid), n_valid, grad_final_in

    # Pass 1: fp32 logits for every predicting row, one GEMM per vocab chunk
    # (the bf16 head chunk converted once here).
    logits = h.new_empty((n_valid, vocab))
    for c in range(0, vocab, VOCAB_CHUNK):
        if pause_fn is not None:
            pause_fn()
        e = min(c + VOCAB_CHUNK, vocab)
        logits[:, c:e] = h @ lm_w[c:e].float().t()
    total_loss = F.cross_entropy(logits, tgt, reduction="sum")
    p = torch.softmax(logits, dim=-1)
    del logits
    p[torch.arange(n_valid, device=device), tgt] -= 1.0
    p /= n_valid                                      # dL/dlogits, [m, vocab]

    # Pass 2: grad w.r.t. the normed rows = p @ W, chunked over vocab (the
    # head chunk converted once again); scatter back to the full [n, D].
    grad_rows = p.new_zeros((n_valid, D))
    for c in range(0, vocab, VOCAB_CHUNK):
        if pause_fn is not None:
            pause_fn()
        e = min(c + VOCAB_CHUNK, vocab)
        grad_rows += p[:, c:e] @ lm_w[c:e].float()
    grad_normed = grad_rows.new_zeros((n, D))         # fp32 [n, D]
    grad_normed[rows] = grad_rows
    grad_final_in = rmsnorm_backward(final_in, grad_normed, norm_w, eps)
    return float(total_loss.item() / n_valid), n_valid, grad_final_in


def _head_streaming(h, tgt, lm_w, vocab, pause_fn=None):
    """Same loss and gradient as the dense path without the ``[m, vocab]``
    logits: pass 1 keeps a running max / sum-exp per row (online logsumexp)
    and picks out each row's target logit; pass 2 recomputes each vocab
    chunk's logits and accumulates ``(softmax - onehot) / m @ W``. Each head
    chunk is converted to fp32 once per pass, as in the dense path; the
    logits GEMM runs twice. Returns ``(loss_sum: float, grad_rows [m, D])``."""
    m = h.shape[0]
    run_max = h.new_full((m,), float("-inf"))
    run_sum = h.new_zeros((m,))
    tgt_logit = h.new_zeros((m,))
    for c in range(0, vocab, VOCAB_CHUNK):
        if pause_fn is not None:
            pause_fn()
        e = min(c + VOCAB_CHUNK, vocab)
        logits = h @ lm_w[c:e].float().t()            # [m, Vc] fp32
        new_max = torch.maximum(run_max, logits.max(dim=-1).values)
        run_sum = (run_sum * torch.exp(run_max - new_max)
                   + torch.exp(logits - new_max[:, None]).sum(dim=-1))
        run_max = new_max
        sel = (tgt >= c) & (tgt < e)
        idx = sel.nonzero(as_tuple=True)[0]
        tgt_logit[idx] = logits[idx, tgt[idx] - c]
        del logits
    lse = run_max + torch.log(run_sum)                # [m]
    loss_sum = float((lse - tgt_logit).sum().item())

    grad_rows = h.new_zeros(h.shape)
    for c in range(0, vocab, VOCAB_CHUNK):
        if pause_fn is not None:
            pause_fn()
        e = min(c + VOCAB_CHUNK, vocab)
        w = lm_w[c:e].float()
        p = torch.exp(h @ w.t() - lse[:, None])       # softmax chunk [m, Vc]
        sel = (tgt >= c) & (tgt < e)
        idx = sel.nonzero(as_tuple=True)[0]
        p[idx, tgt[idx] - c] -= 1.0
        p /= m
        grad_rows += p @ w
        del p, w
    return loss_sum, grad_rows
