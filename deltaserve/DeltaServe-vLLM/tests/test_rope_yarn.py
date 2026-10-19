"""Backward RoPE with YaRN == vLLM's YaRNScalingRotaryEmbedding (run with `python`).

RSI-system serves Qwen3-32B with YaRN x4 (131072 ctx). YaRN changes the rotary
frequencies at EVERY position and scales cos/sin by mscale, so the backward's
RoPE remat must reproduce it exactly or the LoRA gradients are wrong.

    python tests/test_rope_yarn.py [--yarn-config DIR]
"""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoConfig

from vllm.deltaserve.bwd_services.common.ops import apply_rope, rope_backward, rope_cos_sin
from vllm.deltaserve.ft_meta import rope_scaling_of
from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import (
    YaRNScalingRotaryEmbedding,
)

PASS = FAIL = 0


def cond(name, ok, detail=""):
    global PASS, FAIL
    PASS += bool(ok)
    FAIL += not ok
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


def _vllm_yarn_cache(hd, orig, theta, factor):
    """vLLM's own YaRN cos/sin cache, via its methods on a bare instance (the
    CustomOp constructor needs a device-bearing vLLM config; the math does not)."""
    from vllm.model_executor.layers.rotary_embedding.common import yarn_get_mscale
    ref = object.__new__(YaRNScalingRotaryEmbedding)
    ref.head_size = ref.rotary_dim = hd
    ref.max_position_embeddings = orig
    ref.base = theta
    ref.scaling_factor = factor
    ref.extrapolation_factor = ref.attn_factor = 1
    ref.beta_fast, ref.beta_slow, ref.truncate = 32, 1, True
    ref.mscale = float(yarn_get_mscale(factor))
    return ref._compute_cos_sin_cache()


def against_vllm(theta=1e6, hd=128, factor=4.0, orig=32768):
    print(f"cos/sin + rotation vs vLLM YaRN (theta={theta:g}, head_dim={hd}, x{factor:g})")
    cache_all = _vllm_yarn_cache(hd, orig, theta, factor)
    sc = {"rope_type": "yarn", "factor": factor, "original_max_position_embeddings": orig}
    pos = torch.tensor([0, 1, 7, 513, 4095, 32767, 40000, 131071])
    cache = cache_all[pos].float()
    cos_r, sin_r = cache[:, : hd // 2], cache[:, hd // 2:]
    cos, sin = rope_cos_sin(pos, hd, theta, sc)
    err = max((cos - cos_r).abs().max().item(), (sin - sin_r).abs().max().item())
    cond("cos/sin == vLLM YaRN cache (incl. mscale)", err < 1e-5, f"max abs err {err:.1e}")
    plain_c, _ = rope_cos_sin(pos, hd, theta)
    cond("YaRN differs from plain RoPE (so the fix matters)",
         (plain_c - cos).abs().max().item() > 1e-2)

    torch.manual_seed(0)
    n, H = len(pos), 4
    q = torch.randn(n, H, hd)
    # NeoX rotation with vLLM's cache, written out independently
    c, s_ = cos_r[:, None, :], sin_r[:, None, :]
    x1, x2 = q[..., : hd // 2], q[..., hd // 2:]
    q_r = torch.cat([x1 * c - x2 * s_, x2 * c + x1 * s_], dim=-1)
    q_o = apply_rope(q, cos, sin)
    err = (q_o - q_r).abs().max().item() / q_r.abs().max().item()
    cond("apply_rope == NeoX rotation with vLLM's cache", err < 1e-5, f"rel err {err:.1e}")
    g = torch.randn(n, H, hd)
    lhs = (g * apply_rope(q, cos, sin)).sum()
    rhs = (rope_backward(g, cos, sin) * q).sum()
    cond("rope_backward is the adjoint under mscale", abs(lhs - rhs) / abs(lhs) < 1e-5)


def config_parsing(yarn_dir):
    print("rope_scaling_of")
    class C:  # transformers 5 / vLLM-normalized form
        rope_parameters = {"rope_theta": 1e6, "rope_type": "yarn", "factor": 4.0,
                           "original_max_position_embeddings": 32768}
    got = rope_scaling_of(C())
    cond("rope_parameters (yarn) parsed", got == {"rope_type": "yarn", "factor": 4.0,
                                                  "original_max_position_embeddings": 32768}, str(got))
    class P:
        rope_parameters = {"rope_theta": 1e6, "rope_type": "default"}
    cond("plain RoPE → None", rope_scaling_of(P()) is None)
    class L:
        rope_parameters = {"rope_theta": 5e5, "rope_type": "llama3", "factor": 8.0}
    try:
        rope_scaling_of(L())
        cond("unsupported scaling raises", False)
    except NotImplementedError:
        cond("unsupported scaling raises", True)
    if yarn_dir:
        try:
            got = rope_scaling_of(AutoConfig.from_pretrained(yarn_dir))
            cond(f"RSI-system YaRN model config parsed ({yarn_dir.rsplit('/', 1)[-1]})",
                 got is not None and got["factor"] == 4.0
                 and got["original_max_position_embeddings"] == 32768, str(got))
        except OSError as e:
            print(f"  (skip real config: {e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yarn-config",
                    default="/mnt/weka/home/jianshu.she/RSI-system/benchmark/ACE/models/Qwen3-32B-yarn131k")
    a = ap.parse_args()
    against_vllm()
    config_parsing(a.yarn_config)
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
