# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-derived constants the backward child needs (the ``meta`` dict).

Kept outside the worker so every read of ``hf_config`` / the adapter config
that the finetuning path depends on lives in one deltaserve-owned place, with
no model-family branches. ``build_backward_meta`` is the single producer of the
dict that ``BackwardProcess.share_weights`` ships to the child and that the
trainer's ``_build_state`` consumes.
"""

from __future__ import annotations

import json
import os
from typing import Any

from vllm.deltaserve import dprint

_DEFAULT_ROPE_THETA = 10000.0


def rope_theta_of(hf_config: Any) -> float:
    """RoPE base frequency for the backward's RoPE rematerialization.

    transformers >= 5 (and vLLM's config normalization) move the value into
    ``hf_config.rope_parameters["rope_theta"]`` and drop the top-level
    ``rope_theta`` attribute. Reading only the attribute silently yields the
    10000 default for every modern model (Llama-3 uses 5e5, Qwen3 1e6), which
    desynchronizes the remat from the served forward — so check the nested
    dict first and warn loudly when neither carries a value."""
    rp = getattr(hf_config, "rope_parameters", None)
    if isinstance(rp, dict) and rp.get("rope_theta") is not None:
        return float(rp["rope_theta"])
    theta = getattr(hf_config, "rope_theta", None)
    if theta is not None:
        return float(theta)
    dprint("[deltaserve] WARNING: hf_config carries no rope_theta "
           f"(rope_parameters={rp!r}); backward RoPE falls back to "
           f"{_DEFAULT_ROPE_THETA}. The remat will NOT match the served model "
           "if it uses a different base frequency.")
    return _DEFAULT_ROPE_THETA


_YARN_KEYS = ("extrapolation_factor", "attn_factor", "beta_fast", "beta_slow",
              "apply_yarn_scaling", "truncate")


def rope_scaling_of(hf_config: Any) -> dict | None:
    """RoPE scaling for the backward's RoPE remat, in the form
    ``bwd_services.common.ops.rope_cos_sin`` takes: None (plain RoPE) or
    ``{"rope_type": "yarn", "factor", "original_max_position_embeddings", …}``
    with the same optional YaRN keys vLLM's ``get_rope`` forwards. Any other
    scaling type raises: the backward would otherwise silently rematerialize
    RoPE differently from the served forward (e.g. Qwen3-32B served with
    YaRN x4 for 131k context, as RSI-system does)."""
    rp = getattr(hf_config, "rope_parameters", None)
    if not isinstance(rp, dict):
        rp = getattr(hf_config, "rope_scaling", None)
    if not isinstance(rp, dict):
        return None
    kind = rp.get("rope_type", rp.get("type", "default"))
    if kind in (None, "default"):
        return None
    if kind == "yarn" and "mrope_section" not in rp:
        out = {"rope_type": "yarn", "factor": float(rp["factor"]),
               "original_max_position_embeddings":
                   int(rp["original_max_position_embeddings"])}
        out.update({k: rp[k] for k in _YARN_KEYS if k in rp})
        return out
    raise NotImplementedError(
        f"[deltaserve] the backward's RoPE remat does not implement rope "
        f"scaling {rp!r}; supported: none or plain yarn")


def read_lora_scaling(adapter_path: str | None) -> float:
    """``lora_alpha / r`` from the adapter's ``adapter_config.json`` (1.0 if
    the file is absent). The backward bakes this into B at publish time."""
    if not adapter_path:
        return 1.0
    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return 1.0
    with open(cfg_path) as f:
        acfg = json.load(f)
    r = float(acfg.get("r", 1) or 1)
    alpha = float(acfg.get("lora_alpha", r))
    return alpha / r


def effective_save_attn_qkv(arch: str, ft_cfg: Any) -> bool:
    """``finetune.save_attn_qkv`` as the accumulator should apply it: off when
    the family's backward cannot consume saved q/k at all, so the worker does
    not allocate and fill buffers the child will ignore."""
    if not ft_cfg.save_attn_qkv:
        return False
    from vllm.deltaserve.bwd_services.registry import get_family

    fam = get_family(arch)
    if fam is not None and not fam.supports_saved_qkv:
        dprint(f"[deltaserve] save_attn_qkv is not exact for {fam.name}; "
               "the backward recomputes Q/K/V — not allocating attn_qkv buffers")
        return False
    return True


def saved_qkv_pre_transform(arch: str) -> bool:
    """Whether the accumulator must capture q/k at the INPUT of the family's
    per-head transform modules (``self_attn.q_norm`` / ``k_norm`` — Qwen3)
    instead of post-RoPE at ``self_attn.attn`` (Llama-3). See
    ``Family.saved_qkv_pre_transform``."""
    from vllm.deltaserve.bwd_services.registry import get_family

    fam = get_family(arch)
    return bool(fam is not None and fam.saved_qkv_pre_transform)


def build_backward_meta(hf_config: Any, ft_cfg: Any, *, lm_head_key: str | None,
                        embed_weight_key: str | None, has_final_norm: bool,
                        lora_scaling: float, tp_size: int, tp_rank: int) -> dict:
    """The ``meta`` dict shipped to the backward child with the shared weights.

    Model dims (for the per-layer forward rematerialization), the LM-head /
    norm / embedding weight names the worker resolved, optimizer + backward
    feature flags from ``FinetuneConfig``, and this rank's TP geometry. Every
    read is a generic ``hf_config`` attribute; nothing here is family-specific.
    ``head_dim`` falls back to ``hidden_size // num_attention_heads`` for
    configs that omit it (Llama-3); Qwen3 states it explicitly (and its
    ``head_dim * num_heads`` may differ from ``hidden_size``)."""
    num_heads = int(hf_config.num_attention_heads)
    head_dim = int(getattr(hf_config, "head_dim", None)
                   or hf_config.hidden_size // num_heads)
    return {
        "lm_head_key": lm_head_key,
        "vocab_size": int(hf_config.vocab_size),
        "logit_scale": float(getattr(hf_config, "logit_scale", 1.0) or 1.0),
        "rms_norm_eps": float(getattr(hf_config, "rms_norm_eps", 1e-5) or 1e-5),
        "norm_weight_key": "model.norm.weight" if has_final_norm else None,
        "embed_weight_key": embed_weight_key,
        # Model dims for the backward's per-layer forward rematerialization.
        "hidden_size": int(hf_config.hidden_size),
        "num_hidden_layers": int(hf_config.num_hidden_layers),
        "num_attention_heads": num_heads,
        "num_key_value_heads": int(
            getattr(hf_config, "num_key_value_heads", num_heads)),
        "head_dim": head_dim,
        "intermediate_size": int(hf_config.intermediate_size),
        "rope_theta": rope_theta_of(hf_config),
        "rope_scaling": rope_scaling_of(hf_config),
        "lora_scaling": float(lora_scaling),
        # Optimizer hyperparameters (Phase 3 LoRA backward).
        "learning_rate": float(ft_cfg.learning_rate),
        "weight_decay": float(ft_cfg.weight_decay),
        "gamma": float(ft_cfg.gamma),
        "backward_fp32": bool(ft_cfg.backward_fp32),
        "backward_tf32": bool(getattr(ft_cfg, "backward_tf32", False)),
        "backward_attn_sdpa": bool(getattr(ft_cfg, "backward_attn_sdpa", False)),
        # CUDA-graph backward (Phase 5): bn_max/l_max bound the padded-attention
        # region; s_max comes from max_saved_finetuning_tokens (the same value
        # sizes the activation buffers, so graphs and the pool are aligned).
        "backward_cuda_graph": bool(ft_cfg.backward_cuda_graph),
        "backward_run_ahead_boundaries":
            int(getattr(ft_cfg, "backward_run_ahead_boundaries", 0)),
        "backward_cuda_graph_attn_bn_max":
            int(ft_cfg.backward_cuda_graph_attn_bn_max),
        "backward_cuda_graph_attn_l_max":
            int(ft_cfg.backward_cuda_graph_attn_l_max),
        "max_saved_finetuning_tokens":
            int(ft_cfg.max_saved_finetuning_tokens),
        # [long context] graphs captured at this many rows (0 = the buffer).
        "backward_cuda_graph_max_tokens":
            int(getattr(ft_cfg, "backward_cuda_graph_max_tokens", 0)),
        # Forward-side activation saves the backward may short-circuit on:
        # post-RoPE q/k/v per layer (``attn_qh/kh/vh``) and the attention
        # context (``attn_ctx``). The child re-checks ``save_attn_qkv`` against
        # its family (see ``effective_save_attn_qkv``).
        "save_attn_qkv": bool(ft_cfg.save_attn_qkv),
        "save_attn_ctx": bool(ft_cfg.save_attn_ctx),
        "save_resid_mid": bool(ft_cfg.save_resid_mid),
        # [DeltaServe] Phase 7: TP shard geometry. The backward divides the
        # head counts / intermediate size by tp_size to slice its LOCAL shards,
        # and uses tp_rank to slice the (full, disk-loaded) FT-adapter master
        # into this rank's LoRA shard. tp_size=1 → local == full.
        # Adapter export (PEFT dir): where + how often, and the source adapter
        # whose adapter_config.json (r, alpha, targets) the export copies.
        "adapter_save_dir": getattr(ft_cfg, "adapter_save_dir", None),
        "num_epochs": int(ft_cfg.num_epochs),
        "adapter_save_every": int(getattr(ft_cfg, "adapter_save_every", 100) or 0),
        "finetuning_lora_path": ft_cfg.finetuning_lora_path,
        "tp_size": int(tp_size),
        "tp_rank": int(tp_rank),
        # Rendezvous port for the backward-only NCCL group (the backward
        # children are NOT in vLLM's inference group). Override via env if
        # the default clashes on the box.
        "backward_nccl_port": int(
            os.environ.get("DSERVE_BACKWARD_NCCL_PORT", "29677")),
    }
