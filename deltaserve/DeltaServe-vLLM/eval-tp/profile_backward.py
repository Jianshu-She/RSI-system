#!/usr/bin/env python
"""Profile one LoRA backward cycle at production shape on ONE GPU.

Builds the real Qwen3 backward service (``Qwen3BackwardService``) on random weights with
the per-rank shard of Qwen3-32B at TP=4 (hidden 5120, 16 q heads, 2 kv heads, head 128,
intermediate 6400, full 151936 vocab head) and ``--layers`` layers, feeds it a captured-
activation dict with every production save on (gate_up, q/k/v, ctx, resid_mid), and times
``process_backward`` on a 16K-token batch of mixed sample lengths. No TP collectives (one
rank's compute only). Prints wall time per cycle scaled to 64 layers and the top CUDA ops.

    CUDA_VISIBLE_DEVICES=0 python eval-tp/profile_backward.py [--layers 8] [--lens 2400x6,600x2]
        [--run-ahead 2] [--trace out.json]
"""
import argparse
import time

import torch

from vllm.deltaserve.bwd_services.qwen3 import Qwen3BackwardService


def build(layers, run_ahead, s_max, dev, dt, tf32=False, sdpa=False):
    D, Hq, Hkv, Hd, inter, vocab, r = 5120, 16, 2, 128, 6400, 151936, 16
    fam = Qwen3BackwardService.family
    q, kv = Hq * Hd, Hkv * Hd
    torch.manual_seed(0)
    g = torch.Generator(device=dev).manual_seed(0)

    def w(*s):
        return (torch.randn(*s, device=dev, generator=g) * s[-1] ** -0.5).to(dt)

    def norm(n):
        return (torch.rand(n, device=dev, generator=g) + 0.5).to(dt)

    shapes = {"qkv": (q + 2 * kv, D), "o": (D, q), "gate_up": (2 * inter, D), "down": (D, inter)}
    base = {"model.embed_tokens.weight": w(vocab, D), "model.norm.weight": norm(D),
            "lm_head.weight": w(vocab, D)}
    ft = {}
    for i in range(layers):
        p = f"model.layers.{i}."
        for key, rel in fam.layer_weights.items():
            base[p + rel] = w(*shapes[key]) if key in shapes else norm(D if key in ("in_ln", "post_ln") else Hd)
        pre = f"base_model.model.model.layers.{i}.self_attn."
        for proj, out, inp in [("q", q, D), ("k", kv, D), ("v", kv, D), ("o", D, q)]:
            ft[pre + f"{proj}_proj.lora_A.weight"] = (torch.randn(r, inp, device=dev) * 0.02).to(dt)
            ft[pre + f"{proj}_proj.lora_B.weight"] = torch.zeros(out, r, device=dev, dtype=dt)
    meta = dict(hidden_size=D, num_hidden_layers=layers, num_attention_heads=Hq,
                num_key_value_heads=Hkv, head_dim=Hd, intermediate_size=inter, rope_theta=1e6,
                rms_norm_eps=1e-6, lora_scaling=2.0, vocab_size=vocab, learning_rate=1e-4,
                weight_decay=0.0, gamma=1.0, backward_fp32=False, lm_head_key="lm_head.weight",
                norm_weight_key="model.norm.weight", embed_weight_key="model.embed_tokens.weight",
                save_attn_qkv=True, save_attn_ctx=True, save_resid_mid=True,
                max_saved_finetuning_tokens=s_max, backward_run_ahead_boundaries=run_ahead,
                backward_tf32=tf32, backward_attn_sdpa=sdpa)
    svc = Qwen3BackwardService(0)
    svc.shared = {"base": base, "ft": ft, "meta": meta}
    svc._build_state()
    svc._built = True
    acts = {
        "layer_in": [w(s_max, D) for _ in range(layers)],
        "mlp_gate_up": [w(s_max, 2 * inter) for _ in range(layers)],
        "attn_qh": [w(s_max, q) for _ in range(layers)],
        "attn_kh": [w(s_max, kv) for _ in range(layers)],
        "attn_vh": [w(s_max, kv) for _ in range(layers)],
        "attn_ctx": [w(s_max, q) for _ in range(layers)],
        "resid_mid": [w(s_max, D) for _ in range(layers)],
        "final_in": w(s_max, D), "final_hidden": w(s_max, D),
        "concat_input_ids": torch.randint(0, vocab, (s_max,), device=dev),
    }
    return svc, acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--lens", default="2400x6,600x2")
    ap.add_argument("--run-ahead", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--trace", default="")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--tf32", action="store_true", help="meta backward_tf32: fp32 matmuls (head, attention scores) on TF32")
    ap.add_argument("--sdpa", action="store_true", help="meta backward_attn_sdpa: fused SDPA attention backward")
    ap.add_argument("--no-profile", action="store_true")
    a = ap.parse_args()
    lens = [int(x.split("x")[0]) for x in a.lens.split(",") for _ in range(int(x.split("x")[1]))]
    n = sum(lens)
    dev, dt = "cuda", torch.bfloat16
    svc, acts = build(a.layers, a.run_ahead, n, dev, dt, tf32=a.tf32, sdpa=a.sdpa)   # the trainer sets TF32 from meta

    def cycle():
        svc.process_backward(acts, lens, n, 0)

    cycle()
    torch.cuda.synchronize()
    ts = []
    for _ in range(a.iters):
        t = time.time()
        cycle()
        torch.cuda.synchronize()
        ts.append(time.time() - t)
    per = min(ts)
    print(f"{n} tokens ({len(lens)} samples), {a.layers} layers, run-ahead {a.run_ahead}: "
          f"{per * 1e3:.0f} ms/cycle (tf32 {a.tf32}, sdpa {a.sdpa})")
    if a.no_profile:
        return
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        cycle()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=a.top, max_name_column_width=60))
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=a.top,
                                                            max_name_column_width=30, max_shapes_column_width=90))
    if a.trace:
        prof.export_chrome_trace(a.trace)


if __name__ == "__main__":
    main()
