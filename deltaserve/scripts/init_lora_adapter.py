"""Write a zero-B PEFT LoRA adapter for a Qwen3 checkpoint without loading weights
(standard PEFT init: A kaiming-uniform, B = 0, so the adapter starts as a no-op).

    python deltaserve/scripts/init_lora_adapter.py --model $MODEL_DIR --out $DS_INIT_ADAPTER \
        --rank 16 --alpha 32 --targets q,k,v,o
"""
import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--targets", default="q,k,v,o", help="subset of q,k,v,o (the backward trains attention only)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = json.loads((Path(a.model) / "config.json").read_text())
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    dims = {"q": (cfg["hidden_size"], cfg["num_attention_heads"] * hd),
            "k": (cfg["hidden_size"], cfg["num_key_value_heads"] * hd),
            "v": (cfg["hidden_size"], cfg["num_key_value_heads"] * hd),
            "o": (cfg["num_attention_heads"] * hd, cfg["hidden_size"])}
    targets = [t.strip() for t in a.targets.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": a.model, "bias": "none", "fan_in_fan_out": False,
        "inference_mode": True, "init_lora_weights": True, "lora_alpha": a.alpha,
        "lora_dropout": 0.0, "modules_to_save": None, "peft_type": "LORA", "r": a.rank,
        "target_modules": [f"{t}_proj" for t in targets], "task_type": "CAUSAL_LM",
        "use_rslora": False}, indent=2) + "\n")
    torch.manual_seed(a.seed)
    w = {}
    for i in range(cfg["num_hidden_layers"]):
        for t in targets:
            fin, fout = dims[t]
            pre = f"base_model.model.model.layers.{i}.self_attn.{t}_proj.lora_"
            A = torch.empty(a.rank, fin)
            torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            w[pre + "A.weight"] = A
            w[pre + "B.weight"] = torch.zeros(fout, a.rank)
    save_file(w, str(out / "adapter_model.safetensors"))
    print(f"saved {len(w)} tensors to {out}")


if __name__ == "__main__":
    main()
