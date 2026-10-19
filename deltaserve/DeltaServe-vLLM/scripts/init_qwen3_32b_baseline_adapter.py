"""Initialize PEFT QKV LoRA rank 8 / alpha 32 without loading base weights.

Run with dserve-vllm/.venv/bin/python. This is the ordinary zero-B LoRA
initialization used for the four-H200 baseline comparison, not a trained adapter.
"""

import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    root = Path(__file__).resolve().parents[1]
    model = Path("/mnt/weka/home/jianshu.she/scratch/Qwen3-32B")
    cfg = json.loads((model / "config.json").read_text())
    out = root / "adapters/qwen3-32b-qkv-r8-initial"
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "base_model_name_or_path": str(model), "bias": "none",
        "fan_in_fan_out": False, "inference_mode": True,
        "init_lora_weights": True, "lora_alpha": 32, "lora_dropout": 0.0,
        "modules_to_save": None, "peft_type": "LORA", "r": 8,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
        "task_type": "CAUSAL_LM", "use_rslora": False,
    }
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    torch.manual_seed(0)
    weights = {}
    for i in range(cfg["num_hidden_layers"]):
        for proj, heads in (("q", cfg["num_attention_heads"]),
                            ("k", cfg["num_key_value_heads"]),
                            ("v", cfg["num_key_value_heads"])):
            prefix = f"base_model.model.model.layers.{i}.self_attn.{proj}_proj.lora_"
            a = torch.empty(8, cfg["hidden_size"], dtype=torch.float32)
            torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            weights[prefix + "A.weight"] = a
            weights[prefix + "B.weight"] = torch.zeros(heads * cfg["head_dim"], 8)
    save_file(weights, str(out / "adapter_model.safetensors"))
    print(f"Saved {len(weights)} tensors to {out}")


if __name__ == "__main__":
    main()
