"""Merge a PEFT LoRA (as exported by DeltaServe) into a full checkpoint: W += (alpha / r) * B @ A, in fp32, saved bf16.

The result is the "customized model": one set of weights, served by plain vLLM with no LoRA kernels, so its
latency is the base model's. Non-weight files (config with YaRN, tokenizer, ...) are copied from the base dir.

  python merge_lora.py <base_model_dir> <adapter_dir> <out_dir>
"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    base, adapter, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    cfg = json.load(open(f"{adapter}/adapter_config.json"))
    scale = cfg["lora_alpha"] / cfg["r"]
    ad = safe_open(f"{adapter}/adapter_model.safetensors", "pt")
    lora = {}
    for k in ad.keys():  # base_model.model.<module>.lora_{A,B}.weight -> <module>.weight
        mod, ab = k[len("base_model.model."):].rsplit(".lora_", 1)
        lora.setdefault(mod + ".weight", {})[ab[0]] = ad.get_tensor(k).float()
    done = 0
    for fn in sorted(os.listdir(base)):
        src = os.path.join(base, fn)
        if not fn.endswith(".safetensors"):
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(out, fn))
            continue
        tensors = {}
        with safe_open(src, "pt") as f:
            meta = f.metadata()
            for k in f.keys():
                t = f.get_tensor(k)
                if k in lora:
                    d = lora[k]
                    t = (t.float() + scale * (d["B"] @ d["A"])).to(t.dtype)
                    done += 1
                tensors[k] = t
        save_file(tensors, os.path.join(out, fn), metadata=meta or {"format": "pt"})
        print(f"{fn}: {len(tensors)} tensors", flush=True)
    assert done == len(lora), f"merged {done} of {len(lora)} LoRA modules"
    print(f"merged {done} modules (scale {scale}) -> {out}")


if __name__ == "__main__":
    main()
