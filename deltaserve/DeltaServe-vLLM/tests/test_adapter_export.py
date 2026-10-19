"""FT adapter export (PEFT dir) gates — run with `python`, not pytest. CPU/gloo.

1. TP=1: ``export_adapter`` writes the fp32 masters under their original PEFT
   names and full shapes, plus an adapter_config.json, and refreshes latest/.
2. TP=2: after identical training, the shards gathered + merged by rank 0
   equal the TP=1 export (q/k/v B row shards, o A column shards).
3. ``adapter_save_every`` exports from inside process_backward.

    python tests/test_adapter_export.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file

TESTS = str(Path(__file__).resolve().parent)
sys.path.insert(0, TESTS)
import test_tp_trainer_graph_nccl as T  # noqa: E402

T.GEOM["qwen3"].update(hidden=48, Hq=8, Hkv=4, inter=128)
PORT = 29741
PASS = FAIL = 0


def cond(name, ok, detail=""):
    global PASS, FAIL
    PASS += bool(ok)
    FAIL += not ok
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


def train(svc, full, steps=3):
    starts = [0, T.SEQ_LENS[0]]
    for _ in range(steps):
        acts = T.remat_activations(svc, full["embed"], full["ids"], T.SEQ_LENS, starts)
        svc.process_backward(acts, T.SEQ_LENS, sum(T.SEQ_LENS), epoch=0)


def worker(rank, path, out):
    full = torch.load(path)
    meta = T.make_meta("qwen3", tp_size=2, tp_rank=rank, port=PORT, graph=False)
    svc = T.make_service("qwen3", T.shard_base(full["base"], rank, 2, "qwen3"),
                         full["ft"], meta, torch.device("cpu"))
    svc._src_adapter = None
    train(svc, full)
    svc.export_adapter(os.path.join(out, "tp2"))
    dist.destroy_process_group()


def main():
    full = T.build_full("qwen3", seed=11)
    with tempfile.TemporaryDirectory(prefix="adapter_export_") as tmp:
        # source adapter_config.json the export should copy
        src = os.path.join(tmp, "src")
        os.makedirs(src)
        json.dump({"peft_type": "LORA", "r": 4, "lora_alpha": 8,
                   "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
                  open(os.path.join(src, "adapter_config.json"), "w"))

        print("TP=1 export")
        meta = T.make_meta("qwen3", tp_size=1, tp_rank=0, port=0, graph=False)
        meta.update(adapter_save_dir=os.path.join(tmp, "periodic"), adapter_save_every=2,
                    finetuning_lora_path=src)
        ref = T.make_service("qwen3", full["base"], full["ft"], meta, torch.device("cpu"))
        train(ref, full)
        ref.export_adapter(os.path.join(tmp, "tp1"))
        sd = load_file(os.path.join(tmp, "tp1", "adapter_model.safetensors"))
        cond("same tensor names as the source adapter", set(sd) == set(full["ft"]))
        cond("full (unsharded) shapes", all(sd[k].shape == full["ft"][k].shape for k in sd))
        masters = {ref._lora_keys[k]: ref.lora[k[0]][k[1]][k[2]].detach() for k in ref._lora_keys}
        cond("values == fp32 masters", all(torch.equal(sd[k], masters[k]) for k in sd))
        cond("training moved the weights",
             any(not torch.equal(sd[k], full["ft"][k].float()) for k in sd))
        cfg = json.load(open(os.path.join(tmp, "tp1", "adapter_config.json")))
        cond("adapter_config copied (r, alpha, targets) + inference_mode",
             cfg["r"] == 4 and cfg["lora_alpha"] == 8 and cfg["inference_mode"] is True)
        cond("latest/ refreshed", os.path.exists(os.path.join(tmp, "latest", "adapter_model.safetensors")))
        cond("adapter_save_every=2 exported cycle2/ during training",
             os.path.exists(os.path.join(tmp, "periodic", "cycle2", "adapter_model.safetensors")))

        print("TP=2 export (gloo) == TP=1")
        path = os.path.join(tmp, "full.pt")
        torch.save(full, path)
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=worker, args=(r, path, tmp)) for r in range(2)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(900)
        cond("both ranks exited cleanly", all(p.exitcode == 0 for p in procs),
             str([p.exitcode for p in procs]))
        f2 = os.path.join(tmp, "tp2", "adapter_model.safetensors")
        if os.path.exists(f2):
            sd2 = load_file(f2)
            err = max((sd2[k] - sd[k]).abs().max().item() / (sd[k].abs().max().item() + 1e-12)
                      for k in sd)
            cond("merged TP=2 adapter == TP=1 adapter", set(sd2) == set(sd) and err < 1e-4,
                 f"max rel err {err:.1e}")
        else:
            cond("rank 0 wrote the TP=2 adapter", False)
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
