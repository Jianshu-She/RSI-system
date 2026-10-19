"""QKV-only LoRA: four-rank bucketed AdamW must match a TP=1 reference.

CPU/gloo test through the real Qwen3 trainer, including clipping and multiple
optimizer steps. Run with dserve-vllm/.venv/bin/python from the repository root.
"""

import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

TESTS = str(Path(__file__).resolve().parent)
sys.path.insert(0, TESTS)
import test_tp_trainer_graph_nccl as T  # noqa: E402
sys.path.insert(0, TESTS)

# Four KV heads allow four shards; Q width 64 differs from residual width 48.
T.GEOM["qwen3"].update(hidden=48, Hq=8, Hkv=4, inter=128)
PORT = 29724


def train(svc, full):
    losses = []
    starts = [0, T.SEQ_LENS[0]]
    for _ in range(T.STEPS):
        acts = T.remat_activations(svc, full["embed"], full["ids"], T.SEQ_LENS, starts)
        loss, _ = svc.process_backward(acts, T.SEQ_LENS, sum(T.SEQ_LENS), epoch=0)
        losses.append(loss)
    masters = {f"L{i}.{proj}{ab}": param.detach().clone()
               for i, layer in svc.lora.items() for proj, factors in layer.items()
               for ab, param in factors.items()}
    return {"losses": torch.tensor(losses), "masters": masters}


def worker(rank, path, out):
    full = torch.load(path)
    svc = T.make_service("qwen3", T.shard_base(full["base"], rank, 4, "qwen3"),
                         full["ft"], T.make_meta("qwen3", tp_size=4, tp_rank=rank,
                         port=PORT, graph=False), torch.device("cpu"))
    assert svc._bucket.num_groups == 2
    result = train(svc, full)
    torch.save(result, Path(out) / f"rank{rank}.pt")
    dist.destroy_process_group()


def main():
    full = T.build_full("qwen3", seed=37)
    full["ft"] = {k: v for k, v in full["ft"].items() if ".o_proj." not in k}
    ref = T.make_service("qwen3", full["base"], full["ft"],
                         T.make_meta("qwen3", tp_size=1, tp_rank=0, port=0,
                                     graph=False), torch.device("cpu"))
    expected = train(ref, full)
    checker = T.H.Checker(tol=1e-4)
    with tempfile.TemporaryDirectory(prefix="qkv_bucket_tp4_") as tmp:
        path = str(Path(tmp) / "full.pt")
        torch.save(full, path)
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=worker, args=(rank, path, tmp)) for rank in range(4)]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(120)
            assert proc.exitcode == 0, proc.exitcode
        ranks = [torch.load(Path(tmp) / f"rank{rank}.pt") for rank in range(4)]
        for rank, result in enumerate(ranks):
            checker.check(f"rank {rank} loss trajectory", result["losses"], expected["losses"])
        for key, param in expected["masters"].items():
            if key.endswith("A"):
                for rank, result in enumerate(ranks):
                    checker.check(f"rank {rank} {key}", result["masters"][key], param)
            else:
                checker.check(f"concat {key}", torch.cat([r["masters"][key] for r in ranks]), param)
        checker.finish()


if __name__ == "__main__":
    main()
