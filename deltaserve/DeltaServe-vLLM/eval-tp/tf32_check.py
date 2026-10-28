# fp32 vs TF32 backward: loss and every LoRA grad, same weights/activations (production shape, 4 layers)
import sys, torch
sys.path.insert(0, "/mnt/weka/home/jianshu.she/deltaserve/DeltaServe-vLLM/eval-tp")
import profile_backward as P
lens = [2400] * 6 + [600] * 2; n = sum(lens)
out = {}
import os
MODE = os.environ.get("MODE", "tf32")   # tf32 | sdpa | both: the variant compared with strict fp32
for tf in (False, False, True):
    torch.backends.cuda.matmul.allow_tf32 = False
    svc, acts = P.build(4, 2, n, "cuda", torch.bfloat16, tf32=tf and MODE in ("tf32", "both"),
                        sdpa=tf and MODE in ("sdpa", "both"))
    loss, nv = svc.process_backward(acts, lens, n, 0)
    torch.cuda.synchronize()
    grads = {f"{i}.{p}.{k}": t.grad.detach().clone() for i, ld in svc.lora.items() for p, d in ld.items() for k, t in d.items()}
    out.setdefault(tf, []).append((float(loss), grads))
(l0, g0), (lr, gr) = out[False]; (l1, g1), = out[True]
print("fp32 rerun determinism: loss diff", abs(lr - l0), "max grad diff", max((gr[k] - g0[k]).abs().max().item() for k in g0))
print(f"[{MODE}] loss fp32 {l0:.6f} variant {l1:.6f} rel diff {abs(l1 - l0) / abs(l0):.2e}")
g0 = {k: v for k, v in g0.items() if k.endswith(".B")}
rel = sorted(((g1[k] - g0[k]).norm() / (g0[k].norm() + 1e-30)).item() for k in g0)
cos = sorted(torch.nn.functional.cosine_similarity(g1[k].flatten(), g0[k].flatten(), dim=0).item() for k in g0)
print(f"{len(rel)} LoRA grads: rel-L2 diff median {rel[len(rel)//2]:.2e} max {rel[-1]:.2e}; cosine min {cos[0]:.6f}")
