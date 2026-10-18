# RSI-system

Reproducible pipelines for **self-improving LLM agents with open models**: evaluate a memory / context-learning method
on an agent benchmark, turn the agent's own trajectories into training data, fine-tune with LoRA, and compare
"learning in context" against "learning in weights" on held-out tasks — all with local vLLM models (Qwen3-32B) on one
8×GPU node.

| Benchmark | Method reproduced | What is here | Results |
|---|---|---|---|
| [WebArena-Shopping](benchmarks/webarena_reasoningbank) (187 tasks, web agent) | [ReasoningBank](https://arxiv.org/abs/2509.25140) (memory of distilled strategies) | eval (no memory / ReasoningBank / cross-check judge), trajectory → SFT data, LoRA, eval on unseen templates | [RESULTS](benchmarks/webarena_reasoningbank/results/RESULTS.md) |
| [ACE](benchmarks/ace): Finance (FiNER, Formula) + AppWorld | [Agentic Context Engineering](https://arxiv.org/abs/2510.04618) (evolving playbook) | eval (base / ICL / Dynamic Cheatsheet / ACE offline+online ±GT), playbook-distillation SFT data, LoRA, eval | [RESULTS](benchmarks/ace/results/RESULTS.md) |

## Layout
```
rsi/finetune/            shared LoRA SFT trainer (1..N GPUs, torchrun) + the chat-JSONL data format all benchmarks emit
benchmarks/<name>/
  README.md              quickstart for that benchmark
  env.sh                 the only place with paths / ports / model endpoints
  setup/                 install.sh: clone upstream at a pinned commit + apply patches/, create venvs; data/images
  patches/               every change vs upstream as a git diff (upstream code is not vendored)
  scripts/ (+ jobs/)     serve models, run evaluations, SLURM job templates (one node each, resumable)
  datagen/               trajectories -> SFT JSONL (train split only)
  finetune/              benchmark-specific training jobs (calls rsi/finetune or a pinned copy)
  eval/ or tools/        scoring, paired significance tests, audits
  results/               RESULTS.md (numbers + caveats), raw summaries, experiment logs
```

## Typical workflow
1. **Set up** — `cd benchmarks/<name> && bash setup/install.sh` (set the big-artifact dir first: `WA_WORK=…` or the
   variables in that benchmark's `env.sh`).
2. **Evaluate** a base model and the method (memory / playbook) — see the benchmark README; every run is resumable and
   writes per-step traces.
3. **Generate data** — run a teacher (or the model itself) on the *train* split and build the SFT set with `datagen/`.
4. **Fine-tune** — `torchrun --nproc_per_node=4 rsi/finetune/train_lora.py --data sft.jsonl --out ckpt/x …`
   ([details](rsi/finetune/README.md)).
5. **Evaluate the adapter** on the held-out split (vLLM `--enable-lora`) and compare **paired** against base and the
   in-context method with the benchmark's report tool.

## Findings so far (Qwen3-32B)
- **ACE**: reproduces on Formula (offline +GT 86.5 vs paper 85.5); same direction but smaller gains on FiNER and AppWorld.
  Distilling ACE traces into a LoRA recovers part of the gain without any context (Formula 69 → 75, FiNER 62 → 68.5).
- **ReasoningBank**: not reproduced (+1.6 on 187 tasks, n.s., more steps). On unseen templates ReasoningBank is the best
  point estimate (+4.7, n.s.); LoRA v1 gave no gain (its final checkpoint overfit to a looping trace style); LoRA v2's run
  was invalidated by a site-search bug that is now fixed and guarded — it needs to be re-run.
- Single runs: differences below ~2 pts (ACE, temperature 0) / ~5 pts (WebArena, temperature 0.7) are within noise.

## Requirements
Linux, NVIDIA GPUs (tested on 8× H200), [uv](https://docs.astral.sh/uv/), vLLM 0.31, SLURM optional
(job templates use `--account/--qos/--partition` placeholders). WebArena sites run rootless with
[enroot](https://github.com/NVIDIA/enroot) — no docker or sudo needed.

Upstream code (google-research/reasoning-bank, ace-agent/ace, AppWorld, Dynamic Cheatsheet) keeps its own licenses;
this repo ships only patches and new code.
