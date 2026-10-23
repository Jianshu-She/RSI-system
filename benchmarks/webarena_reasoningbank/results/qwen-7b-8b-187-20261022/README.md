# ReasoningBank: Qwen2.5-7B and Qwen3-8B on WebArena-Shopping

## Final results

All four lanes completed the same 187 shopping task IDs in numeric order. Success is official WebArena `cum_reward == 1`, not the agent self-evaluation. Errors remain in the denominator. These are single runs.

| Model | Method | Success / total | Success rate | Average steps | Terminal errors |
|---|---|---:|---:|---:|---:|
| qwen25-7b | Base | 28/187 | 14.97% | 19.82 | 10 |
| qwen25-7b | ReasoningBank | 22/187 | 11.76% | 20.30 | 17 |
| qwen3-8b | Base | 41/187 | 21.93% | 8.29 | 0 |
| qwen3-8b | ReasoningBank | 34/187 | 18.18% | 15.11 | 3 |

RB decreased success by 3.21 percentage points on 7B and 3.74 points on 8B. Both comparisons cover identical task sets; RB also used more steps. The results do not establish a general model-size effect.

## Paper and repository comparisons

| Source | Model | Tasks | Base SR | RB SR | Difference |
|---|---|---:|---:|---:|---:|
| Paper-reported | Gemini-2.5-Flash | 187 | 39.0% | 49.7% | +10.7 pp |
| Paper-reported | Gemini-2.5-Pro | 187 | 45.5% | 51.9% | +6.4 pp |
| Paper-reported | Gemma-3-12B | 187 | 17.1% | 24.1% | +7.0 pp |
| Earlier repository reproduction | Qwen3-32B | 187 | 32.6% | 34.2% | +1.6 pp |
| This reproduction | Qwen2.5-7B-Instruct | 187 | 14.97% | 11.76% | -3.21 pp |
| This reproduction | Qwen3-8B | 187 | 21.93% | 18.18% | -3.74 pp |

Paper figures are recorded in [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md); earlier reproduction figures are in [RESULTS.md](../RESULTS.md). Backbone, embedding, grader serving, and environment-build differences prevent a controlled cross-model comparison. The earlier 86-task subset is a separate evaluation scope and is not mixed here.

## Experimental configuration

| Setting | Value |
|---|---|
| Backbone | Qwen/Qwen2.5-7B-Instruct or Qwen/Qwen3-8B |
| Agent / self-evaluation / induction | Same backbone within each run |
| Agent sampling | Temperature 0.7; maximum 30 browser steps |
| Reasoning | Generic-agent CoT enabled for both; native Qwen3 thinking parser for 8B |
| Generation budget | Requested maximum 65,536; effective maximum min(requested, 131,072 - actual templated input tokens - 256) |
| Context | 131,072 tokens; YaRN factor 4, original context 32,768 |
| Agent serving | vLLM 0.11, PyTorch 2.8 CUDA 12.8, Transformers 4.57.3; TP=1; prefix cache; max sequences 4 |
| Observation / browser | AX-tree; viewport 1500 x 1280; slow_mo 30 ms |
| Memory | Online ReasoningBank, original task order; independent bank and shopping site per lane |
| Retrieval | Qwen3-Embedding-8B; top-1 task memory retrieval |
| Official fuzzy-match grader | OpenRouter qwen/qwen3.5-397b-a17b; other tasks use their official string/URL/HTML evaluator |
| Self-evaluation / induction sampling | Temperature 0 / 1.0, respectively |
| Weight training | None |

The dynamic budget prevents an input-plus-reserved-output context rejection; it does not truncate input history. No experiment hyperparameters were changed in response to observed test failures.

## Recovery and deviations

An external-grader HTTP 402 interrupted the initial allocations. Sites were shut down cleanly and run files hashed before release. Following credit restoration and a successful grader authentication probe, the existing sites and banks were restored. Four uncommitted browser episodes were replayed: 7B Base 234, 7B RB 235, 8B Base 324, and 8B RB 160. Their saved actions were inspected as read-only interactions; original timestamped trajectories were retained. This was episode replay, not exact browser-step continuation. Previously committed episodes were skipped.

Cluster-side lifecycle supervision was subsequently installed for dependent 12-hour successor jobs and complete-task boundary stops. A transport retry wrapper was added for connection failures, timeouts, and transient HTTP errors without changing request content. These operational interventions and stochastic reruns limit strict reproducibility. The runs use a WebArena-Verified shopping environment rather than the original paper image.

## Observed failure examples and hypothesis

In 8B task 25, retrieved exact-keyword guidance narrowed a semantic review query into literal matching; RB omitted required reviewer names. In task 190, a retrieved page-specific element ID was repeatedly clicked unsuccessfully, while another memory referred to pending rather than completed orders. Base returned the required amount, $65.32. These trajectories support possible negative transfer from context-bound memories.

Other differences include navigation loops (48 and 161), premature abandonment (47), incorrect order status (96), and paraphrasing instead of extracting required sentences (165). No memory was injected in the replayed task 160, so its failure cannot be attributed to misleading retrieval. These observations motivate an applicability-judgment hypothesis, not a demonstrated causal or model-size conclusion.

## Audit

[results-audit.json](results-audit.json) records all 748 committed-task summaries, official rewards, step counts, terminal errors, summary/self-evaluation SHA256 hashes, and both memory-bank hashes. Each lane's committed IDs exactly matched its 187 configured shopping IDs; each RB bank contained exactly those 187 unique IDs. Raw trajectories and site roots remain in the experiment archive and are not included in this PR. Credentials are excluded.
