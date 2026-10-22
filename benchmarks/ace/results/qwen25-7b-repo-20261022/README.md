# ACE Finance: Qwen2.5-7B-Instruct Reproduction

Completed on 22 October 2026. All four offline and four online results are complete. This report covers the repository-default protocol, not the archived five-epoch experiment.

## Results

All entries are test accuracy (%). Formula uses numerical-answer accuracy; FiNER uses per-label accuracy across 441 examples. FiNER's `correct` field counts fully correct examples and is not the numerator of its label accuracy.

| Dataset | Method | Original paper: DeepSeek-V3.1 | Repository: Qwen3-32B | This run: Qwen2.5-7B-Instruct |
|---|---|---:|---:|---:|
| Formula | Empty-playbook base | 67.5 | 68.9 | 55.13 |
| Formula | ICL | 67.0 | 82.0 | Not run |
| Formula | MIPROv2 | 69.5 | Not run | Not run |
| Formula | GEPA | 71.5 | Not run | Not run |
| Formula | ACE offline, GT | 85.5 | 86.5 | 62.50 |
| Formula | ACE offline, no GT | 83.0 | 81.0 | 59.00 |
| Formula | ACE online, GT | 76.5 | 75.5 | 61.00 |
| Formula | ACE online, no GT | 78.5 | 79.5 | 60.50 |
| FiNER | Empty-playbook base | 70.7 | 62.8 | 40.29 |
| FiNER | ICL | 72.3 | 70.4 | Not run |
| FiNER | MIPROv2 | 72.4 | Not run | Not run |
| FiNER | GEPA | 73.5 | Not run | Not run |
| FiNER | ACE offline, GT | 78.3 | 72.6 | 51.76 |
| FiNER | ACE offline, no GT | 71.1 | 65.8 | 35.09 |
| FiNER | ACE online, GT | 76.7 | 71.0 | 48.19 |
| FiNER | ACE online, no GT | 67.3 | 60.8 | 32.94 |

Base values for the local models average the four initial empty-playbook evaluations; these are not four independent seeds. The 7B Formula baselines were 54.5, 55.0, 56.5 and 54.5%; FiNER baselines were approximately 40.31, 40.42, 40.36 and 40.08%.

## Protocol and execution parameters

| Parameter | Original paper | Repository 32B run | This 7B run |
|---|---|---|---|
| Generator / Reflector / Curator | DeepSeek-V3.1 for all roles | Qwen3-32B for all roles | Qwen2.5-7B-Instruct for all roles |
| Native thinking | Non-thinking | Disabled | Non-reasoning model |
| Weight updates | None | None | None |
| Offline epochs | Maximum 5 | 1 | 1 |
| Maximum reflection rounds | 5 | 3 | 3 |
| Adaptation batch size | 1 | 1 | 1 |
| Formula train / validation / test | Counts not explicitly listed in inspected paper text | 500 / 300 / 200 | 500 / 300 / 200 |
| FiNER train / validation / test | Counts not explicitly listed in inspected paper text | 1000 / 500 / 441 | 1000 / 500 / 441 |
| Actual offline processing | Not disclosed per lane | Formula: 500 each; FiNER GT: 936, no GT: 834 | Formula: 500 each; FiNER: 1000 each |
| Offline playbook selection | Exact schedule not established from inspected paper text | Best full-validation result | Best full-validation result |
| Validation interval | Not established | 100 steps | 100 steps |
| Online evaluation | Sequential prediction before adaptation; same shuffled test sequence across methods | 15-example windows | 15-example windows |
| Order used locally | — | Released file order | Same released file order |
| Curator frequency | Delta constructed from each example | Every step | Every step |
| Temperature | Not established from inspected paper text | 0 | 0 |
| Requested maximum output tokens | Not established | 4096 | 4096 |
| Context window | Not established | 131072, YaRN x4 | 131072, YaRN x4 |
| Playbook budget | Not established | 80000-token prompt hint | 80000-token prompt hint |
| Bullet analyzer / deduplication | Grow-and-refine described | Disabled by default | Disabled by default |
| Validation/test workers | Not established | Repository default | 20 |
| Serving stack | DeepSeek service | vLLM 0.31 | vLLM 0.11; torch 2.8 CUDA 12.8; transformers 4.57.3 |
| Repetitions | Not established | One run per setting | One run per setting |
| Checkpoints | Not established | Native checkpoints / recovery | Native initial snapshots; atomic per-step offline recovery and complete online-window checkpoints |

GT means ground-truth feedback is available during adaptation. It does not give the Generator the answer before the prediction being scored. Offline validation uses the full validation split, never the test split, to select the best playbook. Online scores retain the cumulative predictions from the test stream.

## Interpretation and limitations

- The 7B model benefits from GT feedback on FiNER: offline GT exceeds no GT by 16.67 percentage points; online GT exceeds no GT by 15.25 points. No-GT FiNER adaptation falls below its approximately 40.29% empty-playbook baseline.
- Formula improves over the 7B empty-playbook baseline under all four ACE settings. The online GT/no-GT difference is only 0.50 points; single runs do not support a significance claim.
- These are not model-size-only comparisons: model families differ, and the original paper used up to five epochs/reflection rounds while both local reproductions use one epoch/three rounds.
- The old 32B FiNER offline runs stopped at 936 and 834 steps after context overflow. Their reported results use earlier best-validation playbooks, not completed 1000-step runs.
- The paper describes sample-wise online adaptation; the released implementation used here evaluates 15-example windows. This implementation difference is disclosed rather than treated as identical to the paper description.
- The playbook budget is a prompt hint, not an enforced token limit. The ReasoningBank dynamic-token-budget change is not part of this ACE experiment.
- Node transitions retained original indices, historical validation records, best playbooks, cumulative online counters and baseline provenance. Only uncommitted windows may replay; they are not counted twice. This preserves accounting but does not guarantee byte-identical generations after interruption.
- The old five-epoch 7B results remain archived and are excluded from this report.

## Verification and provenance

Upstream ACE reference: `82709de`. Result paths in the audit are relative to `experiments/ciai-qwen25-7b/results/ace-repo-20261021/finance`. Raw traces and model weights are not included in this PR. Every final result was checked for the expected test total (200 or 441). All eight final result hashes and exact source paths are recorded in [the result audit](results-audit.json).

The 331267-to-333683 transition retained a complete 270-example FiNER no-GT online checkpoint. The resumed run first committed 285 examples and eventually completed all 441. The other three completed online lanes explicitly skipped inference. Final-state manifests include online snapshots, SHA256 values, validation history and script/configuration/log/result hashes.

Original paper: [Agentic Context Engineering, Table 2 and experimental setup](https://arxiv.org/html/2510.04618v1). Prior 32B results and caveats: [repository RESULTS.md](../RESULTS.md).

## Exact completed results

| Dataset | Mode | Feedback | Accuracy | Test examples | SHA256 |
|---|---|---|---:|---:|---|
| finer | offline | gt | 51.75736961% | 441 | `dc1b2af05a76fd452b5ecf7ab87395dbc8529c9c5ff4383cd3e6393293f34939` |
| finer | offline | nogt | 35.09070295% | 441 | `dfcddec5b719963083550791809c8e154ab443c0954f83f49882c4308ac249d8` |
| finer | online | gt | 48.18594104% | 441 | `bbdb89755ef86b5365e9f82751d331b688c7e4e475a0a3762f916ee89e6538b1` |
| finer | online | nogt | 32.93650794% | 441 | `c91c5bccb13b63222b71506c1971ec1c19d073416bf195015840f9bef3f627f2` |
| formula | offline | gt | 62.50000000% | 200 | `dd0881e8fa567162301802d0f8e4e93bf52c22231f41ea4ec6e5cf6959e20bdd` |
| formula | offline | nogt | 59.00000000% | 200 | `caccdcd95f3d4c6f21440f7a87ffcbfaf133bbba4e5eb91cd7ece7fdec9955e4` |
| formula | online | gt | 61.00000000% | 200 | `9422ec9c876a3af53decee52beb3ce0f711a35db76e6b7f6af2406dc52746a44` |
| formula | online | nogt | 60.50000000% | 200 | `1bba8182f84cfb61e6b0a4d30d157308fce97d5f5ce12a7cf5798eb521dd8e70` |
