# Paper metrics

## 1. Accuracy: {base, LoRA (rsi)} × {no skills, skills in context}

| bench | n | base | base + skills | LoRA | LoRA + skills |
|---|---|---|---|---|---|
| formula | 200 | 68.5 | 86.0 | 75.0 | 85.5 |
| finer | 441 | 61.8 | 57.2 | 61.9 | 56.1 |
| gsm8k | 1319 | 95.5 | 94.8 | 94.9 | 94.2 |
| mmlu | 1000 | 76.6 | 77.1 | 76.4 | 77.7 |

| bench | comparison | Δ (pts) | +/− | sign-test p |
|---|---|---|---|---|
| formula | base_skills vs base_empty | +17.5 | +36/−1 | 0.0000 |
| formula | rsi_empty vs base_empty | +6.5 | +22/−9 | 0.0294 |
| formula | rsi_skills vs base_skills | -0.5 | +2/−3 | 1.0000 |
| formula | rsi_skills vs rsi_empty | +10.5 | +22/−1 | 0.0000 |
| formula | rsi_empty vs base_skills | -11.0 | +1/−23 | 0.0000 |
| finer | base_skills vs base_empty | -4.6 | +19/−49 | 0.0004 |
| finer | rsi_empty vs base_empty | +0.1 | +22/−31 | 0.2717 |
| finer | rsi_skills vs base_skills | -1.1 | +27/−20 | 0.3817 |
| finer | rsi_skills vs rsi_empty | -5.8 | +25/−39 | 0.1034 |
| finer | rsi_empty vs base_skills | +4.7 | +47/−26 | 0.0186 |
| gsm8k | base_skills vs base_empty | -0.7 | +18/−27 | 0.2327 |
| gsm8k | rsi_empty vs base_empty | -0.5 | +12/−19 | 0.2810 |
| gsm8k | rsi_skills vs base_skills | -0.5 | +16/−23 | 0.3368 |
| gsm8k | rsi_skills vs rsi_empty | -0.7 | +31/−40 | 0.3425 |
| gsm8k | rsi_empty vs base_skills | +0.2 | +28/−26 | 0.8919 |
| mmlu | base_skills vs base_empty | +0.5 | +60/−55 | 0.7093 |
| mmlu | rsi_empty vs base_empty | -0.2 | +5/−7 | 0.7744 |
| mmlu | rsi_skills vs base_skills | +0.6 | +15/−9 | 0.3075 |
| mmlu | rsi_skills vs rsi_empty | +1.3 | +60/−47 | 0.2459 |
| mmlu | rsi_empty vs base_skills | -0.7 | +54/−61 | 0.5760 |

## 2. Serving cost: skills in context vs LoRA (stock vLLM 0.31, TP=4) — prefix cache on; the 4-rps rows repeat the 1-rps prompts (full-prompt cache hits)

KV pool: 1784768 tokens (plain), 1779312 (LoRA server); KV per token 256 KiB

| condition @ rps | prompt tok | out tok | KV/request (no sharing) | TTFT p50 / p99 (ms) | TBT p50 / p99 (ms) | E2E p50 / p99 (s) | KV usage mean / peak | running peak | waiting peak | prefix hit | out tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base@1 | 390 | 210 | 0.10 GiB | 43 / 62 | 7.0 / 9.1 | 1.40 / 5.41 | 0.0 / 0.1 % | 6 | 0 | 54.0 % | 201 |
| base@4 | 390 | 208 | 0.10 GiB | 26 / 32 | 7.1 / 8.5 | 1.38 / 5.78 | 0.1 / 0.3 % | 14 | 0 | 97.9 % | 775 |
| lora@1 | 390 | 259 | 0.10 GiB | 110 / 503 | 9.6 / 19.8 | 2.53 / 8.45 | 0.1 / 0.2 % | 8 | 0 | 54.0 % | 246 |
| lora@4 | 390 | 256 | 0.10 GiB | 35 / 86 | 9.8 / 13.7 | 2.40 / 6.84 | 0.2 / 0.4 % | 19 | 0 | 97.9 % | 897 |
| lora_skills@1 | 30136 | 418 | 7.36 GiB | 171 / 280 | 11.2 / 96.0 | 3.57 / 34.48 | 1.8 / 2.2 % | 13 | 0 | 98.9 % | 351 |
| lora_skills@4 | 30136 | 429 | 7.36 GiB | 127 / 227 | 18.3 / 44.5 | 5.68 / 48.58 | 2.3 / 2.8 % | 39 | 0 | 100.0 % | 936 |
| merged@1 | 390 | 253 | 0.10 GiB | 41 / 66 | 6.7 / 9.2 | 1.65 / 5.24 | 0.0 / 0.1 % | 5 | 0 | 53.7 % | 241 |
| merged_skills@1 | 30136 | 445 | 7.36 GiB | 106 / 159 | 7.9 / 13.5 | 2.33 / 23.33 | 1.8 / 2.1 % | 10 | 0 | 98.9 % | 400 |
| skills@1 | 30136 | 344 | 7.36 GiB | 109 / 159 | 8.1 / 15.4 | 2.29 / 23.83 | 1.6 / 2.1 % | 9 | 0 | 99.4 % | 304 |
| skills@4 | 30136 | 356 | 7.36 GiB | 106 / 182 | 11.8 / 30.1 | 3.49 / 31.19 | 2.0 / 2.3 % | 26 | 0 | 100.0 % | 893 |

## 2. Serving cost: skills in context vs LoRA (stock vLLM 0.31, TP=4) — prefix cache on, 4 rps, formula TRAIN split: every prompt new to the server

KV pool: 1784608 tokens (plain), 1778976 (LoRA server); KV per token 256 KiB

| condition @ rps | prompt tok | out tok | KV/request (no sharing) | TTFT p50 / p99 (ms) | TBT p50 / p99 (ms) | E2E p50 / p99 (s) | KV usage mean / peak | running peak | waiting peak | prefix hit | out tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base@4 | 391 | 206 | 0.10 GiB | 42 / 68 | 6.8 / 24.6 | 1.40 / 5.41 | 0.1 / 0.2 % | 12 | 0 | 54.2 % | 781 |
| lora@4 | 391 | 253 | 0.10 GiB | 109 / 236 | 9.5 / 85.1 | 3.11 / 9.07 | 0.2 / 0.4 % | 23 | 0 | 54.1 % | 954 |
| lora_skills@4 | 30137 | 467 | 7.36 GiB | 233 / 2950 | 27.7 / 97.5 | 10.10 / 91.89 | 3.5 / 4.7 % | 64 | 11 | 99.4 % | 1351 |
| merged@4 | 391 | 251 | 0.10 GiB | 42 / 85 | 6.9 / 24.7 | 1.77 / 5.08 | 0.1 / 0.3 % | 15 | 0 | 54.1 % | 951 |
| merged_skills@4 | 30137 | 441 | 7.36 GiB | 143 / 834 | 22.8 / 51.0 | 6.34 / 65.08 | 2.9 / 3.8 % | 53 | 1 | 99.2 % | 1340 |
| skills@4 | 30137 | 337 | 7.36 GiB | 119 / 861 | 13.4 / 46.8 | 4.06 / 36.96 | 2.2 / 2.7 % | 33 | 1 | 99.2 % | 1104 |

## 2. Serving cost: skills in context vs LoRA (stock vLLM 0.31, TP=4) — prefix cache OFF: skills not shared between requests (per-user skills)

KV pool: 1784608 tokens (plain), None (LoRA server); KV per token 256 KiB

| condition @ rps | prompt tok | out tok | KV/request (no sharing) | TTFT p50 / p99 (ms) | TBT p50 / p99 (ms) | E2E p50 / p99 (s) | KV usage mean / peak | running peak | waiting peak | prefix hit | out tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base@1 | 390 | 205 | 0.10 GiB | 38 / 55 | 6.7 / 8.5 | 1.32 / 5.08 | 0.0 / 0.2 % | 6 | 0 | – % | 196 |
| skills@1 | 30136 | 345 | 7.36 GiB | 33962 / 99236 | 318.1 / 488.2 | 117.46 / 224.15 | 76.2 / 99.6 % | 59 | 63 | – % | 200 |

## 3. DeltaServe co-serving: same traffic, inference only (A) vs training in the gaps (C)

| phase | n | TTFT p50 / p99 (ms) | TTFT < 0.3 s | TBT p50 / p99 (ms) | E2E p50 / p99 (s) | SM active | GPU util (nvidia-smi) | power (W/GPU) |
|---|---|---|---|---|---|---|---|---|
| A | 200 | 35 / 51 | 99.5 % | 8.7 / 9.8 | 1.72 / 6.45 | 19.2 % | 59 % | 274 |
| C | 116 | 81 / 268 | 99.1 % | 8.3 / 25.8 | 2.06 / 5.17 | 56.0 % | 96 % | 493 |

Training in C: 34 cycles, 526208 tokens in 281 s (1873 FT tok/s) from 427 self-generated rows (427/500 correct traces).

## 4. Prompt-extraction leakage (12 attacks, union over attacks)

| deployment | bullet ids leaked | bullets reproduced verbatim |
|---|---|---|
| skills | 282/282 (100.0 %) | 149/282 (52.8 %) |
| base | 1/282 (0.4 %) | 0/282 (0.0 %) |
| lora | 1/282 (0.4 %) | 7/282 (2.5 %) |
