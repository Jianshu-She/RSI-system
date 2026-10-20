# agent-1: sequential vs co-serving (TraceLab agent workload, Qwen3-32B TP=4)

| | makespan (both jobs) | FT job done at | agent task completion p50 / p90 / p99 (s) |
|---|---|---|---|
| sequential: inference, then FT | 2386 s | 2386 s | 38.9 / 272.3 / 905.2 |
| sequential: FT, then inference | 2386 s | 255 s | 42.3 / 354.1 / 922.0 |
| co-serving | 2161 s | 1612 s | 46.6 / 306.2 / 936.1 |

co / sequential makespan = 0.906

| | inf only | co-serving |
|---|---|---|
| kv_tokens | 1755680 | 561936 |
| rounds | 900 | 900 |
| errors | 0 | 0 |
| compactions | 15 | 15 |
| replay_s | 2130.259731054306 | 2160.364844083786 |
| task_n | 83 | 83 |
| task_time | [38.928, 272.291, 905.246] | [46.62, 306.239, 936.11] |
| task_time_mean | 108.25 | 116.88 |
| task_llm | [20.142, 106.532, 181.166] | [23.875, 130.211, 214.034] |
| round_e2e | [1.652, 9.378, 32.996] | [2.003, 11.406, 40.238] |
| ttft | [0.152, 0.328, 0.959] | [0.179, 0.376, 1.13] |
| tbt_mean | [0.008, 0.009, 0.016] | [0.01, 0.011, 0.023] |
| tbt_p99_round | [0.014, 0.063, 0.114] | [0.019, 0.092, 0.133] |
| sm_active | 33.4 | 36.4 |

| FT | alone | co-serving |
|---|---|---|
| ft_s | 255.32944226264954 | 1612.4649348258972 |
| ft_tokens | 1052416 | 1052416 |
| ft_cycles | 68 | 68 |
| ft_tok_s | 4121.8 | 652.7 |
| ft_loss_first_last | [0.5227110314045607, 0.05364223624738096] | [0.5227110314045607, 0.052586122513964315] |
