# 能放进论文的 DeltaServe 实验（索引）

本地根目录：`/mnt/weka/home/jianshu.she/.cache/rsi-push/RSI-system/deltaserve/results/`
GitHub：https://github.com/Jianshu-She/RSI-system/tree/main/deltaserve/results
文字说明：同目录 `RESULTS.md` 的对应章节。优化点汇总：`OPTIMIZATIONS.md`。

## 1. 真实 bursty 时间线（RESULTS §8，主实验，最推荐）
- 对比：DeltaServe / 停服务训练（seq）/ vLLM + torchtune，两段负载窗口（r1、r2）
- 结论：比 torchtune 的请求 p99 和 TTFT p99 低 6–7 倍，训练早完成 41–42%；比 seq 尾延迟低，且不 hold 任何用户
- 图：
  - /mnt/weka/home/jianshu.she/.cache/rsi-push/RSI-system/deltaserve/results/rsi_burst_2026-10-28/fig_mix_r1.png
  - /mnt/weka/home/jianshu.she/.cache/rsi-push/RSI-system/deltaserve/results/rsi_burst_2026-10-28/fig_mix_r2.png
- 汇总表：rsi_burst_2026-10-28/rsi_mix.md、rsi_mix.json
- 原始数据：rsi_burst_2026-10-28/{coguard_r1, coguard_r2, seq_r1, seq_r2, tt_r1, tt_r2}/

## 2. Poisson 混合流量 + torchtune 对比 + 优化消融（RESULTS §7 / §7.1）
- 结论：训练周转比 torchtune 快 22–42%，尾延迟约为它的一半；优化逐步累加，每轮训练等待 302 → 74 s
- 图：
  - /mnt/weka/home/jianshu.she/.cache/rsi-push/RSI-system/deltaserve/results/rsi_mix_2026-10-28/fig_mix_r1.png
  - /mnt/weka/home/jianshu.she/.cache/rsi-push/RSI-system/deltaserve/results/rsi_mix_2026-10-28/fig_mix_r2.png
  - 所有变体：rsi_mix_2026-10-28/fig_mix_all_r1.png、fig_mix_all_r2.png
- 汇总表：rsi_mix_2026-10-28/rsi_mix.md（co、cofix、coloose、cofast、coguard、cofastb、seq、seqfast、tt）
- 原始数据：rsi_mix_2026-10-28/<变体>_r1|r2/

## 3. 端到端 RSI 闭环准确率（RESULTS §1）
- 结论：Formula 68.0 → 75.0（p = 0.009）；复现 66.5 → 74.5（p = 0.007）
- 无图，数据：
  - e2e_2026-10-19/summary.json
  - e2e_2026-10-19_repro/summary.json
  - 新版重跑：e2e_fast_2026-10-28/、e2e_fastbwd_2026-10-28/
  - 每个目录的 bwd.csv：每个 backward cycle 的 loss 和吞吐

## 4. LoRA vs 技能放进 context（RESULTS §3）
- 内容：准确率、context 长度与 KV 开销、TTFT/TBT、GPU 利用率、prompt 泄露攻击
- 表：paper1_2026-10-20/paper_metrics.md、paper_metrics.json
- 原始数据：paper1_2026-10-20/leak/（泄露）、paper1_2026-10-20/eval/（准确率）
- 暂无图

## 每个子目录里的原始数据
- traffic.jsonl.gz：每个请求的 TTFT、TBT、端到端延迟、对错
- bwd.csv：每个 backward cycle 一行
- traffic.jsonl.ft.json：每轮训练的开始和完成时间
- config.yaml：完整配置
- driver.log：运行日志

## 不建议放进论文
- §5 / §6（rsi_poisson_2026-10-20/、rsi_tracelab_2026-10-22/）：旧调度器，训练周转输给 seq，已被 §7、§8 取代
- §4（agent1_2026-10-20/）：早期版本
- WebArena：torchtune 组崩溃，DeltaServe 组训练被饿住，暂无可用结果

注意：以上每组只跑了一次；§7.1 有一次重复，可用来估计误差。
