# RoboChallenge Table30v2

StarVLA 在 RoboChallenge Table30v2 上的复现（单任务 `shred_paper`，UR5，QwenOFT）。

## 状态：✅ 全流程复现完成（P0–P4）

- **P1 数据**：8.9GB raw，1016 episodes / 971,436 frames，state `[7]` / action `[8]`
- **P2 训练**：QwenOFT（Qwen3.5-0.8B）100-step 全参数微调，loss 0.56→0.31，checkpoint 2.24GB
- **P3 self-test**：3× `(8,8)`，avg latency 1.95s
- **P4 mock 协议**：28 次迭代 / 62s 闭环

## 目录

- `plans/` — 复现计划、服务器就绪评估
- `evidence/` — 数据契约、训练配置、`dataset_statistics.json`、loss 轨迹、mock 补丁（详见 [`evidence/results.md`](evidence/results.md)）

## 产物位置

- 代码补丁：fork [`TangYishanE/starVLA`](https://github.com/TangYishanE/starVLA) → 分支 `repro/robochallenge-table30v2`
- checkpoint：HF [`TangYishan/starvla-robochallenge-table30v2-shred-paper-100step`](https://huggingface.co/TangYishan/starvla-robochallenge-table30v2-shred-paper-100step)（私有）

## 边界

跨任务协议验证（训练 `shred_paper`，mock 回放官方 UR5 记录 `arrange_fruits`）；未做线上 submission、未使用实体 UR5。
