# StarVLA-Demo

StarVLA real-robot demo 复现笔记与证据归档。

## 目录结构

- `Project_Analysis/` — 复现计划、资源评估、代码库分析（markdown 笔记）
- `evidence/robochallenge/` — RoboChallenge Table30v2 `shred_paper` 单任务复现证据（配置、数据契约、loss、mock 补丁）

## 复现状态（2026-08-31）

RoboChallenge Table30v2 `shred_paper`（UR5）单任务全流程复现完成（P0–P4）：

| 阶段 | 结果 |
| --- | --- |
| P1 数据 | 8.9GB raw，1016 episodes / 971436 frames，state `[7]` / action `[8]`，双相机 |
| P2 训练 | QwenOFT（Qwen3.5-0.8B）100-step 全参数微调，loss 0.56→0.31，checkpoint 2.24GB |
| P3 self-test | 3× `(8,8)` 动作，avg latency 1.95s |
| P4 mock 协议 | 28 次迭代 / 62s 连续闭环 |

代码补丁见 fork：[`TangYishanE/starVLA`](https://github.com/TangYishanE/starVLA) 分支 `repro/robochallenge-table30v2`。

## 说明（边界）

- 本次为**跨任务协议验证**：checkpoint 训练的是 `shred_paper`，但 mock 回放用官方仓库自带的唯一 UR5 记录 `arrange_fruits`（仓库无 shred_paper 回放数据）。
- 动作输出接近动作均值（模型仅训 100 step，属冒烟验证），**不具任务语义，不证明会撕纸**。
- 未做线上 submission、未使用实体 UR5。
