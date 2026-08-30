# StarVLA-Demo

StarVLA real-robot examples 的复现笔记、计划与证据归档。

## 项目目录

每个 real-robot example 一个**以项目命名的文件夹**，统一结构：

```
<项目名>/
  README.md     项目概述 + 复现状态
  plans/        计划 / 评估文档
  scripts/      复现自动化脚本（可选）
  evidence/     复现证据（数据契约、配置、loss、补丁）
```

| 项目 | 状态 |
| --- | --- |
| [RoboChallenge_Table30v2](RoboChallenge_Table30v2/) | ✅ P0–P4 全流程复现完成 |
| [UnitreeG1_WholeBody](UnitreeG1_WholeBody/) | 📋 计划阶段 |
| [EgoVLA](EgoVLA/) | 🟡 P1 零训练闭环大部分完成，P1d/e 阻塞于 MANO 许可证；P2 训练未执行 |

## 通用分析

[`Project_Analysis/`](Project_Analysis/) — StarVLA 代码库分析、examples 概览、资源评估等跨项目文档。

## 命名约定

- 项目文件夹名对应 `starVLA/examples/realRobots/` 下的 example 目录名（如 `RoboChallenge_Table30v2`、`UnitreeG1_WholeBody`）。
- 代码补丁 → fork `TangYishanE/starVLA`；checkpoint 等大文件 → Hugging Face；本仓库只放文档与小证据。
