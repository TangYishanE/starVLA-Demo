# StarVLA 官方 Example 复现时间与资源占用评估

评估日期：2026-08-30  
目标机器：单张 NVIDIA RTX 5090 32 GiB、4 CPU、64 GiB RAM、100 GB 存储配额  
评估目标：在没有真实机械臂的条件下，形成可向老师汇报的 StarVLA real-demo 前置完整流程复现。

## 1. 结论摘要

- 建议的报告型核心组合需要约 **50–140 GPU 小时**、**12–22 个工程人日**；若按单人、单卡串行执行，计划 **3–5 周**较稳妥。
- 若把核心 benchmark 扩大到官方完整评测规模，单策略合计约 **330–820 GPU 小时**，即单卡纯运行约 **14–34 天**；加上环境适配和失败重跑，日历时间约 **5–9 周**。
- 当前 **100 GB 存储配额不够**。报告型核心组合建议准备 **250–600 GB 可用工作盘**；保留多个完整数据集、模型和视频时建议 **1–3 TB**。
- Qwen3-VL-4B 全参数微调在当前实现中于 Adam 状态初始化时达到约 32,116 MiB 并 OOM。延长运行时间无法解决容量问题；单卡应使用官方 checkpoint、冻结主干、LoRA 或更小 backbone。
- 优先级应是 RoboChallenge 前硬件闭环、LIBERO-plus 鲁棒性、SimplerEnv 跨 embodiment、RoboDojo/RoboTwin 双臂、多视角和动态场景，而不是为每个 benchmark 都执行全量训练。

## 2. 已有实测校准点

| 校准项 | 当前结果 | 对估算的作用 |
|---|---:|---|
| LIBERO 官方 checkpoint | 392/400，98.0% | 证明推理、双相机、8-step chunk、策略服务和仿真闭环可用 |
| LIBERO 400 episodes | 约 3 小时的已有规划量级 | 折合约 27 秒/episode，用作普通操作任务 rollout 基准 |
| LIBERO 2,000 episodes | 约 14–15 小时 | 与 400 episodes 基准近似线性 |
| 冻结 VLM 的 OFT 30K | 1:16:39，6.52 steps/s | 相近 action-head-only 训练的乐观吞吐基准 |
| transition 训练 500 steps | 4:37，1.80 steps/s | LoRA/额外损失与采样逻辑下的保守吞吐基准 |
| OFT 推理峰值 | 10,642.9 MiB | 单独 policy server 的低显存基线 |
| GR00T 推理峰值 | 11,400.0 MiB | 双臂/扩散头的较低显存基线 |
| PI_v3 推理峰值 | 16,198.7 MiB | 与 GPU 仿真同卡时需要重点做容量探针 |
| PI 推理峰值 | 24,096.6 MiB | 与重型 GPU 仿真同卡风险很高 |
| 完整 checkpoint | 约 9.14 GB/个 | 100 GB 配额下只能短期保留少量版本 |
| compact checkpoint | 约 262 MB/个 | 适合大量 smoke/消融实验归档 |
| 当前项目逻辑占用 | 约 58 GB | 100 GB 配额实际只剩约 42 GB，不足以展开新 benchmark |

## 3. 建议范围与官方全量范围

“建议范围”用于形成报告证据链；“官方全量”表示尽量跑完官方 README 描述的任务或评测规模。GPU 小时是单张 RTX 5090 串行规划值，不包含下载时间；工程人日包括安装、适配、故障定位、结果整理和一次合理重跑。

| Example | 建议复现范围 | 工程人日 | 建议 GPU 小时 | 官方全量 GPU 小时 | 工作盘建议 | 峰值显存规划 | 置信度 |
|---|---|---:|---:|---:|---:|---:|---|
| RoboChallenge Table30v2 | 单任务数据转换、100-step、离线自检、Mock WebSocket | 2–4 | 1–4 | 不适用；真实提交仍为 TODO | 20–50 GB | 11–13 GB | 中高 |
| LIBERO-plus | 7 类扰动的分层抽样，约 300–1,000 cases | 1–2 | 4–12 | 75–170（10,030 cases） | 25–80 GB | 11–24 GB | 中高 |
| SimplerEnv | 一个公开 checkpoint 的完整 WidowX 评测 | 1–3 | 8–24 | 8–24 | 30–80 GB | 12–20 GB | 中 |
| RoboDojo | 12 个代表任务×10 episodes、一个 head | 2–4 | 4–12 | 30–70/策略（2,100 episodes） | 85–130 GB | 18–30 GB（含仿真） | 中 |
| RoboTwin 2.0 | 10 个任务、clean/randomized、各10 episodes | 2–4 | 8–20 | 100–250（50任务、双设置） | 40–100 GB（仅评测） | 18–30 GB（含仿真） | 中低 |
| RoboCasa365 | OpenDrawer 全流程＋约10个代表任务 | 2–4 | 4–12 | 40–100（365任务抽样复核） | 30–80 GB | 16–28 GB（含仿真） | 中 |
| DOMINO | 10个动态任务×10 episodes | 2–4 | 6–18 | 30–80（35任务） | 60–150 GB | 18–30 GB（含仿真） | 中低 |
| CALVIN | 约200条长序列 | 2–4 | 4–10 | 12–30（标准长序列评测） | 80–150 GB | 16–28 GB（含仿真） | 中 |
| VLA-Arena | 4个代表 suite，覆盖 L0–L2 | 3–6 | 8–20 | 30–80（11 suite×3级） | 80–200 GB | 18–30 GB（含仿真） | 中低 |
| MetaWorld MT50 | 官方 500 episodes | 1–2 | 4–8 | 4–8 | 20–40 GB | 18–24 GB | 中高 |
| Unitree G1 WholeBody | 数据 schema、78D接口、短训练、假客户端 | 2–5 | 1–4 | 无硬件时不能完成物理闭环 | 20–50 GB | 12–24 GB | 中低 |
| UMI4Pretraining | DexWild/小样本数据验证、20-step smoke | 2–5 | 1–6 | 取决于最多400案例的数据获取范围 | 80–300 GB | 11–18 GB | 低 |

## 4. 三档总体计划

### A. 汇报最小闭环

范围：RoboChallenge、LIBERO-plus 分层抽样、SimplerEnv、MetaWorld、RoboDojo 小规模代表任务。

- GPU：约 **25–60 小时**。
- 工程：约 **8–13 人日**。
- 单人日历：约 **2–3 周**。
- 工作盘：建议 **200–350 GB**；若数据集按阶段下载和清理，可压到约 150–250 GB。
- 适用结论：标准性能、鲁棒性、跨 embodiment、策略服务、准真实双臂接口。

### B. 推荐的完整汇报组合

在 A 的基础上加入 RoboTwin、RoboCasa365、DOMINO 和 VLA-Arena 代表任务。

- GPU：约 **50–140 小时**。
- 工程：约 **12–22 人日**。
- 单人日历：约 **3–5 周**。
- 工作盘：建议 **250–600 GB**。
- 适用结论：进一步覆盖双臂、移动厨房、动态目标和安全约束，已经足以形成完整的课程/课题汇报。

### C. 官方全量评测型复现

核心 benchmark 尽量运行官方完整任务规模；每个 benchmark 先只选一个公开策略，避免把动作头数量与任务数量相乘。

- GPU：约 **330–820 小时**，单卡纯计算约 **14–34 天**。
- 工程：约 **25–40 人日**。
- 单人日历：约 **5–9 周**。
- 工作盘：评测优先约 **0.5–1.5 TB**；如保留训练集、多个 checkpoint 和全量视频，建议 **1–3 TB**。
- 若 RoboDojo 三种 head、RoboTwin 两种环境和更多随机种子全部展开，GPU 时间还会增加约 2–3 倍。

## 5. 训练时间的单卡换算

对于与当前 OFT 配置接近的冻结主干或参数高效训练，已有两条实测吞吐可作为边界：

| 步数 | 6.52 steps/s 乐观边界 | 1.80 steps/s 保守边界 | checkpoint 策略 |
|---:|---:|---:|---|
| 100 | 约15秒 | 约56秒 | 仅保存 compact 或不保存 |
| 1K | 约2.6分钟 | 约9.3分钟 | 保存1个 compact gate |
| 30K | 约1.28小时 | 约4.63小时 | 只保留里程碑版本 |
| 50K | 约2.13小时 | 约7.72小时 | 最多保留2–3个完整版本 |
| 100K | 约4.26小时 | 约15.43小时 | 推荐 compact＋最终完整权重 |
| 130K | 约5.54小时 | 约20.06小时 | 适用于 RoboDojo 量级的训练规划 |

这些换算不适用于全参数 Qwen3-VL-4B、原始 PI 大头或需要重型视频编码器的模型；它们的单步显存和速度需重新做 20/100-step gate。当前 32 GiB 单卡的全参数微调已经被实测 OOM 否决。

## 6. 占用风险与调度建议

### GPU

- OFT/GR00T policy server 单独运行约需 11–12 GiB；PI_v3 约16.2 GiB，PI约24.1 GiB。
- RoboDojo、RoboTwin、RoboCasa、DOMINO、VLA-Arena 的仿真器也可能占 GPU。单卡同时运行策略和仿真时，先做峰值探针；PI 和重型仿真同卡不应直接进入全量任务。
- 只有单张 GPU 时，不建议用多个 policy server 并行。任务级并发应以 CPU 仿真是否成为瓶颈为依据逐步从1增加到2。

### CPU 与 RAM

- 64 GiB RAM 对单个环境通常够用，但4 CPU会限制视频解码、仿真并发和多任务调度。
- 评测以1个仿真任务/进程起步；确认 RAM 峰值和 CPU 利用率后再增加到2。超过2通常不一定缩短总时间。

### 磁盘

- 现有58 GB占用加上一个9.14 GB checkpoint后只剩约33 GB，连 RoboDojo 的64 GB数据集都放不下。
- 应将 dataset、conda/pip cache、仿真 assets 和视频输出迁移到额外数据盘；项目盘只保留代码、配置、JSON结果和精选视频。
- 全量视频不是必需证据。建议每任务保存首回合、首个成功、首个失败和异常回合，其余仅保存JSON指标。
- 训练中间点用262 MB级 compact checkpoint；只对最终候选保存9.14 GB完整权重，并做SHA-256后转移归档。

## 7. 建议执行顺序

1. 扩容或挂载至少300 GB工作盘；在开始 RoboDojo/RoboTwin 前最好达到500 GB。
2. RoboChallenge 完成 pre-hardware L2：数据转换、100-step、local self-test、mock server、假机器人客户端。
3. LIBERO-plus 先抽样300 cases；依据方差决定是否扩到1,000或完整10,030 cases。
4. SimplerEnv和MetaWorld用公开 checkpoint完成完整评测，建立跨 benchmark 基线。
5. RoboDojo与RoboTwin各先跑10–12个代表任务；只有服务稳定且结果可解释时才扩到官方全量。
6. RoboCasa365、DOMINO、VLA-Arena用于补齐移动操作、动态环境和安全约束。
7. Unitree G1和UMI仅做数据/接口/Mock扩展，不在没有硬件时声称完成真实机器人闭环。

## 8. 估算边界

- GPU小时主要根据现有LIBERO episode耗时和官方任务/episode规模外推；不同仿真器的物理步进和最大episode长度会造成约2倍甚至更大的偏差。
- 工程人日不是纯下载/运行时间，而是包含环境冲突、版本固定、日志整理与一次失败重跑的日历规划。
- RoboTwin、DOMINO、VLA-Arena和UMI的完整数据体积未在对应README中统一给出，因此磁盘区间为容量规划值，不是下载清单的精确和。
- 网络下载速度、服务器排队时间和真实硬件调试时间未计入。

## 9. 证据来源

- `D:/ProgrammingProjects/StarVLA/docs/reports/StarVLA当前完整测试汇总报告-20260730.md`
- `D:/ProgrammingProjects/StarVLA/docs/reports/StarVLA已完成结果复核与无时间限制实验方案.md`
- `D:/ProgrammingProjects/StarVLA/docs/reports/StarVLA-ActionHead比较报告-20260730.md`
- StarVLA `starVLA_dev` 分支中各 `examples/simBenchmarks`、`examples/realRobots` 和 `examples/human2robots` README（本次核对基线为本地 `upstream/starVLA_dev`）。
