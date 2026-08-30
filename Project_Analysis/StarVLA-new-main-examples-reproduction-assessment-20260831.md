# StarVLA 新 main 分支 Examples 内容与复现评估

审计对象：`starVLA` 的 `main@48e588114071a898b861b5eeb59d5defde2ba361`（2026-08-28）。本文只评估无实体机械臂、单张 RTX 5090 32 GiB、可逐项目清理空间的复现条件。概率是达到“指定工程验收”的估计，不是模型任务成功率。

## 结论

新的 main 已从旧 fork 的扁平 Examples 重组为四个类别，共 **24 个项目**：13 个仿真 benchmark、5 个真实机器人案例、5 个模型扩展及 1 个 human-to-robot 预训练案例。它增加了 MetaWorld、RoboDojo、VLN-CE、EgoVLA、Realman、Unitree G1 WholeBody、DiscreteDiffusion、MiniCPM、UMI4Pretraining 等能力线。

建议老师汇报采用“**已完成 LIBERO 基线 + 3 个新仿真正式 benchmark + 1 个训练/部署流程补充**”的组合：

1. MetaWorld MT50：最优先的新 benchmark。
2. CALVIN：补足长时序组合任务。
3. RoboCasa Tabletop 或 SimplerEnv：前者偏桌面操作指标，后者偏真实机器人迁移仿真。
4. RoboCasa365 冻结 VLM/LoRA 短训练闭环，或 RoboChallenge mock server：作为数据到部署服务的补充。
5. 后续再扩展 RoboTwin；不要把它作为第一个新项目。

既有 RTX 5090 结果显示：官方 Qwen3-VL-OFT 的 LIBERO 为 **392/400（98%）**；4B 全参数 AdamW 在初始化优化器状态时已 OOM；OFT、GR00T、PI_v3、PI 推理峰值约为 10.6、11.4、16.2、24.1 GiB。因此，**不建议以全量微调作为主要目标**。优先使用公开 checkpoint 完成 policy server—client/simulator—正式评测—JSON/视频证据闭环；训练只选一个 Example 完成冻结 VLM 或 LoRA 的 20–100 step 工程验证。

## 当前 Examples 的组成

```text
examples/
├─ simBenchmarks/     Behavior, CALVIN, DOMINO, LIBERO, LIBERO-plus,
│                     MetaWorld, RoboDojo, RoboCasa_365, RoboCasa_tabletop,
│                     RoboTwin, SimplerEnv, VLA-Arena, VLN-CE
├─ realRobots/        EgoVLA, Franka, Realman, RoboChallenge_table30v2,
│                     UnitreeG1_WholeBody
├─ modelExtensions/   CoTrainVLM, DiscreteDiffusion, Gemma4, MiniCPM, NeuralVLA
└─ human2robots/      UMI4Pretraining
```

## 先处理的 main 分支问题

静态审计覆盖 `examples` 下 284 个文件（95 Python、93 shell、40 Markdown、31 YAML）。95 个 Python 都能通过 AST 解析；93 个 shell 有 91 个通过 `bash -n`。但这不代表 README 命令可以零修改运行。

| 发现 | 影响 | 复现前处理 |
| --- | --- | --- |
| 33 处 shell 仓库根目录推导中仅 2 处实际指向 repo root | 31 处可能落入 `starVLA/examples` 或 repo 父目录 | 统一修正相对层级；支持覆盖者显式设置 `STARVLA_DIR` |
| RoboCasa365 两个训练脚本语法失败 | `export WANDB_API_KEY=<...>` 中的尖括号使脚本启动即退出 | 删除占位符、加引号，或使用 `WANDB_MODE=disabled` |
| 7 个 Markdown 相对链接失效 | 文档跳转错误，反映目录重组未完全收尾 | 按新层级修正文档链接 |
| SimplerEnv 默认脚本仅启用一个 WidowX 任务 | 运行默认脚本不能代表完整官方结果 | 预先写明 4-task×seed 或明确的缩减协议 |

受根路径回归影响的目标包括 MetaWorld、CALVIN、LIBERO、LIBERO-plus、SimplerEnv、RoboTwin、DOMINO、VLA-Arena、VLN-CE、Franka、RoboChallenge 和 Unitree G1。建议先建立一个小型兼容补丁并对每个项目做 `import → server handshake → 1 episode` gate，再投入正式评测。

## 仿真 benchmark 复现排序

| 优先级 | Example | 官方内容与建议验收 | 5090 完成可能性 | 时间 / 新增峰值空间 |
| --- | --- | --- | --- | --- |
| P0 | MetaWorld | MT50；50 task × 10 episode；四难度桶与 `summary.json`；两份约 8.8 GB PI_v3 权重 | 85–95%（修路径后） | 1.5–3 天 / 25–55 GB |
| P0 | CALVIN | 长时序任务链；公开 Qwen2.5-GR00T checkpoint；正式序列指标与视频 | 75–90% | 2–4 天 / 45–100 GB |
| P0 | RoboCasa Tabletop | 24 task × 50 rollout；QwenOFT/GR00T 权重；厨房桌面操作 | 70–85% | 3–6 天 / 70–150 GB |
| P0 | SimplerEnv | WidowX/Google Robot 迁移仿真；GR00T、PI_v3 结果 | 65–85% | 3–6 天 / 60–120 GB |
| P1 | RoboTwin | 50 个双臂任务、14D action、h=50；公开 OFT | 55–75% | 5–10 天 / 100–250 GB |
| P1 | LIBERO-plus | 10,030 个测试案例；适合作为已完成 LIBERO 的鲁棒性补充 | 80–95% 抽样；60–80% 全量 | 1–3 天抽样 / 25–120 GB |
| P1 | RoboCasa365 | Panda-Omron；数据→训练→服务→评测教程 | 70–85% 冻结/LoRA；全参默认流程 10–25% | 3–6 天 / 50–110 GB |
| P1/P2 | RoboDojo | 42 task × 50；多视角双臂；XPolicyLab 与 Isaac 生态 | 45–65% 单任务；25–50% 全量 | 4–8 天起 / 180–320 GB |
| P2 | VLN-CE | Habitat 视觉语言导航，外部评测仓库 | 45–70% | 4–8 天 / 100–250 GB |
| P2 | VLA-Arena | 4 domain、11 suite、3 level；成功率与安全成本 | 30–55% | 5–10 天 / 100–250 GB |
| P2 | DOMINO | 35 个动态 task；历史图像接口 | 25–45% | 1–2 周 / 120–300 GB |
| P3 | Behavior-1K | OmniGibson 长任务；当前目录标注 Under construction | 环境 25–45%；完整结果低于 20% | 1–3 周 / 250 GB–1 TB+ |

说明：RoboCasa Tabletop 适合做“性能复现”；RoboCasa365 适合做“完整训练/部署流程复现”。两者不要混为同一验收目标。SimplerEnv 的官方文档明确需要 NVIDIA GPU 与 Vulkan；RoboDojo、Unitree G1 等 Isaac 系项目还需满足 RTX/图形栈和外部资产要求。[SimplerEnv 官方文档](https://github.com/allenai/SimplerEnv/blob/main/README.md)；[Isaac Sim 系统要求](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)。

## 其他 11 个 Example 的定位

| 类别 | 项目 | 无实体机器人时可完成的内容 | 建议 |
| --- | --- | --- | --- |
| realRobots | RoboChallenge_table30v2 | data conversion、短训、local self-test、mock server | 最推荐作为部署协议补充；production 仍为 TODO |
| realRobots | UnitreeG1_WholeBody | 仿真遥操作、LeRobot 转换、1-step smoke、服务/控制器 dry-run | 高级流程展示，依赖外部 WBC/Isaac |
| realRobots | EgoVLA | 有授权权重/MANO/retarget 资源时运行 G1 red-ball sim | 许可证与资源门槛高 |
| realRobots | Franka | dataloader、训练 smoke、PolicyServer 接口 mock | 教程型内容，不是无机器人闭环 demo |
| realRobots | Realman | 自备兼容数据后的 VM4A 训练 | 私有数据未发布、无正式 eval |
| modelExtensions | CoTrainVLM | VLA+VLM 的 20–100 step 共训 smoke | 可作为扩展性证据 |
| modelExtensions | DiscreteDiffusion | forward/train/RTC API smoke | 无发布闭环权重；部署脚本在作者 fork |
| modelExtensions | Gemma4 | 单卡 smoke 或已有权重评测 | 96% LIBERO 正式训练使用 8×H100，不复刻训练 |
| modelExtensions | MiniCPM | 轻量 1.3B backbone/head smoke | 适合后续轻量消融，依赖较新 |
| modelExtensions | NeuralVLA | history-state 接口 smoke | 未提供公开权重和正式结果 |
| human2robots | UMI4Pretraining | 选单 family 完成下载、verify、dataloader、20-step smoke | 全量 400 case 需约 1 TB 级工作盘；100 GB 仅做子集 |

## 建议的汇报型执行计划

1. **第 1–2 天：** 先修 main 的路径回归，并用已完成 LIBERO 做 1–4 episode 回归；留下补丁、环境锁文件与日志。
2. **第 2–4 天：** MetaWorld 完成 500 episodes、四桶统计、逐 task JSON 与精选视频，形成第二个正式 benchmark。
3. **随后 5–10 天：** CALVIN 完整复现；再在 RoboCasa Tabletop 与 SimplerEnv 中选择一个做第三个 benchmark。
4. **额外 3–6 天：** 用 RoboCasa365 的冻结/LoRA 短训练，或 RoboChallenge mock，展示数据→checkpoint→server→client 流程。
5. **后续扩展：** RoboTwin 先完成 3-task gate，再扩至 10/50 task；将 UMI、Unitree、RoboDojo 作为加分项，而非主线依赖。

空间按逐项目清理计算。完成一个项目后只保留：环境 YAML/lock、启动命令、配置、逐任务 JSON、精选视频、checkpoint 哈希和关键日志；删除权重 cache、解压资产副本和全量视频。当前 100 GB 可滚动完成 MetaWorld 与短流程；CALVIN、RoboCasa、SimplerEnv 建议争取 200 GB 以上临时盘，RoboTwin 建议 300 GB 以上。

## 验收口径

“完成一个 Example”应至少包括：环境及版本记录、公开 checkpoint 或短训权重、policy server 正常握手、客户端/仿真环境正式 rollout、逐任务 JSON、与官方口径的定量对照、可播放视频，以及失败项说明。只有 dataloader/forward/单 step 不应单独宣称为完整 demo；但可作为模型扩展或真实机器人接口的工程 smoke 证据。
