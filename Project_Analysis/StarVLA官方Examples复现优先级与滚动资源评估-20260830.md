# StarVLA 官方 Examples 复现优先级与滚动资源评估

评估日期：2026-08-30  
代码参考：本地 `upstream/starVLA_dev`，提交 `48e588114071a898b861b5eeb59d5defde2ba361`（2026-08-28）  
计算环境：单张 NVIDIA RTX 5090 32 GiB、4 CPU、64 GiB RAM、当前服务器约 100 GB 存储配额  
既有基础：LIBERO 官方 Qwen3-VL-OFT 50K checkpoint 已完成 400 episodes，`392/400 = 98.0%`

## 1. 技术摘要

老师提出“完成得过于简单、需要跑通几个 Example”，更合理的理解是：不能只提交一个 LIBERO 官方 checkpoint 评测和大量 smoke/资源诊断，而应从 StarVLA 官方 `examples` 中选择若干具有不同能力边界的项目，分别完成环境、资产、模型服务、仿真评测、结果统计和官方结果对照。

在允许“每完成一个项目就清理其数据、环境缓存和非必要视频”的条件下，空间需求不再是所有项目容量之和，而由**单个项目的最大峰值工作集**决定。建议如下：

- **最稳妥的三项目组合**：MetaWorld MT50、SimplerEnv、RoboCasa365，预计 `8–13` 个日历日、`16–44` GPU 小时，建议服务器总配额 `200–250 GB`。
- **最推荐的四项目组合**：在上述基础上加入 RoboTwin，预计 `3–5` 周、`40–120` GPU 小时，建议总配额 `300–400 GB`。
- **研究展示增强组合**：再加入 RoboDojo，预计 `5–8` 周、`80–220` GPU 小时，峰值总配额建议 `500 GB`，更稳妥为 `1 TB`。
- RoboChallenge 适合作为部署协议补充，LIBERO-plus 适合作为已有 LIBERO 结果的鲁棒性扩展，但二者不应替代三个新的完整仿真 Example。
- 单张 RTX 5090 的策略推理能力已经被现有 LIBERO 结果验证。新的主要风险是外部仿真环境、版本和归一化键，而不是纯模型算力。

## 2. “完整复现一个 Example”的验收标准

后续不再把单次 `forward()`、随机输入 action shape 检查或 5–20 step smoke 单独计为一个完整 Example。一个项目至少满足以下条件中的前 8 项：

1. 固定 StarVLA commit、外部仿真器 commit、Python、PyTorch、CUDA 和关键依赖版本。
2. 完成独立环境安装并通过环境自检。
3. 下载并校验官方资产、数据或 checkpoint。
4. 验证 observation、state、action、language、action horizon 和归一化键。
5. 启动 StarVLA policy server，并记录启动显存和可用 `unnorm_key`。
6. 启动官方仿真 client，完成至少一个 episode。
7. 完成预先定义的正式评测规模，生成成功率、逐任务 JSON 和失败清单。
8. 保存代表性成功/失败视频、服务器日志和仿真日志。
9. 与官方结果比较，报告绝对差值、可能原因和统计协议差异。
10. 若官方 Example 包含训练 walk-through，则完成训练、checkpoint 保存、严格重载和评测。
11. 提供一条从干净环境到结果文件的可重复执行命令或脚本。

对老师汇报时，必须区分以下三种状态：

- **工程跑通**：环境、server 和 client 能连续运行并产生 rollout。
- **性能复现**：成功率与官方值在合理采样误差和协议差异内接近。
- **流程复现**：完成数据、训练、checkpoint、部署和评测全链路，但短训练 checkpoint 不一定具有任务能力。

## 3. 推荐项目优先级

### 3.1 按研究价值排序

| 优先级 | Example | 核心价值 | 推荐复现范围 | 为什么值得计为独立项目 |
|---:|---|---|---|---|
| P0-1 | SimplerEnv | Bridge/RT-1 真实数据、跨 embodiment、WidowX 仿真 | 官方 GR00T 或 PI_v3 checkpoint；完整 WidowX 评测 | 与 LIBERO 的任务、数据来源和机器人形态明显不同，并有非零官方成功率 |
| P0-2 | RoboCasa365 | 数据—训练—服务—MuJoCo client 完整教程 | OpenDrawer 官方 100-step；增加 1K 训练和 10–20 episodes | 能证明不是只会下载 checkpoint，而是跑通自训练全流程 |
| P0-3 | RoboTwin 2.0 | 双臂、14D action、horizon 50 | 官方 OFT；先 10 任务，再扩到 50 任务 clean | 补齐双臂与长 action chunk，和 LIBERO 差异最大 |
| P0-4 | MetaWorld MT50 | 50 任务、500 episodes、公开 PI_v3 checkpoint | 两个公开 checkpoint 至少完整评测一个，建议都跑 | 完整度高、风险较低，容易快速形成第二份完整结果表 |
| P1-1 | RoboDojo | 三视角双臂、42 任务、Isaac Sim、三种 action head | OFT 单任务 smoke → 12 任务 → 42 任务 | 研究展示价值最高，但外部依赖、资产和单卡资源风险最大 |
| P1-2 | RoboChallenge | 真实数据转换和 Mock 部署协议 | 单任务数据→100-step→offline→Mock server | 适合“部署迁移”章节，但没有仿真成功率，不应代替主 benchmark |
| P1-3 | LIBERO-plus | 现有 98% 模型的分布外鲁棒性 | 先分层抽样 300–1,000 cases，再决定是否跑 10,030 cases | 能深化已有 LIBERO 结论，但与原项目重合度较高 |
| P2 | CALVIN / DOMINO / VLA-Arena | 长时序、动态环境、安全约束 | 在前四个完成后各选代表任务 | 能增加广度，但不应在基础完整 Example 尚未形成时分散精力 |

### 3.2 建议实际执行顺序

研究价值最高的项目不一定最适合第一个执行。建议采用从低风险到高风险的顺序：

1. **MetaWorld MT50**：最快取得一个新的完整 benchmark 成果，并验证新版本 policy server。
2. **SimplerEnv**：完成真实数据到 WidowX 仿真的跨 embodiment 复现。
3. **RoboCasa365**：完成一个包含训练的端到端 walk-through。
4. **RoboTwin**：进入双臂和长 action chunk。
5. **RoboChallenge**：补充真实数据转换与部署协议。
6. **RoboDojo**：扩容并确认 Isaac Sim 后作为高级成果。
7. **LIBERO-plus**：利用空闲计算时间扩展鲁棒性样本量。

这个顺序的目标是先快速形成两个可展示的完整成功项目，再进入环境适配和资源风险更高的案例。

## 4. 单项目时间与滚动空间评估

以下“峰值工作空间”是该项目运行时的总新增工作集估算，包括独立环境、官方 checkpoint、仿真资产、结果和临时视频，不包含当前约 58 GB 的既有 StarVLA 项目逻辑占用。“建议服务器总配额”已经把当前占用和安全余量计入。

| Example | 环境与准备 | 正式运行 | 整理与复核 | 预计日历时间 | 新增峰值空间 | 清理后建议保留 | 建议服务器总配额 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MetaWorld MT50 | 0.5–1.5 天 | 4–8 GPU 小时 | 0.5 天 | 1.5–3 天 | 35–70 GB | 3–8 GB | 150–200 GB |
| SimplerEnv | 1–2 天 | 8–24 GPU 小时 | 0.5–1 天 | 2.5–5 天 | 60–120 GB | 4–10 GB | 200–250 GB |
| RoboCasa365 | 1–2 天 | 4–12 GPU 小时 | 0.5–1 天 | 2.5–5 天 | 50–100 GB | 3–10 GB | 180–250 GB |
| RoboTwin 10任务阶段 | 2–4 天 | 8–20 GPU 小时 | 1 天 | 4–7 天 | 80–180 GB | 5–15 GB | 250–350 GB |
| RoboTwin 50任务 clean | 在前阶段基础上 | 20–60 GPU 小时 | 1–2 天 | 3–6 天追加 | 100–250 GB | 8–20 GB | 300–400 GB |
| RoboDojo 12任务 | 3–6 天 | 8–25 GPU 小时 | 1 天 | 5–9 天 | 180–280 GB | 8–20 GB | 350–500 GB |
| RoboDojo 42任务×50 | 在 smoke 基础上 | 30–70 GPU 小时/策略 | 1–2 天 | 4–8 天追加 | 200–320 GB | 10–30 GB | 500 GB–1 TB |
| RoboChallenge | 1–2 天 | 1–4 GPU 小时 | 0.5–1 天 | 2–4 天 | 30–70 GB | 2–8 GB | 150–200 GB |
| LIBERO-plus 抽样 | 0.5–1 天 | 4–12 GPU 小时 | 0.5 天 | 1–3 天 | 25–80 GB | 3–10 GB | 150–250 GB |
| LIBERO-plus 10,030 cases | 在抽样基础上 | 75–170 GPU 小时 | 1–2 天 | 5–10 天 | 50–150 GB | 5–20 GB | 250–350 GB |

### 4.1 时间区间的含义

- 环境与准备包含外部仓库、conda 环境、资产路径、Vulkan/EGL、server/client 协议和首个 episode 排错。
- 正式运行使用单张 RTX 5090 串行估算，不包含服务器排队和网络下载。
- 整理与复核包含 JSON 汇总、视频筛选、官方结果对照和报告更新。
- 新环境首次运行的不确定性通常大于纯 GPU 计算时间，因此不能只用 episode 数估算日历工期。

### 4.2 为什么清理后仍需保留若干 GB

每个项目完成后不建议只保留一张成功截图。长期证据至少包括：

- 固定后的配置和运行脚本；
- `pip freeze`、conda spec、StarVLA和外部仓库commit；
- 完整汇总 JSON、逐任务结果和失败日志；
- 5–20个代表性成功/失败视频；
- checkpoint的URL、SHA-256和 `dataset_statistics.json`；
- 若为自训练模型，保留 compact checkpoint；最终重要模型再归档完整权重。

公开 checkpoint 可以在完成哈希记录后删除，后续按需重新下载。自训练完整 checkpoint 建议移到长期归档盘，不要长期占用滚动工作盘。

## 5. 各项目具体执行合同

### 5.1 MetaWorld MT50：先取得低风险完整成果

建议正式协议：

- 50 tasks × 10 episodes = 500 episodes；
- 官方 `la_finetune` checkpoint 必做；
- `baseline_finetune` 建议追加，形成预训练对照；
- 记录 easy、medium、hard、very_hard 四档成功率；
- 保存每档至少两个成功和两个失败视频。

完成判定：500 episodes 无缺失；结果 JSON 可重复汇总；公开 checkpoint 至少一个得到非零且任务分布合理的成功率。

预计新增峰值空间 `35–70 GB`。如果默认每个 episode 都保存视频，空间可能继续增长，因此正式运行前应改为精选视频策略，或在汇总完成后删除重复视频。

### 5.2 SimplerEnv：最重要的跨 embodiment 成果

建议正式协议：

- 先运行环境验证脚本；
- 优先使用 Qwen3VL-GR00T 或 Qwen3VL-PI_v3；
- 使用官方 `oxe_bridge`/`oxe_rt1` 归一化键；
- 完成 WidowX 官方任务评测；
- 如时间允许，再扩展 Google Robot visual matching；
- 不首先使用存在公开复现分歧的 OFT checkpoint。

完成判定：所有目标 episodes 正常结束；无 server/simulator异常；成功率非零；与官方约 65%–70% 结果比较，并解释采样协议和环境版本差异。

主要风险：NumPy 1.24.4、SAPIEN/Vulkan、双环境通信和旧checkpoint统计量键映射。

### 5.3 RoboCasa365：必须包含训练的项目

建议正式协议：

1. 下载厨房资产和 OpenDrawer LeRobot 数据。
2. 完成 dataloader 和 modality 检查。
3. 严格复现官方 100-step walk-through。
4. checkpoint 严格重载并启动 WebSocket server。
5. 完成 2–10 episodes，验证 JSON 和视频。
6. 在资源允许时追加 1K 训练和相同评测，比较动作、loss和成功率。

完成判定重点是“数据—训练—服务—仿真”全链路，而不是要求100-step模型必须成功。报告中必须明确100-step是工程 walk-through，不能把0成功解释为StarVLA方法失败。

### 5.4 RoboTwin：形成双臂完整项目

建议分两阶段：

- Gate A：3 tasks × 1 episode，确认外部补丁、server和14D动作。
- Gate B：10个代表任务，clean/randomized各10 episodes。
- 正式阶段：50 tasks clean，每任务10 episodes；若要与官方表严格比较，再按官方episode协议扩大。

建议使用官方QwenOFT checkpoint。暂不把QwenFAST作为主线，因为当前社区仍有动作序列解码和配置复现问题。

若只进行评测，不需要保留完整训练数据；这会显著降低磁盘峰值。只有在官方checkpoint评测稳定后，才考虑下载clean训练集做小规模再训练。

### 5.5 RoboDojo：高级项目而非首批项目

RoboDojo需要StarVLA策略环境、Isaac Sim/Isaac Lab、XPolicyLab、约64GB LeRobot数据（训练时）以及大体积仿真资产。当前4 CPU仅适合低并发，策略和仿真同卡还需要显存探针。

建议顺序：

1. offline wiring；
2. 单任务单episode；
3. 3个能力维度代表任务；
4. 12任务×10 episodes；
5. 最后才执行42任务×50 episodes。

首个策略只使用OFT。GR00T和PI_v3属于后续对照，不应在环境尚未稳定时同时展开。

## 6. 滚动磁盘管理方案

### 6.1 空间模型

逐项目清理后，总容量可以按下式规划：

```text
所需总配额
= 当前长期保留内容
+ 当前项目峰值工作集
+ 15%–20% 安全余量
```

不应按“所有项目数据集之和”购买空间，但也不能只按checkpoint大小估算，因为conda环境、仿真资产、Hugging Face cache、视频和同一数据的转换前后副本会同时存在。

### 6.2 目录分层

建议服务器采用：

```text
StarVLA/
├─ code/                 # 长期保留，git仓库
├─ results_archive/      # JSON、配置、日志、精选视频、compact ckpt
├─ shared_models/        # 当前项目使用的模型；完成后可清理
└─ work/
   └─ <example_name>/    # 当前项目数据、仿真资产、临时视频和cache
```

每个项目结束后，只清理明确位于 `work/<example_name>`、该项目独立conda环境和已校验可重下的cache。不要对仓库根目录、共享模型目录或未确认路径执行递归删除。

### 6.3 每个项目的清理验收

删除工作集前必须确认：

1. 汇总 JSON 可以独立读取。
2. 成功率可由保留的逐episode结果重新计算。
3. 配置、commit和环境清单已归档。
4. 代表视频已复制到 `results_archive`。
5. 自训练checkpoint已做严格重载。
6. 完整checkpoint已计算SHA-256，或已转移到长期归档盘。
7. 原始数据和资产存在可重复下载来源。

完成后，公开checkpoint、全量视频、下载cache和可重建的转换中间数据可以删除。

## 7. 三档执行计划

### 计划 A：快速形成三个完整项目

项目：MetaWorld MT50、SimplerEnv、RoboCasa365。

| 指标 | 估算 |
|---|---:|
| 日历时间 | 8–13 天 |
| GPU 小时 | 16–44 |
| 同时峰值新增空间 | 60–120 GB |
| 建议服务器总配额 | 200–250 GB |
| 项目后长期保留 | 10–30 GB |

适合尽快向老师展示：已经从一个LIBERO扩展到三个结构明显不同、完整运行的官方Example。

### 计划 B：推荐的四项目组合

在计划A基础上加入RoboTwin。

| 指标 | 估算 |
|---|---:|
| 日历时间 | 3–5 周 |
| GPU 小时 | 40–120 |
| 同时峰值新增空间 | 100–250 GB |
| 建议服务器总配额 | 300–400 GB |
| 项目后长期保留 | 20–50 GB |

这是当前最推荐的最终汇报范围：单臂、多任务、真实数据迁移、训练全流程和双臂长动作块均得到覆盖。

### 计划 C：加入高级展示项目

在计划B基础上加入RoboDojo，并将RoboChallenge作为部署协议补充。

| 指标 | 估算 |
|---|---:|
| 日历时间 | 5–8 周 |
| GPU 小时 | 80–220 |
| 同时峰值新增空间 | 200–320 GB |
| 建议服务器总配额 | 500 GB，推荐1 TB |
| 项目后长期保留 | 30–80 GB |

计划C研究价值最高，但不应在前三个完整项目尚未形成时提前进入Isaac Sim环境排错。

## 8. 决策建议

### 8.1 当前立即执行

1. 将服务器存储配额从100GB扩展到至少250GB；若确定做RoboTwin，直接申请400GB更合理。
2. 更新到固定的StarVLA提交，不在运行中追随 `starVLA_dev` 最新HEAD。
3. 先执行MetaWorld，目标是一周内形成第二个完整benchmark报告。
4. 随后执行SimplerEnv和RoboCasa365。
5. 完成前三个后，再决定RoboTwin全50任务协议和RoboDojo投入。

### 8.2 暂不执行

- 不进行Qwen3-VL-4B全参数单卡微调；已有实测在Adam状态初始化阶段OOM。
- 不同时安装和保留所有仿真环境。
- 不为所有episodes永久保存视频。
- 不把Unitree G1、Franka或Realman的接口说明当作已经完成的官方仿真项目。
- 不在官方checkpoint尚未跑通前训练新的大模型配置。

## 9. 不确定性和复核方法

本报告中的空间和时间均为容量规划区间，不是对尚未运行项目的实测承诺。误差主要来自：

- 外部仿真资产版本和实际下载体积；
- 视频保存策略；
- policy与sim是否同GPU；
- 每个任务最大episode长度；
- 首次安装时的依赖修复次数；
- 外部仓库是否需要补丁。

每个新项目完成10个episode后，应重新记录：

- 秒/episode；
- policy峰值显存；
- simulator峰值显存；
- CPU和RAM峰值；
- 单episode视频大小；
- 失败重试率。

然后按实际结果更新正式评测时间和磁盘上限。10-episode gate之后，预计可将大多数时间区间收紧到约±20%–30%。

## 10. 证据来源

- `D:/ProgrammingProjects/StarVLA/docs/reports/StarVLA当前完整测试汇总报告-20260730.md`
- `D:/ProgrammingProjects/StarVLA/docs/reports/StarVLA已完成结果复核与无时间限制实验方案.md`
- `D:/ProgrammingProjects/StarVLA/docs/reports/StarVLA-ActionHead比较报告-20260730.md`
- 本地 `upstream/starVLA_dev` 的 `examples/simBenchmarks`、`examples/realRobots` 和 `examples/human2robots` README。
- StarVLA SimplerEnv、MetaWorld、RoboCasa365、RoboTwin、RoboDojo、LIBERO-plus和RoboChallenge官方示例文档。

## 11. 最终建议结论

本阶段不再以“是否属于real robot目录”为首要标准，而以“是否能形成完整、可展示、可与官方结果对照的项目复现”为标准。最终建议提交以下主成果：

1. 已完成：LIBERO官方checkpoint，392/400。
2. 新增：MetaWorld MT50完整500 episodes。
3. 新增：SimplerEnv官方checkpoint完整评测。
4. 新增：RoboCasa365 OpenDrawer训练—部署—仿真全流程。
5. 新增：RoboTwin双臂代表任务，条件允许时扩到50任务。
6. 扩展：RoboChallenge部署协议或RoboDojo高级双臂评测。

采用逐项目清理后，完成前四个新增项目不需要同时准备多TB空间；建议将服务器总配额提升到 `300–400 GB`。如果最终加入RoboDojo，则建议提升到至少 `500 GB`。
