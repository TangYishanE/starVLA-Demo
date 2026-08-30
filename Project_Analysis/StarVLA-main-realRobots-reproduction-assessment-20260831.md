# StarVLA 新 main：realRobots Examples 复现评估

审计对象：`main@48e588114071a898b861b5eeb59d5defde2ba361` 的 `examples/realRobots`。场景边界是：单张 RTX 5090 32 GiB、远程服务器、无实体机械臂或人形机器人；可以使用公开数据、离线回放、mock server 和第三方仿真，但**不把任何非实体执行结果称为 real-robot 闭环成功**。

## 技术结论

若目标是回应“跑通 StarVLA 的几个 real demo”，最合理的顺序是：

1. **RoboChallenge_table30v2：P0，首个必须完成。** 它是五项中唯一同时给出公开数据下载、转换、100-step 训练、离线 self-test 和上游 mock-server 协议的项目。无机械臂时仍能形成最完整、最诚实的 real-robot-style 流程证据。
2. **UnitreeG1_WholeBody：P1，高价值但高集成风险。** 它覆盖仿真遥操作/数据、LeRobot、78D 训练、WebSocket server 与 SONIC/控制器桥接；最适合展示系统设计能力，但远程 5090 上无法证明真实 G1 的安全执行。
3. **Franka：P2，只有获得兼容数据后才启动。** 这是通用单/双臂数据转换和 PolicyServer 教程，不附数据、权重、仿真环境或成功率基准。应做成“自有 Franka 数据接入模板”，不要作为独立 demo 主线。
4. **EgoVLA：P2/P3，受许可证和跨本体迁移风险约束。** 有已核验的 VILA checkpoint 装载和 G1 red-ball Isaac-Lab 桥接，但关键 checkpoint、MANO 与手部重定向网络需自行取得，且模型原生为 H1 数据，G1 抓取没有可靠性承诺。
5. **Realman：P3，暂不复现。** 它是基于私有数据验证过的 ACT/Diffusion Policy 训练配方；数据和评测均未随 StarVLA 发布，只适合未来拿到 RM-75 兼容数据后做离线训练 smoke。

已有实测还决定了训练策略：单卡 5090 上，StarVLA 4B 全参数 AdamW 已在优化器状态初始化时 OOM；因此 realRobots 主线不应以全参数微调为验收。优先运行公开/已有小模型，完成数据—checkpoint—policy server—client/mock/simulation 的闭环；只有 RoboChallenge 或 Unitree 选择冻结 VLM 或 LoRA 做 20–1,000 step 训练证据。

## 评估口径

| 层级 | 本报告中的含义 | 可作为“完成 real demo”吗 |
| --- | --- | --- |
| 数据检查 | 下载/转换、LeRobot 元数据、视频、状态与动作维度一致 | 否，只是前置条件 |
| 训练 smoke | dataloader、forward/backward、checkpoint 保存与重载 | 否，只证明训练链条 |
| 服务协议 | policy server 与 client 的 action chunk、归一化、延迟均正确 | 可作为流程证据的一部分 |
| mock/replay | 对录制轨迹或上游 mock server 连续推理并正确 POST action | 是，可称“无实体机器人的完整流程复现” |
| 仿真闭环 | 仿真器或第三方控制器消耗动作并产生可解释任务状态 | 是，但必须标为 simulation closed loop |
| 物理闭环 | 实机感知、动作、任务成功率与安全记录 | 否；当前资源不具备此验收条件 |

“5090 成功可能性”表示在先修复当前 main 的路径/配置问题后，达到报告所列验收层级的工程概率；不是算法在真实机器人的成功率。

## 项目优先级、验收与资源总览

| 排名 | 项目 | 无实体机器人时的最高诚实验收 | 5090 完成可能性 | 首次完成日历时间 | 预计 GPU 时间 | 建议新增磁盘 |
| --- | --- | --- | --- | --- | --- |
| 1 | RoboChallenge_table30v2 | 公开数据→LeRobot→100-step checkpoint→离线 self-test→上游 mock server 连续 POST | 80–90% | 1–3 天 | 2–6 h | 40–100 GB |
| 2 | UnitreeG1_WholeBody | 数据 schema/replay→1/1000-step 训练→WebSocket server→local self-test→GR00T/SONIC 仿真或 mock controller | 45–65% 完整仿真链；80% 训练/服务 smoke | 3–7 天 | 8–24 h | 120–250 GB |
| 3 | Franka | 自有数据→LeRobot v2.1→forward→冻结/LoRA checkpoint→WebSocket client mock | 70–85%（已具备合格数据）；无数据时 0% | 2–5 天 | 4–16 h | 30–120 GB + 数据 |
| 4 | EgoVLA | 获得许可资源后：真实权重 forward→G1 ZMQ server→Isaac-Lab red-ball bridge | 25–45% | 3–7 天 | 2–12 h | 80–180 GB |
| 5 | Realman | 自有 LeRobot 数据→ACT 或 DP 训练→checkpoint 重载/离线 action 回放 | 40–60%（已具备数据）；无数据时 0% | 1–3 天 | 2–10 h | 20–80 GB + 数据 |

时间是工程计划值，不是官方实测值，包含环境适配、首轮排错和结果整理。GPU 时间不含大文件下载；当前服务器仅 4 CPU 核时，视频解码、数据转换及 Isaac/仿真初始化往往比 GPU 训练更拖慢日历时间。

## 1. RoboChallenge_table30v2：首选的无机器人完整流程

### 前置性能、数据与模型

- 数据下载器指向公开 Hugging Face 数据集 `RoboChallenge/Table30v2`，按任务分片下载、拼接、解包，再转换为 LeRobot。当前 registry 已明确列出 UR5 的 `arrange_fruits`、`shred_paper`，ARX5 的 `arrange_flowers` 和 DOS-W1 的 `fold_the_clothes`；建议只从 `shred_paper` 单任务开始。
- UR5 数据契约为两路图像（`cam_global`、`cam_arm`）、7D state（6 joint + gripper）、8D action（7D EE quaternion pose + gripper）。模型回传 horizon=8 的 `(8, 8)` action chunk。
- 默认配置是 **Qwen3.5-0.8B + QwenOFT**，`action_dim=8`、`state_dim=7`、224×224、batch=2、100 step、gradient checkpointing。它是官方明确写出的“轻量 walkthrough/smoke”，不能与任务成功率或全量训练等同。
- 离线 self-test 会加载 checkpoint，在合成观测上输出 action chunk 和延迟；README 只写“hundreds-of-ms”级预期，不提供完整成功率。随后可连接 RoboChallengeInference 的 `mock_robot_server.py`，完成真实 HTTP I/O 形状与连续 action POST 验证。

### 需求与运行要求

- StarVLA `starVLA_dev` 环境、CUDA toolkit/nvcc、Hugging Face 下载权限、`accelerate`/DeepSpeed；当前脚本的 CUDA 路径是集群示例，必须改为服务器上的实际路径。
- 需克隆上游 `RoboChallengeInference` 的 `cvpr` 分支，在独立 shell 启动 mock server。它不需要机械臂，也不需要仿真图形卡。
- 训练前修复当前分支的两处相对根路径问题，确认 `starVLA`、`deployment` 与数据 registry 目录都能被定位。W&B 已默认 `disabled`，不应填写占位符密钥。
- 至少预留 40 GB；若下载多个任务、保留 raw tar、转换产物和 checkpoint，同时占用会显著增加，因此推荐 80–100 GB 临时空间。

### 工期与最小证据

| 阶段 | 预计时间 | 应保留的证据 |
| --- | --- | --- |
| 下载一个任务并转换/检查 LeRobot | 2–8 h，主要受网络和视频影响 | `info.json`、`modality.json`、一段可解码视频、维度检查日志 |
| 100-step 训练 | 0.5–2 GPU h | config、checkpoint、`dataset_statistics.json`、loss 日志 |
| local self-test | 0.5–1 h | action shape、若干次 latency、归一化 key |
| mock-server 连通 | 1–4 h | GET state/POST action 日志、连续循环截图或终端记录 |

**结论：这是当前最值得向老师展示的 real demo。** 最终表述应为“RoboChallenge 公开数据上的训练—PolicyServer—挑战协议 mock 全流程复现”，而不是“已完成 RoboChallenge 线上或实机提交”；production `job_loop` 在当前 README 仍是 TODO。

## 2. UnitreeG1_WholeBody：最完整、但不宜作为第一项目

### 前置性能、数据与模型

- 该 Example 的边界清楚：StarVLA 负责 LeRobot 数据、训练与 WebSocket policy server；PICO/XR、Unitree SDK、相机、SONIC WBC、实时平衡与急停由 GR00T-WholeBodyControl 或用户基础设施负责。
- 文档给出的 `test_sonic` 数据契约为 49 episodes、56,919 frames、50 fps、单 ego video。推荐训练目标是 **64D SONIC motion token + 7D 左手 + 7D 右手 = 78D**；状态输入配置为 72D，action horizon=8。
- 默认模型仍为 **Qwen3.5-0.8B + QwenOFT + MLP action head**，batch=1、gradient accumulation=4、1000 step、224×224。README 只证明过单 RTX 4090 的 **1-step training smoke**，并没有给出 1000-step 收敛、G1 仿真成功率或实机结果。
- 当前 fork 检查结果显示 `test_sonic` 数据、0.8B 权重均不在工作区；README 中的“local dataset already present”是作者机器叙述，不能视为本地可用资产。

### 需求与运行要求

- 最低可做路径：下载或获得兼容 LeRobot v2.1 数据，运行 metadata/parquet/video 检查，训练 checkpoint，启动 server，使用 `local_self_test.py` 与 mock controller/replay 验证 78D 分组。
- 仿真全链还需要 GR00T-WholeBodyControl、SONIC C++ deploy、MuJoCo 或其对应仿真路径、ZMQ、可能的 PICO teleop 环境。不要把这些第三方系统混入 StarVLA conda 环境，建议分环境，通过网络协议连接。
- RTX 5090 具备 RT Core 与 32 GiB VRAM，显存条件优于当前 Isaac Sim 16 GiB 最低要求；但 NVIDIA 当前“good”档仍建议 8 CPU cores、64 GB RAM、500 GB SSD。现有 4 CPU 核及约 100 GB 配额会使 Isaac/视频/资产链明显受限。[Isaac Sim 官方要求](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- 容器化 Isaac Sim 仅支持 Linux；GPU 驱动、Vulkan/图形会话和资产在线访问需要提前验证。Isaac/GR00T 仿真和 StarVLA server 若共用一张卡，需要先测显存峰值，再决定是否并行运行。

### 工期与最小证据

| 阶段 | 预计时间 | 通过条件 |
| --- | --- | --- |
| 外部栈和资产可启动 | 1–3 天 | MuJoCo/GR00T 或 mock controller 成功启动；不含 VR 设备接入 |
| 数据 schema 与回放 | 0.5–1 天 | 49 episode 或替代集的 video、state、78D action 对齐 |
| 1-step 与 100-step smoke | 2–8 GPU h | 可重载 checkpoint，`dataset_statistics.json` 与 q99 归一化完整 |
| 1000-step 短训 | 6–16 GPU h | loss 曲线、定期 checkpoint、无 NaN/维度漂移 |
| server→adapter→mock/sim | 1–2 天 | 78D 正确切分为 64+7+7，stale-action 与 clipping 记录 |

**结论：它是报告中的“系统工程加分项”，不是最稳的首个完成项。** 在没有 G1、PICO 和安全控制器的条件下，验收上限应写为“G1 数据与政策服务接入、回放/模拟控制器或第三方仿真闭环”，不能声称 G1 real demo 已跑通。

## 3. Franka：泛化接口模板，前提是先有数据

### 前置性能、数据与模型

- 示例提供 raw episode 到 LeRobot v3.0/v2.1 的转换规范、single/dual-arm registry、7D/14D action 说明和 WebSocket client 伪实现；没有附数据、公开 checkpoint、仿真任务或官方任务成功率。
- 单臂 action 为 `[x,y,z,roll,pitch,yaw,gripper]` 7D，双臂为 14D；图像数量与顺序、state/action 维度、normalization statistics 必须与训练一致。
- single-arm 默认配方是 **Qwen2.5-VL-3B + DINOv2-small + DiT-B/GR00T action head**，batch=16、100,000 step，且同时配置 COCO VLM 共训。它超出了单张 5090 可直接全参数复刻的资源边界；应改成仅 VLA 数据、冻结视觉语言骨干或 LoRA、batch=1–2、20–100 step。

### 需求、工期与结论

- 必需输入是自有 Franka episode：同步 RGB 视频、robot state、7D/14D action、task text、timestamp；还需要 `ffmpeg`、pyarrow、LeRobot 以及转换期兼容的 `datasets<4.0.0`。
- 只有在已经有 30–100 个以上质量可检查的单任务 episode 时才建议立项；否则数据采集本身已经超出“远程无机械臂复现”的范围。
- 数据已经就绪时，转换/metadata 0.5–1 天、dataloader+forward 0.5 天、冻结/LoRA 训练 4–16 GPU h、server client mock 0.5–1 天。没有数据时不估算训练时间，项目应直接标为阻塞。

**结论：Franka 不应替代 RoboChallenge 或 Unitree。** 它的正确价值是以后实验室获取 Franka 数据后，作为将 StarVLA 接到本体 action-space 的标准模板。

## 4. EgoVLA：模型/跨本体适配验证，许可证是主风险

### 前置性能、数据与模型

- StarVLA 内部实现了 EgoVLA：SigLIP-384 + Qwen2-1.5B + VILA projector + transformer trajectory decoder。作者声明 public `ckpt-6720` 的 vision/LLM/projector/decoder state-dict 均已逐项匹配，并能端到端输出 `(T,48)` 动作。
- 每一步是双手 48D camera-frame action：每只手 3D wrist translation、6D rot6d、15D MANO pose；默认 horizon=30、图像 384×384、配置为 5000 step。
- G1 red-ball 路径会把 48D action 经 MANO、手部 actuation net 与 IK 映射到 G1。README 明确说明 checkpoint 来自 H1 数据，G1 抓取可靠性取决于输出，可靠抓取需要 G1 数据微调；因此“可加载且能关节化”不等于“会抓取”。

### 需求、工期与结论

- 必须自行取得：EgoVLA Release checkpoint、研究许可的 MANO left/right 模型、`hand_actuation_net.pth`、EgoVLA Release checkout；另外安装 `pinocchio`、`smplx`、`chumpy`。这些关键资源均不在当前仓库。
- 若只做模型/协议 smoke：先通过许可和下载，1–2 天完成真实权重 forward 与 ZMQ server；再配置外部 GR00T-WBC-Bridge + Isaac-Lab G1 red-ball，额外 2–5 天。GPU 计算本身不是瓶颈，许可证、外部版本兼容和手部标定才是。
- 1.5B 模型在 5090 上可尝试 bf16 推理和冻结/LoRA 训练；不要直接采用 384px、5000-step、`freeze_modules: ''` 的全参配置作为首轮计划。

**结论：只有在资源授权已确认时才升为 P2。** 否则它会消耗时间但无法形成可复查的完整结果，优先级应低于公开数据的 RoboChallenge 和 Unitree。

## 5. Realman RM-75：当前只有训练配方

- 该目录提供 ACT 与 Diffusion Policy 两套烟雾训练 YAML：两路 RGB（`cam0_rgb`、`cam1_rgb`）、8D state、8D action（7 joint delta + 绝对 gripper）；ACT chunk=50，DP horizon=16。
- 两者均要求“替换为自己的兼容 LeRobot dataset”，没有分发数据、权重、PolicyServer client、仿真器、mock server 或成功率。因此它不能独立构成当前可汇报的 demo。
- 一旦以后有 RM-75 数据，ACT/DP 的 ResNet18 级视觉骨干对 5090 很轻，5,000-step smoke 约 2–10 GPU h，20–80 GB 加数据即可；但这只能证明经典 imitation policy 流程，而不是 StarVLA 的视觉语言模型能力。

**结论：P3，保留为后续对照实验。** 若老师要求“多个官方 real demo”，应先用前三项形成可对比的三种证据类型，而不是花时间在未公开资产的 Realman 上。

## 统一运行前置清单

1. **先修 main 路径。** 当前重组后，Franka、RoboChallenge、Unitree 等 shell 的仓库根目录推导存在层级回归；每个项目先跑 `bash -n`、`python --help/import` 与一个最小启动命令。
2. **分离运行环境。** StarVLA 训练/推理环境、LeRobot 转换环境、RoboChallenge mock 环境、GR00T/Isaac/SONIC 环境分开创建；只以 WebSocket/ZMQ/文件数据交接，避免互相覆盖 PyTorch、CUDA、ffmpeg 或系统库。
3. **锁定数据契约。** 每次训练前保留 `info.json`、`modality.json`、state/action keys、camera 顺序、FPS、normalization statistics 和一个 episode 的 shape dump。
4. **先压低模型风险。** 0.8B QwenOFT 用于 RoboChallenge/Unitree 首轮；Franka 3B 和 EgoVLA 1.5B 都以冻结/LoRA 或只推理开始。不要在 32 GiB 上初始化 4B 全参 AdamW。
5. **policy server 先离线。** 先以 synthetic observation，再以 recorded episode、mock controller，最后才启动第三方仿真；每一步记录 action shape、latency、unnorm key、clipping 与 stale-action 行为。
6. **空间滚动清理。** 每项目结束后清理 raw tar、Hugging Face cache 副本、解压资产和全量视频；保留 config、环境 lock、数据元数据、checkpoint 哈希、JSON、少量视频与日志。100 GB 可完成 RoboChallenge 单任务；Unitree/Isaac 建议 200 GB 以上临时盘。

## 建议向老师提交的 real demo 组合

- **项目 A（主项目）：** RoboChallenge Table30v2 单任务。证据覆盖公开数据、短训、model load、离线 action、mock robot HTTP 协议。
- **项目 B（系统项目）：** Unitree G1 WholeBody。证据覆盖 78D action 数据契约、训练、WebSocket、adapter、mock/replay 或第三方仿真。
- **项目 C（本体迁移模板）：** Franka 数据接入的 schema/forward/server mock；仅在真实数据已得到时升级到短训。

这样可向老师清楚展示：StarVLA 不只在仿真 benchmark 里跑分，也已在三种不同 real-robot 形态中复现了可审计的部署链路；同时严格说明当前没有物理机器人，未声称未经验证的实体成功率。
