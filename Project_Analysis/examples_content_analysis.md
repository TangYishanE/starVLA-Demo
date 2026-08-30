# StarVLA `examples/` 内容分析

## 1. 文档目的

本文档整理 StarVLA 项目 `examples/` 目录中各示例的用途、组织方式、成熟度和相互关系，重点回答以下问题：

1. 每个一级文件夹对应什么机器人平台、仿真基准或模型扩展？
2. 哪些示例用于训练，哪些只用于评测？
3. 哪些内容与真实机器人部署直接相关？
4. 阅读和复用这些示例时需要注意哪些接口与版本问题？

## 2. 分析所依据的版本

当前工作区信息：

```text
仓库：D:\ProgrammingProjects\StarVLA_Demo\starVLA
当前分支：starVLA_dev
当前 commit：5ce9e2e（2026-05-08）
origin：TangYishanE/starVLA
upstream：starVLA/starVLA
```

本地 Git 对象中记录的 `upstream/starVLA_dev` 为 `48e5881`（2026-08-28），比当前检出分支多 52 个 commit。当前检出版本使用扁平目录结构；新版官方分支已经按示例性质重组：

```text
examples/
├── simBenchmarks/
├── realRobots/
├── modelExtensions/
└── human2robots/
```

本文首先分析当前工作区中实际可见的 `examples/`，最后给出新版官方目录映射。

## 3. 当前 `examples/` 总览

当前版本包含以下一级目录：

```text
examples/
├── Behavior/
├── calvin/
├── CoTrainVLM/
├── DOMINO/
├── Franka/
├── Gemma4/
├── LIBERO/
├── LIBERO-plus/
├── modelExtensions/
├── Robocasa_365/
├── Robocasa_tabletop/
├── RoboChallenge_table30v2/
├── Robotwin/
├── SimplerEnv/
└── VLA-Arena/
```

可以分为三大类：

| 类别 | 目录 | 主要用途 |
| --- | --- | --- |
| 仿真基准 | `LIBERO`、`LIBERO-plus`、`SimplerEnv`、`Robotwin`、`Robocasa_tabletop`、`Robocasa_365`、`calvin`、`Behavior`、`DOMINO`、`VLA-Arena` | 训练、复现实验和策略评测 |
| 真实机器人/真实平台 | `Franka`、`RoboChallenge_table30v2` | 数据接入、训练、Policy Server、机器人或平台接口适配 |
| 模型与训练扩展 | `CoTrainVLM`、`Gemma4`、`modelExtensions/NeuralVLA` | 联合训练、替换 VLM 骨干、增加历史状态等能力 |

## 4. 仿真与评测基准

### 4.1 LIBERO

路径：[`examples/LIBERO/`](../starVLA/examples/LIBERO/)

`LIBERO` 是 StarVLA 当前最完整、最标准的入门示例，覆盖四套经典任务：

- LIBERO-Spatial
- LIBERO-Object
- LIBERO-Goal
- LIBERO-Long / LIBERO-10

它包含完整的训练与评测链路：

```text
下载或准备 LeRobot 数据
→ 注册数据 mixture 和 modality
→ 选择 StarVLA action head
→ 启动训练
→ 启动 GPU Policy Server
→ 在独立 LIBERO 环境中运行客户端
→ 汇总任务成功率
```

主要子目录：

- `train_files/`：训练 YAML、数据配置和启动脚本。
- `eval_files/`：LIBERO 环境评测程序、Policy Server 启动脚本和模型接口。

适用场景：

- 初次理解 StarVLA 的训练配置。
- 验证新 action head 或 VLM backbone。
- 学习 Policy Server 与环境客户端之间的通信方式。
- 在接入真实机器人之前进行端到端流程验证。

成熟度：高，是最适合首先复现的示例。

### 4.2 LIBERO-plus

路径：[`examples/LIBERO-plus/`](../starVLA/examples/LIBERO-plus/)

`LIBERO-plus` 用于测试模型的零样本泛化和分布偏移鲁棒性。它通常直接使用只在 LIBERO 上训练的 checkpoint，不重新训练模型。

主要扰动维度：

- Camera：相机位置或视角变化。
- Robot：机器人外观或构型变化。
- Language：指令表达变化。
- Light：光照变化。
- Background：背景变化。
- Noise：观测噪声。
- Layout：物体与场景布局变化。

适用场景：评估模型是否只是记住 LIBERO 的视觉分布，还是具有一定跨分布泛化能力。

成熟度：以评测为主，不包含独立训练流程。

### 4.3 SimplerEnv

路径：[`examples/SimplerEnv/`](../starVLA/examples/SimplerEnv/)

`SimplerEnv` 主要用于 Google Robot、WidowX 等 OXE 体系策略的仿真评测，同时也包含基于 OXE、Bridge、RT-1 等数据训练 StarVLA 的示例。

主要内容：

- SimplerEnv 环境安装与最小验证。
- Policy Server 与仿真客户端分环境运行。
- Google Robot/WidowX 任务评测。
- OXE 数据准备和训练配置。
- 针对多个 checkpoint、端口、任务和种子的自动批量评测。
- 自动收集客户端日志并生成结果汇总。

`eval_files/auto_eval_scripts/` 特别适合大规模 checkpoint sweep，但其中脚本通常包含 GPU、端口和目录约定，使用前需要检查本机环境。

成熟度：较高，训练和评测都较完整。

### 4.4 Robotwin

路径：[`examples/Robotwin/`](../starVLA/examples/Robotwin/)

该目录对接 RoboTwin 2.0 双臂机器人仿真基准。

主要特点：

- 覆盖约 50 个双臂操作任务。
- 包含 Easy 和 Hard 设置。
- 支持 clean demonstrations 和 randomized demonstrations。
- 提供 StarVLA 与其他 VLA 模型的结果对比。
- 同时包含 `train_files/` 和 `eval_files/`。

适用场景：

- 双臂 VLA 训练和评测。
- 大规模多任务数据训练。
- 检查双臂 action ordering、相机顺序和 action chunk 执行。

成熟度：较高，但环境、数据和评测开销明显大于 LIBERO。

### 4.5 Robocasa_tabletop

路径：[`examples/Robocasa_tabletop/`](../starVLA/examples/Robocasa_tabletop/)

该目录对接 `robocasa-gr1-tabletop-tasks`，面向 NVIDIA GR1 tabletop 仿真任务。

主要内容：

- GR1 tabletop 环境安装。
- 官方或 StarVLA checkpoint 下载。
- GPU Policy Server。
- RoboCasa/GR1 仿真客户端。
- 训练数据和训练脚本。

注意：它依赖的是特定的 RoboCasa GR1 fork，并不是官方 RoboCasa 365-task 版本。

成熟度：较完整，但具有特定平台依赖。

### 4.6 Robocasa_365

路径：[`examples/Robocasa_365/`](../starVLA/examples/Robocasa_365/)

该目录面向官方 RoboCasa 365-task 基准，机器人为 PandaOmron 移动机械臂。

官方 walk-through 包含：

1. 单独创建 RoboCasa 仿真环境。
2. 下载场景资产和 LeRobot v2.1 数据。
3. 使用一个任务进行 100-step 训练冒烟测试。
4. 启动 StarVLA Policy Server。
5. 在 MuJoCo/RoboCasa 环境中评测。

与 `Robocasa_tabletop` 的区别：

| 对比项 | `Robocasa_tabletop` | `Robocasa_365` |
| --- | --- | --- |
| 上游项目 | GR1 tabletop fork | 官方 RoboCasa |
| 机器人 | NVIDIA GR1 | PandaOmron |
| 任务范围 | Tabletop task 集合 | 365 个厨房任务 |
| 目的 | 复现 GR1 tabletop 结果 | 官方 RoboCasa 端到端示例 |

成熟度：较高，文档结构清晰。

### 4.7 calvin

路径：[`examples/calvin/`](../starVLA/examples/calvin/)

该目录用于 CALVIN 长时序语言条件操作基准。

主要内容：

- 将 CALVIN 数据转成 LeRobot 格式。
- 安装或复制 `modality.json`。
- 注册数据 mixture。
- 训练 StarVLA。
- 评测连续多个语言子任务。

CALVIN 与 LIBERO 的主要区别是更强调长时序和连续子任务完成能力。常见指标包括连续完成 1～5 个任务的成功率和平均任务链长度。

成熟度：包含训练和评测流程。

### 4.8 Behavior

路径：[`examples/Behavior/`](../starVLA/examples/Behavior/)

该目录对接 BEHAVIOR-1K/OmniGibson，按照 2025 BEHAVIOR Challenge 结构组织约 50 个完整家庭任务。

主要内容：

- BEHAVIOR-1K 和 OmniGibson 环境安装。
- StarVLA 与 Behavior 两套环境之间的服务通信。
- 串行或并行评测脚本。

官方 README 明确标记为 `Under construction`。

硬件注意事项：OmniGibson 对光线追踪能力有要求。README 特别提醒 A100/H100 缺少 RT Core，可能出现低分辨率或 segmentation fault。

成熟度：仍在建设，不建议作为第一个复现目标。

### 4.9 DOMINO

路径：[`examples/DOMINO/`](../starVLA/examples/DOMINO/)

DOMINO 是动态操作基准，机器人需要响应运动物体和随时间变化的场景。

主要特点：

- 35 个动态任务。
- 同一策略覆盖所有任务。
- 场景中可能出现运动目标和惩罚事件。
- 主要指标为 Success Rate（SR）和 Manipulation Score（MS）。

与 LIBERO 的区别：LIBERO 主要测试静态桌面操作，DOMINO 更强调感知—决策—控制闭环的动态响应能力。

成熟度：包含训练和评测流程。

### 4.10 VLA-Arena

路径：[`examples/VLA-Arena/`](../starVLA/examples/VLA-Arena/)

VLA-Arena 是综合泛化和安全评测基准。

主要特点：

- 4 个评测领域。
- 11 个 task suite。
- L0、L1、L2 三档难度。
- 每个等级包含多个任务。
- 除成功率外，还评估安全相关的 constraint cost。

适用场景：

- 比较不同数据规模带来的泛化差异。
- 评估模型在复杂指令和场景变化下的表现。
- 研究成功率与安全约束之间的权衡。

成熟度：包含数据准备、训练与评测。

## 5. 真实机器人与真实平台示例

### 5.1 Franka

路径：[`examples/Franka/`](../starVLA/examples/Franka/)

Franka 是当前工作区中最明确的真实机器人开发示例，覆盖数据准备、训练和推理接口。

整体流程：

```text
Franka 示教数据
→ LeRobot v3.0/v2.1
→ modality 和 DataConfig
→ 单臂或双臂训练
→ StarVLA Policy Server
→ WebSocket 推理客户端
→ 用户实现的相机与机器人控制器
```

动作空间：

单臂 7D：

```text
[x, y, z, roll, pitch, yaw, gripper]
```

双臂 14D：

```text
[left position 3, left rotation 3, left gripper,
 right position 3, right rotation 3, right gripper]
```

子目录：

- `franka2lerobot/`：介绍原始数据到 LeRobot 的转换要求。
- `train_files/`：单臂/双臂训练配置、数据注册与启动脚本。
- `eval_files/`：Policy Server 启动脚本和推理客户端模板。

重要限制：

- 原始数据转换器没有随示例提供，需要针对用户的数据格式实现。
- 相机读取是伪代码。
- `env.reset()`、`env.step()` 和 `env.get_obs()` 尚未实现。
- 不包含 Franka SDK、控制器、碰撞检测或急停实现。
- 当前 Franka 文档仍按旧服务器接口解释 action 反归一化，不能直接用于当前 Policy Server。

因此该目录的准确定位是“真实机器人接入模板”，而不是下载后即可运行的 Franka 硬件 Demo。

### 5.2 RoboChallenge_table30v2

路径：[`examples/RoboChallenge_table30v2/`](../starVLA/examples/RoboChallenge_table30v2/)

该目录对接 RoboChallenge Table30v2 真实平台，重点是 UR5 机器人训练数据和挑战平台通信协议。

主要结构：

```text
RoboChallenge_table30v2/
├── train_files/
│   ├── 数据下载/转换
│   ├── data_registry
│   └── 训练配置与脚本
└── eval_files/
    ├── 本地模型自测
    ├── mock server 测试
    └── 平台协议适配器
```

典型 I/O：

- 状态读取：6D UR5 joints + 1D gripper。
- 动作下发：7D EEF pose + 1D gripper。
- 模型以 action chunk 形式输出多步动作。

新版官方流程将其分为：

1. 无网络本地自测。
2. 对接上游 mock robot server。
3. 对接真实比赛平台。

其中第 3 步 production 部署仍标记为 TODO。因此该示例适合学习真实平台协议适配，但尚不是完整的生产部署程序。

## 6. 模型与训练方法扩展

### 6.1 CoTrainVLM

路径：[`examples/CoTrainVLM/`](../starVLA/examples/CoTrainVLM/)

该目录演示 VLA 机器人数据与普通 VLM 图文数据的联合训练。

数据组合：

```text
LeRobot 机器人轨迹
+
QwenVL/LLaVA conversations 格式的图文数据
→ StarVLA multi-objective co-training
```

主要目标：

- 保持通用视觉和语言理解能力。
- 减轻只在机器人轨迹上训练导致的 VLM 能力退化。
- 同时优化 VLA action loss 和 VLM language loss。

主要内容：

- VLM JSON 数据格式。
- VLM 数据集注册。
- `vlm_data` 和 `vla_data` 双数据配置。
- VLA/VLM loss 权重设置。
- 联合训练脚本。

它不是新的机器人环境，而是一种训练策略示例。

### 6.2 Gemma4

路径：[`examples/Gemma4/`](../starVLA/examples/Gemma4/)

该目录演示用 Google Gemma 4 E2B 替换 Qwen-VL 作为 StarVLA 的视觉语言骨干。

主要内容：

- Gemma 4 模块和框架冒烟测试。
- Gemma4 + PI action head。
- Gemma4 + GR00T action head。
- LIBERO 四套任务训练与本地评测。
- Slurm/HPC 多 GPU 训练脚本。

它属于模型架构扩展，不是机器人或仿真环境本身。

### 6.3 modelExtensions/NeuralVLA

路径：[`examples/modelExtensions/NeuralVLA/`](../starVLA/examples/modelExtensions/NeuralVLA/)

该目录为 NeuroVLA 增加固定长度的历史机器人状态支持。

核心思路：

```python
state_indices = list(range(-16, 0))
```

模型每次预测不仅接收当前状态，还接收过去 16 个时间步的 state history。

需要同时修改：

- DataConfig 中的状态索引。
- 数据加载器输出形状。
- LIBERO 环境评测时的状态历史缓存。
- 模型的 state dimension。

关键约束：训练和评测必须使用相同的历史长度、状态顺序和拼接方式，否则模型虽然可能正常运行，但输入语义会错位。

## 7. 示例目录的通用文件结构

多数 benchmark 使用如下组织方式：

```text
example_name/
├── README.md
├── train_files/
│   ├── data_registry/
│   │   └── data_config.py
│   ├── modality.json
│   ├── training_config.yaml
│   └── run_train.sh
└── eval_files/
    ├── model2xxx_interface.py
    ├── eval_xxx.py
    ├── run_policy_server.sh
    └── run_eval.sh
```

### 7.1 `README.md`

通常说明：

- 上游 benchmark 安装方式。
- checkpoint 和数据下载。
- 训练命令。
- 评测命令。
- 官方复现实验结果。

### 7.2 `train_files/`

负责训练侧配置，常见文件包括：

- `data_registry/data_config.py`：定义机器人 embodiment、相机、state、action、language key 和归一化模式。
- `modality.json`：将 LeRobot 原始字段映射为 StarVLA 的逻辑 modality。
- `*.yaml`：模型、数据集、action head、batch size 和 trainer 配置。
- `run_*.sh`：封装 `accelerate launch`、GPU 数量、checkpoint 路径和实验名称。

### 7.3 `eval_files/`

负责环境或机器人客户端：

- 采集环境图像和状态。
- 构造 StarVLA `examples` 请求。
- 调用 WebSocket Policy Server。
- 解析服务器返回的 action chunk。
- 将动作转换成环境或机器人控制接口。
- 记录视频、成功率和日志。

### 7.4 `model2xxx_interface.py`

这类文件是 Policy Server 与具体环境之间最关键的适配层，一般负责：

```text
环境 observation
→ 图像排列/裁剪/色彩转换
→ state 拼接
→ language prompt
→ StarVLA 请求
→ action chunk
→ 环境动作格式
```

真实机器人部署时，通常需要参考并重写这一层。

### 7.5 `auto_eval_scripts/`

常见于 SimplerEnv 等需要大量组合实验的 benchmark，用于：

- 扫描多个 checkpoint。
- 为不同 GPU 和任务分配端口。
- 启动多个 Policy Server。
- 运行多个随机种子。
- 收集并汇总客户端日志。

使用前应检查脚本中的硬编码路径、GPU 编号、端口和环境名称。

## 8. Policy Server 的公共关系

这些 example 大多共享同一套部署架构：

```text
                   GPU / StarVLA 环境
Checkpoint ──────> Policy Server
                        ▲
                        │ WebSocket + msgpack_numpy
                        ▼
环境/机器人 ─────> benchmark client / robot adapter
```

服务端入口：

```text
deployment/model_server/server_policy.py
```

客户端：

```text
deployment/model_server/tools/websocket_policy_client.py
```

当前服务器的主要职责：

1. 从 checkpoint 加载 StarVLA framework。
2. 接收 `examples`、可选 `unnorm_key` 和推理参数。
3. 运行模型得到 normalized actions。
4. 使用训练时的 DataConfig 和统计量在服务端反归一化。
5. 返回：

```python
{
    "status": "ok",
    "data": {
        "actions": actions  # [B, T, D]，已经反归一化
    }
}
```

客户端仍负责：

- 图像和 state 的环境适配。
- action chunk 调度。
- sticky gripper 或 action ensemble。
- 控制频率。
- 安全裁剪和超时检查。
- 最终的环境或硬件动作执行。

## 9. 版本与一致性风险

### 9.1 旧客户端可能重复反归一化

新版 Policy Server 返回的是已经反归一化的 `actions`。部分旧 example 仍读取 `normalized_actions`，或者收到 `actions` 后继续调用本地 `unnormalize_actions()`。

这会导致二次反归一化，在真实机器人上可能产生越界动作。迁移客户端时应以 `deployment/model_server/README.md` 和 `policy_wrapper.py` 的当前接口为准。

### 9.2 相机数量和顺序必须与训练一致

多个示例使用类似名称：

```text
base_view
ego_view
wrist_image
primary_image
external_camera
```

这些名称不能只按字面替换。必须验证：

- 图像数量。
- 相机排列顺序。
- RGB/BGR。
- resize、crop 和分辨率。
- 是否使用历史帧。

### 9.3 state/action schema 必须形成单一事实来源

需要同时保持一致：

```text
LeRobot parquet 字段
↔ meta/modality.json
↔ DataConfig state_keys/action_keys
↔ YAML state_dim/action_dim
↔ Policy Server normalization statistics
↔ 客户端动作分组
↔ 机器人控制器语义
```

只要其中一个环节顺序或维度不同，程序可能不报错，但机器人行为会错误。

### 9.4 Shell 脚本通常包含环境假设

示例脚本中可能出现：

- 作者本机 Python 路径。
- 特定 NCCL 网络接口。
- 固定 GPU 数量。
- 固定 W&B entity。
- 固定数据和 checkpoint 路径。
- Slurm/HPC 环境变量。

因此 `run_*.sh` 应视为命令模板，运行前需要逐行检查。

## 10. 新版官方目录映射

新版官方 `upstream/starVLA_dev` 已将示例重新分类。

### 10.1 `examples/simBenchmarks/`

包含：

```text
Behavior
DOMINO
LIBERO
LIBERO-plus
MetaWorld
RoboDojo
Robocasa_365
Robocasa_tabletop
Robotwin
SimplerEnv
VLA-Arena
VLN-CE
calvin
```

相较当前工作区，新增了 MetaWorld、RoboDojo 和 VLN-CE。

### 10.2 `examples/modelExtensions/`

包含：

```text
CoTrainVLM
DiscreteDiffusion
Gemma4
MiniCPM
NeuralVLA
```

相较当前工作区，增加了 DiscreteDiffusion 和 MiniCPM 等模型扩展示例。

### 10.3 `examples/realRobots/`

包含：

```text
EgoVLA
Franka
Realman
RoboChallenge_table30v2
UnitreeG1_WholeBody
```

其中：

| 目录 | 动作或平台 | 当前定位 |
| --- | --- | --- |
| `Franka` | 单臂 7D、双臂 14D | 数据、训练和部署模板，硬件层需自行实现 |
| `UnitreeG1_WholeBody` | 64D SONIC motion token + 双手各 7D，共 78D | 对接 GR00T-WBC/SONIC 的完整工作流脚手架 |
| `Realman` | 7 个机械臂关节 + 1 个夹爪，共 8D | VM4A ACT/DP 训练 recipe，数据集不公开 |
| `RoboChallenge_table30v2` | UR5/挑战平台 | 本地和 mock 流程可用，production 部署仍为 TODO |
| `EgoVLA` | 双手 48D camera-frame action | G1/GR00T-WBC 桥接，依赖额外 checkpoint 和许可证资源 |

## 11. 推荐阅读和实践顺序

### 11.1 理解 StarVLA 基础训练与评测

```text
根 README
→ docs/starVLA_guideline.md
→ examples/LIBERO
→ deployment/model_server
```

LIBERO 最适合验证环境安装、数据读取、模型训练和 Policy Server 是否形成闭环。

### 11.2 理解多任务和泛化

```text
LIBERO-plus
→ SimplerEnv/OXE
→ RoboTwin
→ VLA-Arena
→ DOMINO
```

### 11.3 理解模型扩展

```text
CoTrainVLM
→ Gemma4
→ NeuralVLA
```

### 11.4 接入真实机器人

```text
LIBERO 端到端闭环
→ Policy Server 当前接口
→ Franka action/data 示例
→ 选择相近的 realRobots 示例
→ 定义自己的 modality 和 DataConfig
→ replay/mock/simulation
→ 低速、限幅真实机器人测试
```

真实机器人部署不应直接从完整 action chunk 全速执行开始。推荐验证顺序：

1. 静态 schema 检查。
2. 单条录制轨迹 replay。
3. Policy Server synthetic observation 冒烟测试。
4. Mock controller 检查 action 分组、范围和频率。
5. 仿真或机器人断动力 dry-run。
6. 真实机器人低速、单步执行。
7. 启用小范围 action chunk。
8. 最后才启用完整闭环。

## 12. 总结

StarVLA 的 `examples/` 并不是一组相互独立的 Demo，而是围绕同一个核心架构组织的多种适配层：

```text
数据集/benchmark/机器人差异
        ↓
modality + DataConfig + YAML
        ↓
统一 StarVLA framework
        ↓
统一 Policy Server
        ↓
benchmark client 或 robot adapter
```

其中：

- `LIBERO` 是最标准的训练和评测参考。
- `SimplerEnv`、`Robotwin`、RoboCasa、CALVIN 等展示不同 benchmark 的数据和客户端适配。
- `CoTrainVLM`、`Gemma4`、`NeuralVLA` 展示模型和训练策略扩展。
- `Franka` 和 `RoboChallenge_table30v2` 是当前工作区中与真实机器人最相关的入口。
- 新版官方分支进一步加入 Unitree G1、Realman 和 EgoVLA，但这些目录仍依赖用户或第三方提供底层机器人控制与安全系统。

复用示例时，最重要的不是复制脚本，而是保证训练数据、归一化、相机顺序、state/action schema 和部署控制器之间严格一致。
