# Realman (RM-75) 无真机复现执行计划

**目标项目：** `examples/realRobots/Realman`，在**无实体 Realman RM-75 机器人**条件下复现其「数据契约 → 数据装载 → ACT/DiffusionPolicy 训练 → checkpoint 重载 → 合成观测推理」链路。  
**运行边界：** 远程单张 RTX 5090 32 GiB（共享）、无实体机器人、无遥操作采集设备。  
**本计划的完成定义：** 得到一份符合 8D state / 8D action / 双相机契约的 LeRobot v2.1 数据集；ACT 与 DiffusionPolicy 两个 checkpoint 训练与重载通过；对合成观测输出形状正确的 `(T, 8)` 动作块并反归一化为有限值。

> 这是一份「数据—训练—重载—推理」的无真机复现计划。Realman 是 **VM4A 小基线**（ACT + Diffusion Policy，ResNet-18 视觉编码器），不是 VLA 大模型，且本示例**只有 `train_files/`，没有任何 `eval_files/`、`step3_deployment/` 或 `model2realman_interface.py`**。因此本计划**没有**真机闭环、没有挑战赛 mock 协议、没有动作分片（64/7/7 之类）；最多验证到「在合成观测上，ACT/DP checkpoint 能输出语义正确的 8D 动作并正确反归一化」。数据若走合成路线，动作**不具任务语义**。

## 0. 计划总览

### 0.1 固定的首轮配置

| 项目 | 首轮选择 | 原因 |
| --- | --- | --- |
| 数据 | 三条路线（见 P1）：首选 **B 公开数据 `nvidia/PhysicalAI-Robotics-Manipulation-SingleArm`**（Franka 7-DoF，8D 关节动作子集）；A 内部数据若可得更好；C 合成数据兜底 | 本 clone 无 `playground/`，README 声明数据集「不随发」；已确认存在 7-DoF 关节空间的公开匹配数据 |
| 机器人/动作 | `realman_rm75_delta_joints`，8D = 7 joint delta + 1 绝对 gripper | registry 已注册的唯一类型，与 YAML 一致 |
| 框架 | `ACT`（chunk=50）+ `DiffusionPolicy`（horizon=16）两个 VM4A 基线 | 本示例的两种 recipe；ResNet-18，显存最小 |
| 模型配置 | 224×224 双相机，`BATCH=1`（ACT）/`BATCH=1`（DP），`MAX_STEPS=100` | 先验证工程闭环；100 step 不是性能训练 |
| 评测 | 自写 in-process self-test：`baseframework.from_pretrained` + `PolicyNormProcessor` | 无模型服务/协议依赖；验证 checkpoint、归一化、图像/状态输入、动作反归一化 |
| 不做的内容 | 真实 RM-75 执行、遥操作采集、任务成功率、WebSocket 策略服务 | 超出「无真机 + 本示例无部署层」边界 |

### 0.2 里程碑、时间与空间

| 阶段 | 主要工作 | 日历时间 | GPU 时间 | 峰值新增空间 | 可判定的通过条件 |
| --- | --- | ---: | ---: | ---: | --- |
| P0 | 环境/依赖检查 + 6 处代码缺口补丁 | 0.5–1 天 | 0 | <1 GB | import、CUDA、`lerobot`/`diffusers`、脚本路径正确 |
| P1 | 数据获取（三路线，见 §3） | 0.5–1 天 | 0 | <2 GB | 数据集 schema 与 8D/8D 契约逐项一致 |
| P2 | dataloader gate + 双 forward gate + ACT/DP 各 100-step 训练 | 2–4 h | ~1 h | +2–5 GB | 两个 checkpoint 与 `dataset_statistics.json` 生成 |
| P3 | checkpoint 重载 + in-process self-test（ACT/DP） | <1 h | <0.5 h | 可忽略 | 连续输出 `(T,8)` 动作，反归一化有限、无 NaN |
| P4 | 结果归档 | 1–2 h | 0 | <1 GB | 配置、日志、哈希、JSON 齐全 |

**首轮总计：1–2 天、约 1–2 GPU 小时、建议服务器空闲空间 ≥ 10 GB。** 这是三个 realRobots 示例里**最轻**的一个：无 VLM 权重下载（ResNet-18 ImageNet 权重仅 ~45 MB，走 torchvision 缓存），数据集合成可 <1 GB，训练峰值显存 ~2–3 GB——甚至可在**本地 RTX 4070 Ti 12GB** 上完整跑通（DeepSpeed 在 Windows 不可用，需 `STARVLA_DISABLE_DEEPSPEED=1` + 单卡 accelerate 配置）。

## 1. 预期输入、输出与数据契约

### 1.1 必须取得/准备的内容

| 资产 | 来源 | 首轮范围 | 说明 |
| --- | --- | --- | --- |
| Realman 数据集 | **✅ 已选定：`nvidia/PhysicalAI-Robotics-Manipulation-SingleArm` → `panda-open-drawer`**（§3 路线 B） | 1273 ep / 154256 帧，LeRobot **v2.1**，无需转换 | 公开 CC-BY-4.0 可商用；合成数据降为兜底 |
| ResNet-18 权重 | `torchvision` ImageNet 预训练 | 自动下载，或 `null`/`false` 走随机初始化 | ~45 MB；离线 smoke 用随机初始化避免下载 |

### 1.2 数据契约（必须逐项确认，来源 = `data_config.py` + `train_realman_{act,dp}.yaml`）

**state 8D，两段拼接（parquet 单列 `observation.state`）：**

| 原始 parquet 键 | 子键（start:end） | 维度 | 归一化 |
| --- | --- | --- | --- |
| `observation.state` | `state.joints`(0:7) + `state.gripper`(7:8) | 7 + 1 | joints=mean_std，gripper=min_max |

**action 8D（parquet 单列 `action`，存储**绝对**关节目标）：**

| 原始 parquet 键 | 子键（start:end） | 维度 | 归一化 | 说明 |
| --- | --- | --- | --- | --- |
| `action` | `action.delta_joints`(0:7) + `action.gripper_close`(7:8) | 7 + 1 | joints=mean_std，gripper=min_max | delta 由训练期 `action_mode: delta` 从 `action[0:7] - state.joints` 现算，gripper 保持绝对 |

> ⚠️ 上表是 Realman **默认**契约（parquet 存**绝对**目标 + 训练期转 delta）。**本计划选用的 NVIDIA `panda-open-drawer` 的 action 列本身已是 delta**（`panda_jointN_delta_pos`），故必须关闭 YAML 的 `action_mode`（§3 路线 B 适配点 1），否则双重差分。

其他关键契约：

| 项 | 约定 | 验收检查 |
| --- | --- | --- |
| 视频 | 双路 `video.cam0_rgb`、`video.cam1_rgb`，resize 224×224 | 双相机、可解码、顺序与 `image_keys` 一致 |
| 语言 | `annotation.human.action.task_description`（task_index → tasks.jsonl） | modality.json annotation 用**扁平键** `human.action.task_description` |
| 归一化 | joints `mean_std`、gripper `min_max`（非 q99、非全 min_max） | `dataset_statistics.json` 中 mean/std、min/max 可读 |
| embodiment | `EmbodimentTag.NEW_EMBODIMENT` | 反归一化键 = `new_embodiment` |
| ACT 动作窗 | `chunk_size=50`，`action_indices=list(range(50))`，`n_obs_steps=1` | 训练/推理一致 |
| DP 动作窗 | `horizon=16`，`action_indices=list(range(16))`，`n_obs_steps=2`，`n_action_steps=8` | 训练/推理一致 |
| 模型输出 | `predict_action` 返回**归一化** `normalized_actions`，shape `(B, T, 8)` | P3 反归一化后 `(T,8)` 有限 |

## 2. P0：前置环境与当前代码缺口

### 2.1 环境检查（不下载、不训练）

在 `starVLA` 仓库根目录执行，保存到 `Project_Analysis/evidence/realman/00_preflight/`：

```bash
git rev-parse HEAD && git status --short
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h .
conda env list
```

激活 StarVLA 环境后执行依赖探针（**Realman 特有**：ACT 需要可选依赖 `lerobot`，DP 需要 `diffusers`）：

```bash
python - <<'PY'
import torch, cv2, pyarrow, numpy
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))
try:
    import lerobot; print('lerobot', lerobot.__version__)
except ImportError as e:
    print('lerobot MISSING ->', e)
try:
    import diffusers; print('diffusers', diffusers.__version__)
except ImportError as e:
    print('diffusers MISSING ->', e)
import torchvision; print('torchvision', torchvision.__version__)
PY
```

通过标准：RTX 5090 可见、可用空间 ≥ 10 GB；`lerobot`（ACT 必需）与 `diffusers`（DP 必需）可 import。若 `lerobot` 缺失，ACT 实例化会抛 `ImportError`（`ACT.py` 顶部有显式 guard），必须先补装到当前环境。

### 2.2 当前代码缺口（必须处理）

| 文件/位置 | 现状 | 处理 |
| --- | --- | --- |
| `train_files/data_registry/data_config.py:111,122` | `DATASET_NAMED_MIXTURES` 两处 `<your_dataset>` 占位符 | 替换为实际数据集目录名（A/C 路线都要做） |
| `train_files/train_realman_dp.sh:24` | `data_mix=${DATA_MIX:-realman_example}` —— 默认指向 **ACT 的 mixture**（动作窗 50），与 `train_realman_dp.yaml:54` 的 `realman_example_dp` 冲突 | 改为 `realman_example_dp`。否则 DP 用 50-step 窗（而非 16），短 episode 会因采样失败而训练不了，长 episode 则是浪费加载 |
| `train_realman_act.sh:64-65` / `train_realman_dp.sh:64-65` | `--config_file starVLA/config/deepseeds/deepspeed_zero2.yaml` + `--num_processes ${NUM_PROCESSES:-4}` 默认 4 卡 | 单卡共享 5090 上设 `NUM_PROCESSES=1`；本地 4070 Ti 则改用 UMI4Pretraining 的 `accelerate_single_gpu.yaml` + `STARVLA_DISABLE_DEEPSPEED=1` |
| `train_realman_act.yaml:28` / `train_realman_dp.yaml:29` | `pretrained_backbone_weights: ResNet18_Weights.IMAGENET1K_V1` / `pretrained_backbone: true` 触发 ImageNet 下载 | 离线 smoke 设 ACT→`null`、DP→`false`（注释已允许）；正式训练保留 |
| `train_realman_act.yaml:50` / `train_realman_dp.yaml:51` + `:13` | `data_root_dir: /path/to/lerobot_datasets`、`wandb_entity: your_wandb_entity` 占位 | 训练走脚本的 `DATA_ROOT_DIR` env；dataloader gate 需改 YAML；`export WANDB_MODE=disabled` |
| 脚本 `--trainer.save_interval 5000`（ACT）/`2500`（DP） | 硬编码，短 smoke 不会产生 `checkpoints/steps_N` | 100-step smoke 只会产出 **`final_model/pytorch_model.pt`**（`train_starvla.py:443 _finalize_training` 恒存）。P3 直接加载 `final_model/pytorch_model.pt`（`read_mode_config` 用 `parents[1]` 定位 run 目录，`final_model/` 与 `checkpoints/` 同级均可） |
| **无部署层** | 无 `model2realman_interface.py`、无 `run_policy_server.sh`；且共享 `deployment/model_server/server_policy.py` 的 `PolicyServerWrapper` 硬性要求 `framework.action_model.action_horizon|future_action_window_size`（`policy_wrapper.py:79-88`），而 VM4A 的 ACT/DP YAML 只有 `framework.chunk_size`/`framework.horizon`，无 `action_model` 子块 → 会 `ValueError` | **不启动 WebSocket 策略服务**。P3 用 in-process self-test：`baseframework.from_pretrained` + `PolicyNormProcessor`（见 §5） |

> 与 RoboChallenge/G1 不同：Realman 脚本的 `cd` 回退（无，直接 `cd` 到 repo root 运行）、`run_root_dir=./results/Checkpoints`（与 G1 一致，非 `playground/Checkpoints`）、不硬编码 conda 环境名、不硬编码 8 卡——这些无需迁移补丁。主要缺口是 registry 占位符 + DP mixture 默认值 + 单卡化 + 自写 eval。

## 3. P1：数据获取（路线 B 已选定，A/C 为兜底）

### 路线 A：获取内部 Realman 数据集（降级兜底，仅当需要真实遥操作语义）

README 声明「adapted from a real-robot recipe validated on a private in-house LeRobot dataset」。先确认来源是否可取得，落盘到 `playground/Datasets/Realman/lerobot/<dataset>/`。**这是外部资产 gate：拿不到就转路线 C，不得伪造数据集存在。**

### 路线 B：公开数据集（✅ 已选定，本计划采用）

**数据源：`nvidia/PhysicalAI-Robotics-Manipulation-SingleArm` 的 `panda-open-drawer` 子集**（CC-BY-4.0，可商用；IsaacSim 生成的 Franka Panda 单臂数据）。已从 `meta/info.json` 实测核实其 schema 与 Realman 契约**逐维一致**：

| 子集 | action | state | episodes | 帧数 |
| --- | --- | --- | ---: | ---: |
| **`panda-open-drawer`（选定）** | 8D 关节 delta + gripper | **25D**（7 eef + 4 手指 + 7 关节位 + 7 关节速） | 1273 | 154256 |
| `panda-open-cabinet-left` | 8D 关节 delta + gripper | 25D | 1512 | 220038 |
| `panda-open-cabinet-right` | 8D 关节 delta + gripper | 25D | 1426 | 224953 |
| `panda-stack-platforms-texture` | 8D 关节 delta + gripper | 81D（含物体位姿） | 6303 | 551191 |
| `panda-stack-wide` | 7D 相对 EE（不用） | 53D | 10243 | 731785 |
| `panda-stack-platforms` | 7D 相对 EE（不用） | 81D | 17629 | 1456899 |

**已核实的硬事实（2026-08-31 从 `meta/info.json` 实测）：**

- `codebase_version: v2.1` ✅ —— 正是 gr00t dataloader 要求的版本，**无需转换**。
- action 列名 `panda_joint1..7_delta_pos + gripper` —— **本身已是 delta**，与 `action.delta_joints`(7) + `action.gripper_close`(1) 逐维对应。
- state 25D 顺序：`eef_pos(0:3) eef_quat(3:7) finger_joint1_pos(7) finger_joint2_pos(8) finger_joint1_vel(9) finger_joint2_vel(10) panda_joint1..7_pos(11:18) panda_joint1..7_vel(18:25)`。
- 双相机 `observation.images.world_camera` + `observation.images.hand_camera`（512×512）；另有 `observation.depths.*`（**忽略**）。
- 数据集**不带 `meta/modality.json`**（404），须按 §1.2 模板自写。

**下载（已在 P0 阶段执行）：**

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download nvidia/PhysicalAI-Robotics-Manipulation-SingleArm \
  --repo-type dataset --include "panda-open-drawer/**" \
  --local-dir playground/Datasets/Realman/lerobot
# 结果：playground/Datasets/Realman/lerobot/panda-open-drawer/{meta,data,videos}
```

**必做适配（registry + modality.json + YAML 层，不写转换器）：**

1. **`action_mode` 必须关闭**：Realman YAML 默认 `action_mode: delta`（把 parquet 的「绝对目标」转 delta），但 NVIDIA 的 action **已是 delta**，不关会双重差分。→ 两个 YAML 里 `action_mode: ""`（或删键），`action_mode_apply_keys`/`action_mode_state_map` 一并移除。
2. **state 切片（modality.json）**：`state.joints` → `observation.state[11:18]`（`panda_joint1..7_pos`）；`state.gripper` → `observation.state[7:8]`（`panda_finger_joint1_pos`，Franka 双指近似对称，取单指作开度代理；或取 `[7:8]` 与 `[8:9]` 均值）。
3. **相机键名**：registry `video_keys`/YAML `image_keys` 改为 `cam0_rgb`→`observation.images.world_camera`、`cam1_rgb`→`observation.images.hand_camera`（或直接改 registry 用原名 `world_camera`/`hand_camera`）。
4. **512×512 → 224×224**：dataloader `obs_image_size: [224,224]` 已 resize，无需动视频。
5. **语义**：IsaacSim 仿真（非真机遥操作），对「无真机复现」完全够用，且是真实任务语义（开抽屉）。

> 选定 `panda-open-drawer`（最小 + 25D state 最干净，无需处理 81D 里的物体位姿）。SO-100/SO-101 族（6-DoF，7D）仅作为备选，不再优先。

### 路线 C：合成数据集（降级兜底，仅当 NVIDIA 数据不可得）

编写一次性生成器 `Project_Analysis/evidence/realman/01_data/gen_synthetic_realman.py`，产出 schema 完全一致的 LeRobot v2.1 数据集（目标路径 `playground/Datasets/Realman/lerobot/realman_synthetic/`）：

```text
realman_synthetic/
├── data/chunk-000/episode_NNNNNN.parquet
├── videos/chunk-000/observation.images.cam0_rgb/episode_NNNNNN.mp4
├── videos/chunk-000/observation.images.cam1_rgb/episode_NNNNNN.mp4
└── meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json, embodiment.json}
```

parquet 每帧字段（维度见 §1.2）：

| 字段 | dtype/shape |
| --- | --- |
| `observation.state` | float32 [8] = 7 关节角 + 1 gripper |
| `action` | float32 [8] = 7 **绝对**关节目标 + 1 gripper 目标 |
| `timestamp` / `frame_index` / `episode_index` / `index` / `task_index` | float32[1] / int64[1]… |

合成要点：

- 每 episode ≥ 64 帧（> ACT 的 50 窗、DP 的 16 窗，留足采样余量）；`state[0:7]` 走平滑轨迹，`action[0:7] = state[t+1][0:7]`（绝对目标），`action[7] = gripper 目标`，保证训练期 `delta = action[0:7] - state[0:7]` 为合理小量。
- 视频用 **PyAV**（`av`）编码 H.264 224×224（服务器 OpenCV 缺 libx264，`cv2.VideoWriter` 会静默写空文件；PyAV 与 dataloader 的 `torchvision_av` 同后端，读回有保证）。
- **`meta/modality.json` 是本示例最关键且最易错的环节**（本示例 repo 内**没有**现成 `modality.json` 可复制，必须生成器自写）。结构如下，注意 `annotation` 用**扁平键**（`get_key_meta` 按 `.` 切分后以 `human.action.task_description` 作整键查找）：

```json
{
  "state": {
    "joints":   {"start": 0, "end": 7, "original_key": "observation.state"},
    "gripper":  {"start": 7, "end": 8, "original_key": "observation.state"}
  },
  "action": {
    "delta_joints":  {"start": 0, "end": 7, "original_key": "action"},
    "gripper_close": {"start": 7, "end": 8, "original_key": "action"}
  },
  "video": {
    "cam0_rgb": {"original_key": "observation.images.cam0_rgb"},
    "cam1_rgb": {"original_key": "observation.images.cam1_rgb"}
  },
  "annotation": {
    "human.action.task_description": {"original_key": "task_index"}
  }
}
```

- `meta/embodiment.json` 的 `embodiment_tag` 写 `new_embodiment`（与 registry 一致）；`meta/tasks.jsonl` 至少一条 prompt。
- 合成数据动作**无任务语义**，仅验证链路，不得表述为「模型会做某任务」。

> 路线 C 是「无真机 + 无外部数据」下 Realman 的最小可行复现。路线 A 能补「真实 8D 分布」，但语义仍非本计划考核点。

## 4. P2：dataloader、模型与双 100-step 训练

### 4.1 训练前三个 gate

先改 `data_config.py` 的 `<your_dataset>` → `realman_synthetic`（或路线 A 目录名），并把两个 YAML 的 `data_root_dir` 改为实际父目录，然后：

```bash
conda activate <starvla env>
export PYTHONPATH="$PWD:${PYTHONPATH}"

# gate 1: dataloader（Realman 双相机 8D/8D 采样）
python starVLA/dataloader/lerobot_datasets.py \
  --config_yaml examples/realRobots/Realman/train_files/train_realman_act.yaml

# gate 2: forward（VM4A 的 ACT.py / DiffusionPolicy.py **没有 __main__**，需用 build_framework 自检）
python - <<'PY'
from omegaconf import OmegaConf
from starVLA.model.framework.base_framework import build_framework
for y in ["examples/realRobots/Realman/train_files/train_realman_act.yaml",
          "examples/realRobots/Realman/train_files/train_realman_dp.yaml"]:
    cfg = OmegaConf.load(y)
    cfg.datasets.vla_data.data_root_dir = "<实际父目录>"
    fw = build_framework(cfg)
    print(y, "->", type(fw).__name__)   # ACT / DiffusionPolicy
PY
```

通过标准：dataloader 能采样两路 224×224 图像、8D state、8D action，且不报 `modality.json`/registry/video 错误；`build_framework` 能实例化 ACT（含 LeRobot ACTPolicy）与 DiffusionPolicy（含 ResNet-18 + DDPM scheduler + EMA）。**任何 gate 失败先修数据契约（尤其 modality.json 子键与 `<your_dataset>`），不启动 accelerate。**

### 4.2 各 100-step 单卡训练

```bash
conda activate <starvla env>
export CUDA_VISIBLE_DEVICES=0
export NUM_PROCESSES=1
export BATCH=1
export MAX_STEPS=100
export RUN_ID=realman_act_smoke      # DP 用 realman_dp_smoke
export WANDB_MODE=disabled
export DATA_ROOT_DIR=/absolute/path/to/lerobot_datasets

bash examples/realRobots/Realman/train_files/train_realman_act.sh  2>&1 | tee tmp/logs/realman_act_100step.log
bash examples/realRobots/Realman/train_files/train_realman_dp.sh   2>&1 | tee tmp/logs/realman_dp_100step.log
```

预期输出（`run_root_dir=./results/Checkpoints`，**不是** `playground/Checkpoints`）：

```text
results/Checkpoints/realman_act_smoke/
  config.yaml  config.full.yaml  dataset_statistics.json
  final_model/pytorch_model.pt          # 100 step < save_interval → 只有 final
results/Checkpoints/realman_dp_smoke/
  config.yaml  config.full.yaml  dataset_statistics.json
  final_model/pytorch_model.pt          # DP 的 pt 应含 ema_averaged.* 键
```

通过标准：loss 连续有限；无 OOM（ResNet-18 双相机 ACT/DP 在 5090 上 ~2–3 GB）；`final_model/pytorch_model.pt` 与 `dataset_statistics.json` 同 run 目录。DP 侧额外确认 `torch.load(...).keys()` 里存在 `ema_averaged.*`（`DiffusionPolicy.state_dict` override 负责持久化 EMA）。记录 wall time、step/s、峰值显存、最终 loss。

## 5. P3：checkpoint 重载 + in-process self-test（无真机）

### 5.1 自写 `local_self_test.py`（交付物）

新建 `examples/realRobots/Realman/train_files/local_self_test.py`（镜像 RoboChallenge/G1 的自测模式，但**不经 WebSocket**，直接进程内加载），职责：

1. `baseframework.from_pretrained(ckpt)` 加载 ACT 或 DP checkpoint；
2. `PolicyNormProcessor(ckpt, unnorm_key="new_embodiment")` 复用训练期 `ComposedModalityTransform` 做反归一化；
3. 构造合成观测：`image` = 两个 224×224×3 合成 ndarray（顺序 `[cam0_rgb, cam1_rgb]`）、`state` = 8D 合成向量、`lang` = 固定 prompt；
4. `predict_action(examples=[obs])` → `normalized_actions` `(1, T, 8)`，再 `proc.unapply_actions(...)` 反归一化；
5. 断言 `T` 与 `D` 正确、数值有限无 NaN，打印 latency，退出码 0 通过。

```python
import numpy as np
from starVLA.model.framework.base_framework import baseframework
from deployment.model_server.policy_norm_processor import PolicyNormProcessor

def run(ckpt, T_expected):
    proc = PolicyNormProcessor(ckpt, unnorm_key="new_embodiment")
    fw = baseframework.from_pretrained(ckpt).to("cuda").eval()
    obs = {
        "image": [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                  np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)],
        "state": np.zeros(8, dtype=np.float32),
        "lang": "perform the requested manipulation task",
    }
    norm = np.asarray(fw.predict_action(examples=[obs])["normalized_actions"])  # (1, T, 8)
    actions = proc.unapply_actions(norm[0])                                    # (T, 8)
    assert actions.shape == (T_expected, 8), actions.shape
    assert np.isfinite(actions).all(), "NaN/Inf in unnormalized actions"
    print(f"[OK] {ckpt.split('/')[-2]} actions {actions.shape}")
    return actions

run("results/Checkpoints/realman_act_smoke/final_model/pytorch_model.pt", T_expected=50)  # ACT chunk=50
run("results/Checkpoints/realman_dp_smoke/final_model/pytorch_model.pt",  T_expected=8)   # DP n_action_steps=8（非 horizon=16）
```

### 5.2 通过标准

1. ACT 返回 `(50, 8)`，DP 返回 `(8, 8)`（`T` 分别等于 ACT 的 `chunk_size` 与 DP 的 **`n_action_steps`（执行切片，非 `horizon`）**）；
2. `PolicyNormProcessor` 以 `unnorm_key="new_embodiment"` 正确解析（`dataset_statistics.json` 与 checkpoint 同 run 目录）；
3. 反归一化输出为有限浮点，无 NaN/维度错误；
4. DP 重载时 `ema_averaged.*` 键被 `DiffusionPolicy.load_state_dict` 正确回填（不报 unexpected key、EMA 非随机）；
5. 记录单轮推理延迟。

这是「无真机复现」的最终最低验收：**证明 StarVLA 的 ACT 与 DiffusionPolicy 两个 VM4A 基线能对接 Realman 8D 契约、训练出可重载的 checkpoint，并对合成观测输出语义正确的 8D 动作。** 它不验证真实 RM-75 执行、不验证遥操作、不验证任务成功率。

## 6. P4：结果归档

```text
Project_Analysis/evidence/realman/
  00_preflight/              # GPU、磁盘、commit、依赖版本（含 lerobot/diffusers）
  01_data/                   # 路线 A/C 数据 schema 证据、合成生成器、modality.json、占用统计
  02_training/               # YAML、训练脚本副本、loss/显存/step-s、两个 ckpt hash
  03_reload_self_test/       # local_self_test.py、(50,8)/(16,8) 输出、反归一化证据
  README.md                  # 实测日期、命令、成功/失败与未完成范围
```

保留两个 `dataset_statistics.json`（反归一化依赖）。明确记录数据来自路线 A 还是 C；若为 C，标注「合成数据，无任务语义」。code 补丁（`data_config.py` 占位符替换、DP mixture 默认值修复、单卡化）→ fork `TangYishanE/starVLA` 分支 `repro/realman`；checkpoint → HF 私有 repo（必须带 `config.yaml` + `dataset_statistics.json`，见 `starvla-demo-archive` 约定）。

## 7. 失败分流与停止条件

| 失败点 | 首先检查 | 停止/继续规则 |
| --- | --- | --- |
| 数据不可得（路线 A） | 内部来源是否提供数据集 | 不伪造；转路线 C 合成 |
| ACT 实例化 `ImportError: lerobot` | 是否 `pip install lerobot` | 补装后重试，不绕过 |
| 合成数据 dataloader 失败 | `modality.json` 子键（尤其 annotation 扁平键 `human.action.task_description`）、parquet 字段名、维度 | 对照 §1.2 契约逐项修 |
| forward gate 失败 | `build_framework` 报错、`data_root_dir` 路径、`lerobot`/`diffusers` | 未通过 gate 不得启动 accelerate |
| DP 短 episode 采样失败 | 是否误用了 `realman_example`（50 窗）而非 `realman_example_dp`（16 窗） | 修 `train_realman_dp.sh:24` 默认值 |
| 训练 OOM | batch、单卡化、DeepSpeed | 固定 ResNet-18，batch=1；不换大 backbone |
| `read_mode_config` 找不到 `config.yaml`/`dataset_statistics.json` | checkpoint 是否与统计文件同 run 目录、路径是否 `final_model/` 或 `checkpoints/` 两级 | 缺失则重训或补拷 |
| DP 重载 unexpected key / EMA 异常 | checkpoint 是否含 `ema_averaged.*`、是否走 `DiffusionPolicy.load_state_dict` | 确认 `_finalize_training` 用的 `_get_state_dict` 走 override |

## 8. 最终交付标准

完成后应在干净 shell 中按序重新执行并通过：

1. 数据 metadata/shape 检查（8D state / 8D action / 双相机 / mean_std+min_max）；
2. dataloader gate；
3. ACT + DP 双 forward gate（`build_framework`）；
4. ACT 100-step + DP 100-step 两个 checkpoint（`final_model/pytorch_model.pt` + `dataset_statistics.json`）；
5. in-process self-test：ACT `(50,8)`、DP `(16,8)` 反归一化有限；
6. 汇总：数据来源（A/C）、峰值显存、step/s、单轮推理延迟、未完成范围。

向老师的建议表述：**「在无实体机器人的条件下，完成 StarVLA Realman RM-75 示例（VM4A 的 ACT 与 DiffusionPolicy 两个基线）的数据契约确认、8D 关节动作训练与 checkpoint 重载，并验证对合成观测输出语义正确的 8D 动作块。」** 需同时注明：本计划未涉及实体 RM-75 与遥操作采集，因此不报告真实机器人闭环或任务成功率；数据若走合成路线，动作不具任务语义。Realman 无部署/eval 层（无 `model2realman_interface.py`、无挑战赛协议），故本计划不含协议闭环验证，这是与 RoboChallenge/G1 的**结构性差异**而非遗漏。

## 附：与 RoboChallenge / UnitreeG1 计划的关键差异

| 维度 | RoboChallenge | UnitreeG1_WholeBody | **Realman（本计划）** |
| --- | --- | --- | --- |
| 模型族 | QwenOFT (0.8B VLA) | QwenOFT (0.8B VLA) | **ACT + DiffusionPolicy（VM4A，ResNet-18）** |
| 数据 | 公开 HF，`--only` 单任务 | 非公开，三路线 | **公开 HF（NVIDIA PhysicalAI，Franka 7-DoF）或合成兜底** |
| 数据量级 | 9.4 GB/任务 | `test_sonic` ~数 GB | **合成 <1 GB** |
| 动作契约 | 8D ee_pose+gripper | **78D SONIC latent（64+7+7）** | **8D 关节 delta + gripper** |
| 归一化 | min_max | q99 | **mean_std(joints) + min_max(gripper)** |
| 显存 | ~5 GB | ~5–7 GB | **~2–3 GB（可跑本地 4070 Ti）** |
| 免真机验证 | mock server（HTTP 协议） | 自写合成观测 + 64/7/7 切分 | **in-process self-test（重载 + 反归一化，无协议）** |
| 代码缺口 | 2 import + 2 cd 层级 | 缺 `local_self_test.py` + PYTHONPATH | **registry 占位符 + DP mixture 默认值 + 单卡化 + 自写 eval（无部署层）** |
| 完成语义 | 协议闭环（GET/POST） | 动作切分（64/7/7） | **双基线训练 + 重载 + (T,8) 反归一化** |
