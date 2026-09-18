# Franka (Real Robot) 无真机复现执行计划

**目标项目：** `examples/realRobots/Franka`，在**无实体 Franka 机械臂**条件下复现其「数据契约 → 数据注册 → 训练 → 策略服务 → StarVLA→Franka 动作适配」链路。  
**运行边界：** 远程单张 RTX 5090 32 GiB、无实体 Franka、无真实相机。  
**本计划的完成定义：** 得到一份符合 6D-state / 7D-action（delta EEF）契约的 LeRobot v2.1 数据集；QwenOFT checkpoint 训练与重载通过；策略服务 `server_policy.py` 可启动；客户端对合成观测（2×224×224 图像 + 语言指令）输出并正确反归一化为 7D delta-EE 动作块 `(16, 7)`。

> 这是一份「数据—注册—训练—策略服务—动作契约」的无真机复现计划。**没有实体 Franka 和真实相机时，不得**表述为真实机器人闭环成功、抓取/放置成功率或安全执行验证。本计划最多验证到「StarVLA 策略服务能对 Franka 观测契约输出语义正确的 7D delta-EE 动作」，这是部署链路的模型侧闭环。

> ⚠️ 与 RoboChallenge / UnitreeG1 两个示例不同，Franka 示例是三者中**代码缺口最多**的一个：数据 mixture 未注册、推理 client 的 import 断裂、策略服务启动脚本 cd 层级少一级、YAML 与训练脚本的框架/基座模型互相矛盾、`modality.json` 与 `data_config.py` 契约不一致、且**不含任何数据集与转换脚本**（`franka2lerobot/` 只有一份描述性的 README）。本计划把「契约归一」和「缺口补丁」提升为 P0 的第一等交付物，而不是像另两个计划那样作为轻量前置。

## 0. 计划总览

### 0.1 固定的首轮配置

| 项目 | 首轮选择 | 原因 |
| --- | --- | --- |
| 机械臂/动作 | **单臂 Franka，7D delta-EE**（`custom_robot_config` = `SingleFrankaRobotiqDeltaEefDataConfig`） | README 主示例、部署 client、`starvla_cotrain_franka_single.yaml`（`action_dim: 7`）都以 7D 单臂为最小闭环；双 14D 需新写 DataConfig，首轮不做 |
| 动作语义 | `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]` = `delta_eef_position(3) + delta_eef_rotation(3) + gripper_close(1)` | 与 `data_config.py` 的 `action_keys` 一致 |
| 观测 | **2 路 RGB**：`video.base_view` + `video.ego_view`，resize 224×224；**`include_state: false`（图像+语言，不进 proprio state）** | 部署 client 只发 `{image, lang}`，与 YAML `include_state: false` 一致 |
| 数据 | 三条路线（见 P1）：优先 A 拿真实 Franka 数据；不可得则 C 合成数据集（默认）；B 为 raw→v2.1 转换备选 | 本 clone 无 `playground/`，Franka 数据不随仓库分发且无自动下载器，数据是首要 gate |
| 框架 | `QwenOFT` + Qwen3.5-0.8B + DiT-B 扩散动作头 | 单卡风险最低，与另两个 realRobots 计划一致；YAML 默认的 `QwenGR00T`/`Qwen2.5-VL-3B` 与脚本的 `QwenOFT`/`Qwen3-VL-4B` 互相矛盾，必须归一 |
| 模型配置 | 224×224，`action_horizon: 16`，`BATCH=1`，`MAX_STEPS=1000` | 先验证工程闭环；1000 step 不是性能训练 |
| 评测 | 策略服务 + 合成观测 → 7D 反归一化验证 | 不需要机器人/相机；验证 checkpoint、归一化、图像输入、动作反归一化 |
| 不做的内容 | 真实 Franka 执行、双 14D 臂、真实相机采集、任务成功率 | 超出「无真机」边界 |

### 0.2 里程碑、时间与空间

| 阶段 | 主要工作 | 日历时间 | GPU 时间 | 峰值新增空间 | 可判定的通过条件 |
| --- | --- | ---: | ---: | ---: | --- |
| P0 | 环境检查 + 契约归一 + 6 处缺口补丁 + mixture 注册 | 0.5–1 天 | 0 | <5 GB | registry 能解析 `franka_eef_joints`；dataloader/forward import 正确 |
| P1 | 数据获取/合成（三路线） | 0.5–2 天 | 0 | 1–20 GB | 数据集 schema 与 6D/7D 契约逐项一致 |
| P2 | dataloader/forward gate 与 1000-step 训练 | 2–6 h | 1–2 h | +5–10 GB | checkpoint 与 `dataset_statistics.json` 生成 |
| P3 | 策略服务 + 合成观测动作反归一化验证 | 1–2 h | <1 h | 可忽略 | 连续输出 `(16,7)` 动作块，gripper 二值化正确，无 NaN |
| P4 | 结果归档 | 1–2 h | 0 | <2 GB | 配置、日志、哈希、JSON 齐全 |

**首轮总计：1–3 天、约 2–3 GPU 小时、建议服务器空闲空间 ≥ 30 GB。** 空间大头是 Qwen3.5-0.8B 权重（~5 GB）与合成/获取的数据集（1–20 GB），与 UnitreeG1 同量级，远小于 RoboChallenge 的百 GB 量级。

## 1. 预期输入、输出与数据契约

### 1.1 必须取得/准备的内容

| 资产 | 来源 | 首轮范围 | 说明 |
| --- | --- | --- | --- |
| Qwen3.5-0.8B | YAML 指定 `./playground/Pretrained_models/Qwen3.5-0.8B` | 单模型 | ~5 GB；`huggingface-cli download Qwen/Qwen3.5-0.8B`（国内可用 `HF_ENDPOINT=https://hf-mirror.com`） |
| Franka 数据集 | 三路线（§3） | 至少一个符合契约的 LeRobot v2.1 数据集 | 仓库**不随发**、无公开 HF 数据、无转换脚本（`franka2lerobot/` 只有 README） |
| 数据注册补丁 | 自写，改 `examples/realRobots/Franka/train_files/data_registry/data_config.py` | mixture `franka_eef_joints` | 现有 YAML 引用的 mixture **未注册**（见 §2.3） |

### 1.2 数据契约（必须逐项确认，来源 = `data_registry/data_config.py` 的 `SingleFrankaRobotiqDeltaEefDataConfig`）

**state 6D（首轮 `include_state: false`，不进模型，但统计/契约仍要一致）：**

| 模态键 | 子键 | 维度 |
| --- | --- | --- |
| `state.eef_position` | (0:3) | 3 |
| `state.eef_rotation` | (0:3) | 3 |
| **合计** | | **6** |

**action 7D：**

| 模态键 | 子键 | 维度 |
| --- | --- | --- |
| `action.delta_eef_position` | (0:3) | 3 |
| `action.delta_eef_rotation` | (0:3) | 3 |
| `action.gripper_close` | (0:1) | 1 |
| **合计** | | **7** |

其他关键契约：

| 项 | 约定 | 验收检查 |
| --- | --- | --- |
| 视频 | 双路 `video.base_view` + `video.ego_view`，resize 224×224 | 两路视频存在、可解码、顺序稳定 |
| 语言 | `annotation.human.action.task_description`（task_index → tasks.jsonl） | tasks.jsonl 存在对应 prompt |
| 归一化 | position/rotation 全部 **min_max**；`gripper_close` **binary** | `dataset_statistics.json` 中 min/max 可读、gripper 为 binary |
| embodiment | `EmbodimentTag.NEW_EMBODIMENT` → 反归一化键 = **`new_embodiment`** | 与 client 的 `embodiment_key` 一致（现 client 硬编码 `franka`，**错**，见 §5.1） |
| horizon | 16（YAML `action_horizon: 16`，registry `action_indices=list(range(16))`） | 训练/部署一致，输出 `(16,7)` |
| state 输入 | `include_state: false` → 模型只吃图像+语言 | 部署 client 只发 `{image, lang}`，一致 |
| 模型输出 | 服务端返回**归一化** `normalized_actions`，shape `[B,16,7]` | P3 反归一化后 7D，gripper ∈ {−1,+1} |
| action_type | `delta_ee` | 与 delta-EE 动作语义一致 |

> 契约权威来源是 `data_registry/data_config.py`（被 `registry.py` → `lerobot_datasets.py` → `policy_norm_processor.py` 三处一致消费）。`train_files/modality.json` 描述的是**另一套旧契约**（18D joints state / 7D x,y,z,roll,pitch,yaw,gripper / quaternion 原始键），与 `data_config.py` 完全不符，属孤儿文件，**忽略，不要按它造数据**。parquet 列名的精确映射（`video.*`→`observation.images.*` 等）在 P0 通过读 `starVLA/dataloader/gr00t_lerobot/datasets.py` 的 `_get_lerobot_modality_meta()` 确认后再落数据。

## 2. P0：前置环境与代码缺口补丁

### 2.1 环境检查（不下载、不训练）

在 `starVLA` 仓库根目录执行，保存到 `Project_Analysis/evidence/franka/00_preflight/`：

```bash
git rev-parse HEAD && git status --short
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h .
conda env list
which nvcc || true
```

激活 StarVLA 环境后执行依赖探针（`torch` / `accelerate` / `cv2` / `pyarrow` / `lerobot` / `huggingface_hub` / `websockets` / `msgpack_numpy`）。通过标准与另两个计划一致：RTX 5090、可用空间 ≥ 30 GB、真实 CUDA toolkit（DeepSpeed 需要真实 nvcc，而非 stub）。

### 2.2 当前 main 必做代码缺口修复（Franka 独有，正式执行前建立独立补丁/提交）

| # | 文件/位置 | 当前问题 | 应修改为 |
| --- | --- | --- | --- |
| G1 | `train_files/data_registry/data_config.py` 的 `DATASET_NAMED_MIXTURES` | **缺 `franka_eef_joints` / `dual_franka_eef_joints`** → 训练与部署都会 `KeyError: data_mix not in DATASET_NAMED_MIXTURES` | 加 `"franka_eef_joints": [("franka_pick_and_place_lerobot", 1.0, "custom_robot_config")]`（见 §2.3） |
| G2 | 同文件 `ROBOT_TYPE_CONFIG_MAP` | 只有 `custom_robot_config`(7D) / `demo_sim_franka_delta_joints`(8D) / `SO101`；**无 14D 双臂 DataConfig** | 首轮不做双臂；若要做，需新写 `DualFrankaRobotiqDeltaEefDataConfig`（14D） |
| G3 | `eval_files/inference_*_example.py` | `from websocketclient import WebsocketClientPolicy` **import 断裂**（仓库无 `websocketclient.py`） | 改为 `from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy`（或加 shim） |
| G4 | `eval_files/run_policy_server.sh` | `STARVLA_DIR="$(cd "$(dirname "$0")/../../..")"` 只到 `examples/`，随后 `cd` 进去跑 `deployment/...` 会失败（迁移后少退一级） | 改为 `../../../../` 回到 repo root（或显式 `export STARVLA_DIR`） |
| G5 | `train_files/modality.json` | 孤儿文件，契约与 `data_config.py` 不符 | 忽略；数据契约以 `data_config.py` 为准 |
| G6 | YAML `framework.name`/`base_vlm` vs 训练脚本 | YAML=`QwenGR00T`+`Qwen2.5-VL-3B-Instruct`；脚本=`QwenOFT`+`Qwen3-VL-4B-Instruct-Action`，互相矛盾 | 统一为 `QwenOFT` + `Qwen3.5-0.8B`（单卡） |
| G7 | `run_franka_train_{single,dual}.sh` | `--num_processes 8` 硬编码、NCCL InfiniBand 变量、`${data_mix}` 未定义（死变量）、`max_train_steps 100000`、`wandb_entity zwanggk` | 见 §4.2 单卡启动命令，逐项覆盖 |
| G8 | `inference_single_example.py` 的 `embodiment_key="franka"` | 与 `custom_robot_config`→`NEW_EMBODIMENT`→`new_embodiment` 不符 | 改为 `new_embodiment`（§5.1） |

修改后运行：

```bash
bash -n examples/realRobots/Franka/eval_files/run_policy_server.sh
PYTHONPATH="$PWD" python -c "from examples.realRobots.Franka.train_files.data_registry.data_config import DATASET_NAMED_MIXTURES; print('franka_eef_joints' in DATASET_NAMED_MIXTURES)"
PYTHONPATH="$PWD" python -c "from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy; print('client import OK')"
```

通过标准：`bash -n` 返回 0；mixture 命中；client import 成功。**不要跳过**：G1/G3/G4 任意一处未修，P2 训练或 P3 部署都会在加载前失败。

### 2.3 数据注册补丁（G1 的最小可运行改法）

编辑 `examples/realRobots/Franka/train_files/data_registry/data_config.py`，在 `DATASET_NAMED_MIXTURES` 中新增（`custom_robot_config` 已在基类 `starVLA/dataloader/gr00t_lerobot/data_config.py` 与示例文件中注册为 `SingleFrankaRobotiqDeltaEefDataConfig`，无需重复定义）：

```python
DATASET_NAMED_MIXTURES = {
    # ... 现有条目保持不变 ...
    "franka_eef_joints": [("franka_pick_and_place_lerobot", 1.0, "custom_robot_config")],
}
```

> `registry.py` 会 `glob("examples/**/train_files/data_registry")` 自动发现并 merge 本文件，无需改 `mixtures.py` 或 `data_config.py` 基类。数据集目录名 `franka_pick_and_place_lerobot` 与 README 示例一致，且 `lerobot_datasets.py` 的 `dataset_path = data_root_dir / data_name`（`data_root_dir=./playground/Datasets`）恰好拼出 `playground/Datasets/franka_pick_and_place_lerobot/`。

## 3. P1：数据获取（三路线，按优先级）

### 路线 A：获取真实 Franka 数据（首选，若可得）

Franka 示例无公开数据集、无下载器。先确认是否有内部/维护者数据：

```bash
test -d playground/Datasets/franka_pick_and_place_lerobot && echo 'franka data present'
```

若可得，落盘到 `playground/Datasets/franka_pick_and_place_lerobot/`（`data/` + `videos/` + `meta/`，v2.1 布局）。**这是外部资产 gate：拿不到就转路线 C，不伪造。**

### 路线 B：raw → v2.1 转换（可选，需自写转换器）

`franka2lerobot/README.md` 描述了「raw → LeRobot v3.0 → v2.1」两段流程，但**仓库不含 `convert.sh` / `convert_dataset.py`**，其 §6 引用的是 `starVLA_franka` 内部脚本。若你有 raw Franka 采集（`demo_*.pkl`/HDF5/ROS bag 等），需按该 README 的 §4–§5 自写两段转换器，产出 §1.2 契约的 v2.1 数据集。工程量大，仅在需要「真实语义」数据时选择。

### 路线 C：合成数据集（默认「无外部依赖」路线）

编写一次性生成器 `Project_Analysis/evidence/franka/01_data/gen_synthetic_franka.py`，产出一个 schema 完全一致的 LeRobot v2.1 数据集（目标路径 `playground/Datasets/franka_pick_and_place_lerobot/`）：

```text
franka_pick_and_place_lerobot/
├── data/chunk-000/episode_NNNNNN.parquet
├── videos/chunk-000/observation.images.base_view/episode_NNNNNN.mp4
├── videos/chunk-000/observation.images.ego_view/episode_NNNNNN.mp4
└── meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json, ...}
```

parquet 每帧至少包含（维度见 §1.2；**列名以 P0 确认的 `_get_lerobot_modality_meta()` 映射为准**）：

| 字段 | dtype/shape |
| --- | --- |
| `observation.images.base_view` / `observation.images.ego_view` | 视频帧索引（或图像列） |
| `observation.state.eef_position` | float32 [3] |
| `observation.state.eef_rotation` | float32 [3] |
| `action.delta_eef_position` | float32 [3] |
| `action.delta_eef_rotation` | float32 [3] |
| `action.gripper_close` | float32 [1]（binary ±1） |
| `timestamp` / `frame_index` / `episode_index` / `index` / `task_index` | float32[1] / int64[1]… |

- 视频用 `cv2.VideoWriter` 生成 224×224 H.264 短片段（两路）
- `meta/modality.json` 按 P0 确认的键映射手写，与 `data_config.py` 的 `modality_keys` 对齐
- 合成数据的动作**无任务语义**，仅用于验证训练/部署链路，不得表述为「模型会抓取/放置」

> 路线 C 是「无真机 + 无数据 + 无转换脚本」下的最小可行复现。路线 A 的真实 Franka 数据能补上「真实 7D 分布」但语义仍非本计划考核点。

## 4. P2：dataloader、模型与 1000-step 训练

### 4.1 训练前两个 gate

```bash
conda activate <starvla env>
export PYTHONPATH="$PWD:${PYTHONPATH}"

python starVLA/dataloader/lerobot_datasets.py \
  --config_yaml examples/realRobots/Franka/train_files/starvla_cotrain_franka_single.yaml

python starVLA/model/framework/VLM4A/QwenOFT.py \
  --config_yaml examples/realRobots/Franka/train_files/starvla_cotrain_franka_single.yaml
```

通过标准：dataloader 能采样双路 224×224 图像、6D state、7D action，且不报 mixture/`modality.json`/registry/video symlink 错误；QwenOFT forward 不出现 key/shape/模型路径错误。**任何 gate 失败先修契约/注册，不启动 DeepSpeed。** 注意：YAML 里 `framework.name: QwenGR00T` 是陈旧值，forward gate 与训练都以 `QwenOFT` 为准（若 `QwenOFT.py` 读 YAML 报框架名不匹配，先在 P0 把 YAML 的 `framework.name` 改为 `QwenOFT`）。

### 4.2 1000-step 单卡训练

不要直接跑 `run_franka_train_single.sh`（8 卡、InfiniBand、4B 基座、10 万 step）。用下面的单卡命令覆盖：

```bash
conda activate <starvla env>
export CUDA_VISIBLE_DEVICES=0
export BATCH=1
export MAX_STEPS=1000
export WANDB_MODE=disabled
# 仅当有真实 CUDA toolkit 时设置 CUDA_HOME

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/realRobots/Franka/train_files/starvla_cotrain_franka_single.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3.5-0.8B \
  --datasets.vla_data.per_device_batch_size ${BATCH} \
  --trainer.max_train_steps ${MAX_STEPS} \
  --trainer.save_interval ${MAX_STEPS} \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100000 \
  --run_root_dir ./results/Checkpoints \
  --run_id franka_smoke \
  2>&1 | tee tmp/logs/franka_train_1000step.log
```

预期输出：

```text
results/Checkpoints/franka_smoke/
  config.yaml
  config.full.yaml
  dataset_statistics.json
  checkpoints/steps_1000_pytorch_model.pt
```

> 注意：Franka 脚本/命令 `run_root_dir=./results/Checkpoints`（不是 `playground/Checkpoints`），与 RoboChallenge 不同、与 G1 相同。

通过标准：loss 连续有限；无 CUDA OOM（0.8B + DiT-B 扩散头在 5090 上预计 ~8–12 GB，实测为准；DiT 扩散头比 G1 的 MLP 头重，batch=1 优先）；checkpoint 与 `dataset_statistics.json` 同目录。记录 wall time、step/s、峰值显存、最终 loss。

## 5. P3：策略服务与动作反归一化验证（无真机）

### 5.1 修复 client（G3 + G8 的落地）

`inference_single_example.py` 两处改：

```python
# G3：import 修复
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

# G8：embodiment key 修复（custom_robot_config → NEW_EMBODIMENT → new_embodiment）
action_norm_stats = load_action_norm_stats(action_stats_path, embodiment_key="new_embodiment")
```

再写一个最小验证客户端 `Project_Analysis/evidence/franka/03_deploy_action_check/client_smoke.py`（镜像 `inference_single_example.py`，但把 `capture_images_from_cameras()`/`YourRobotEnv` 换成合成观测），职责：

1. `WebsocketClientPolicy(host, port)` 连接策略服务；
2. 构造合成请求：`image` = 两个 224×224×3 合成 ndarray（base + ego，顺序与训练一致）、`lang` = 固定 prompt；
3. `parse_response(result)` 得 `[16, 7]` normalized action chunk；
4. `unnormalize_actions(...)` 反归一化，断言 `(16,7)` 且 `action[:,6]` 二值化为 {−1,+1}，无 NaN。

### 5.2 启动策略服务 + 验证

终端 A（策略服务，StarVLA 环境）：

```bash
cd /absolute/path/to/starVLA
bash examples/realRobots/Franka/eval_files/run_policy_server.sh \
  results/Checkpoints/franka_smoke/checkpoints/steps_1000_pytorch_model.pt
# 或显式指定根目录，规避 G4 的默认值缺陷：
#   STARVLA_DIR="$(pwd)" bash examples/realRobots/Franka/eval_files/run_policy_server.sh <ckpt>
```

终端 B（验证客户端）：

```bash
cd /absolute/path/to/starVLA
export PYTHONPATH="$PWD:${PYTHONPATH}"
python Project_Analysis/evidence/franka/03_deploy_action_check/client_smoke.py \
  --host 127.0.0.1 --port 5694
```

### 5.3 通过标准

1. 服务端成功加载 checkpoint 并返回 metadata（`action_chunk_size=16`、`action_keys` 为 delta_eef_position/delta_eef_rotation/gripper_close；日志含 [TRAIN/TEST CONSISTENCY CHECK] 提示）；
2. 客户端收到 `normalized_actions` shape `[1,16,7]`；
3. 反归一化后 7D 动作有限、gripper 二值化为 {−1,+1}、无 NaN；
4. 记录单轮推理延迟。

这是「无真机复现」的最终最低验收：**证明 StarVLA 策略服务能对接 Franka 观测契约并输出语义正确的 7D delta-EE 动作。** 它不验证真实 Franka 运动学、抓取或安全执行。

## 6. P4：结果归档

```text
Project_Analysis/evidence/franka/
  00_preflight/              # GPU、磁盘、commit、依赖版本、registry/mixture 命中证明
  01_data/                   # 路线 A/B/C 数据 schema 证据、合成生成器、占用统计
  02_training/               # YAML、训练命令副本、loss/显存/step-s、ckpt hash
  03_deploy_action_check/    # client_smoke.py、服务端/client 日志、(16,7)→反归一化证据
  README.md                  # 实测日期、命令、成功/失败与未完成范围
```

保留 `dataset_statistics.json`（min_max 反归一化依赖它）。明确记录数据来自路线 A/B/C；若为 C，标注「合成数据，无任务语义」。记录已修 6 处缺口（G1/G3/G4/G6/G7/G8）的 diff，供上游合入。

## 7. 失败分流与停止条件

| 失败点 | 首先检查 | 停止/继续规则 |
| --- | --- | --- |
| `data_mix not in DATASET_NAMED_MIXTURES` | G1 是否已注册、registry 是否自动发现 `data_registry/` | 未注册禁止训练；先补 G1 再重跑 gate |
| 数据不可得（路线 A/B） | 维护者/内部是否提供 Franka 数据；是否愿意自写转换器 | 不伪造；转路线 C 合成数据集 |
| 合成数据 dataloader 失败 | parquet 列名是否与 `_get_lerobot_modality_meta()` 一致、video symlink、维度 | 对照 §1.2 契约逐项修 |
| forward gate 失败 | `framework.name` 是否已改 `QwenOFT`、`base_vlm` 路径、`action_dim=7`、模型加载 | 未通过 gate 不得启动 DeepSpeed |
| 训练 OOM | batch、CUDA/DeepSpeed、DiT-B 扩散头显存 | 固定 0.8B，batch=1；不改用 3B/4B 全参数 |
| 服务端 metadata/unnorm_key 异常 | `dataset_statistics.json` 是否与 checkpoint 同 run 目录、embodiment 是否 `new_embodiment` | 缺失则重训或补拷 |
| client import 失败 | G3 是否已修（`websocketclient` → 真实模块路径） | 修复后重跑，无需重训 |
| `run_policy_server.sh` 找不到 `deployment/...` | G4 的 `STARVLA_DIR` 层级 | 改为 `../../../..` 或显式 `STARVLA_DIR=$(pwd)` |
| gripper 未二值化 / 反归一化维度错 | `unnormalize_actions` 的 gripper 索引（=6）、`embodiment_key` | 抓包确认服务端响应结构与 stats key |

## 8. 最终交付标准

完成后应在干净 shell 中按序重新执行并通过：

1. 数据 metadata/shape 检查（6D state / 7D action / 双相机 / min_max+binary）；
2. mixture 注册命中（`franka_eef_joints` in registry）；
3. dataloader gate；
4. QwenOFT forward gate；
5. 1000-step checkpoint 生成（`steps_1000_pytorch_model.pt` + `dataset_statistics.json`）；
6. 策略服务启动 + 合成观测 → `(16,7)` 动作正确反归一化（gripper ∈ {−1,+1}）；
7. 汇总：数据来源（A/B/C）、峰值显存、step/s、单轮推理延迟、未完成范围。

向老师的建议表述：**「在无实体 Franka 的条件下，完成 StarVLA Franka 示例的数据契约归一（7D delta-EE）、QwenOFT 训练与 checkpoint 加载、策略服务启动，并验证 StarVLA→Franka 适配器对合成观测输出语义正确的 7D delta-EE 动作块。」** 需同时注明：本计划未涉及实体 Franka、真实相机与运动学执行，因此不报告真实机器人闭环或任务成功率；数据若走合成路线，动作不具任务语义。同时列出已修复的 6 处上游缺口（G1/G3/G4/G6/G7/G8）作为额外交付。

## 附：与 RoboChallenge / UnitreeG1 计划的关键差异

| 维度 | RoboChallenge | UnitreeG1 | **Franka（本计划）** |
| --- | --- | --- | --- |
| 数据 | 公开 HF，`--only` 单任务可下 | 不随发/非公开，三路线 | **不随发/非公开，且无转换脚本**（`franka2lerobot/` 只有 README） |
| 数据量级 | 9.4 GB/任务，全量 1 TB | `test_sonic` ~数 GB，合成 <1 GB | 真实 ~数 GB，合成 <1 GB |
| 动作契约 | 8D ee_pose+gripper | 78D SONIC latent（64+7+7） | **7D delta-EE**（3+3+1，单臂） |
| state 输入 | 7D，进模型 | 72D，进模型 | **`include_state: false`，不进模型**（图像+语言） |
| 归一化 | min_max | q99 | **min_max + binary(gripper)** |
| 动作头 | MLP head | MLP head | **DiT-B 扩散动作头**（略重） |
| horizon | 8 | 8 | **16** |
| 免真机验证 | mock server（上游 HTTP 协议） | 自写合成观测 + 动作切分 | **策略服务(WebSocket) + 合成观测反归一化** |
| 代码缺口 | 2 处 import 路径 + 2 处 cd 层级 | 缺 `local_self_test.py` + `PYTHONPATH` 疑点 | **6 处**：mixture 未注册、client import 断裂、server cd 少一级、框架/基座矛盾、`modality.json` 孤儿、embodiment key 错 |
| 完成语义 | 协议闭环（GET/POST） | 动作契约切分（64/7/7） | 动作反归一化（16×7 delta-EE） |
