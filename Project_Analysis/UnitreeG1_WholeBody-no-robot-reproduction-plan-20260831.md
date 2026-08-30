# UnitreeG1_WholeBody 无真机复现执行计划

**目标项目：** `examples/realRobots/UnitreeG1_WholeBody`，在**无实体 G1 机器人**条件下复现其「数据契约 → 训练 → 策略服务 → StarVLA→G1 动作适配」链路。  
**运行边界：** 远程单张 RTX 5090 32 GiB、无实体机器人、无 PICO VR、无 SONIC/WBC 控制器。  
**本计划的完成定义：** 得到一份符合 78D/72D 契约的数据集；QwenOFT（Qwen3.5-0.8B）checkpoint 训练与重载通过；策略服务可启动；`model2unitree_g1_interface.ModelClient` 对合成 GR00T 风格观测输出并正确切分为 `64D motion_token + 7D 左手 + 7D 右手` 的动作块。

> 这是一份「数据—训练—策略服务—动作契约」的无真机复现计划。**没有实体 G1、SONIC 控制器和 PICO 遥操作时，不得**表述为真实机器人闭环成功、平衡控制验证或任务成功率。本计划最多验证到「StarVLA 策略服务器能对 G1 观测契约输出语义正确的 78D 动作」，这是部署链路的模型侧闭环。

## 0. 计划总览

### 0.1 固定的首轮配置

| 项目 | 首轮选择 | 原因 |
| --- | --- | --- |
| 数据 | 三条路线（见 P1）：优先 A 获取 `test_sonic`；不可得则 C 合成数据集（默认）；B 为 MuJoCo 仿真采集备选 | 本 clone 无 `playground/`，`test_sonic` 不随仓库分发，数据是首要 gate |
| 机器人/动作 | `unitree_g1_sonic_dex3`，78D SONIC 动作契约（64 motion token + 7+7 手部关节） | 唯一已注册的 registry 类型与 YAML 配置 |
| 框架 | `QwenOFT` + Qwen3.5-0.8B + MLP head | 官方 walkthrough 配置，单卡风险最低 |
| 模型配置 | 224×224 单相机，horizon=8，`BATCH=1`，`MAX_STEPS=1000` | 先验证工程闭环；1000 step 不是性能训练 |
| 评测 | 策略服务 + `ModelClient` 合成观测 → 78D 分片验证 | 不需要机器人/控制器；验证 checkpoint、归一化、状态/图像输入、动作切分 |
| 不做的内容 | 真实 G1 执行、SONIC/WBC 解码、PICO 遥操作采集、任务成功率 | 超出「无真机」边界 |

### 0.2 里程碑、时间与空间

| 阶段 | 主要工作 | 日历时间 | GPU 时间 | 峰值新增空间 | 可判定的通过条件 |
| --- | --- | ---: | ---: | ---: | --- |
| P0 | 环境/代码路径检查与缺口补丁 | 0.5–1 天 | 0 | <5 GB | import、CUDA、模型下载、脚本路径正确 |
| P1 | 数据获取（三路线，见 §3） | 0.5–2 天 | 0 | 1–20 GB | 数据集 schema 与 72D/78D 契约逐项一致 |
| P2 | dataloader/forward gate 与 1000-step 训练 | 2–6 h | 1–2 h | +5–10 GB | checkpoint 与 `dataset_statistics.json` 生成 |
| P3 | 策略服务 + 合成观测动作切分验证 | 1–2 h | <1 h | 可忽略 | 连续输出 `(T,78)` 并正确切分 64/7/7，无 NaN |
| P4 | 结果归档 | 1–2 h | 0 | <2 GB | 配置、日志、哈希、JSON 齐全 |

**首轮总计：1–3 天、约 2–3 GPU 小时、建议服务器空闲空间 ≥ 30 GB。** 空间大头是 Qwen3.5-0.8B 权重（~5 GB）与合成/获取的数据集（1–20 GB），远小于 RoboChallenge 的百 GB 量级。

## 1. 预期输入、输出与数据契约

### 1.1 必须取得/准备的内容

| 资产 | 来源 | 首轮范围 | 说明 |
| --- | --- | --- | --- |
| Qwen3.5-0.8B | YAML 指定 `./playground/Pretrained_models/Qwen3.5-0.8B` | 单模型 | ~5 GB；`huggingface-cli download Qwen/Qwen3.5-0.8B`（国内可用 `HF_ENDPOINT=https://hf-mirror.com`） |
| G1 数据集 | 三路线（§3） | 至少一个符合契约的 LeRobot v2.1 数据集 | 仓库**不随发**，`test_sonic` 非公开 HF 数据 |
| （可选）GR00T-WholeBodyControl | `NVlabs/GR00T-WholeBodyControl` | 仅路线 B 需要 | 外部仓库，symlink 到 `sdk_tools/` |

### 1.2 数据契约（必须逐项确认，来源 = `data_config.py` + `modality.json`）

**state 72D，由三段拼接：**

| 原始 parquet 键 | 子键（start:end） | 维度 |
| --- | --- | --- |
| `observation.state` | left_leg(0:6) + right_leg(6:12) + waist(12:15) + left_arm(15:22) + left_hand(22:29) + right_arm(29:36) + right_hand(36:43) | 43 |
| `observation.eef_state` | left_wrist_pos(0:3) + left_wrist_abs_quat(3:7) + right_wrist_pos(7:10) + right_wrist_abs_quat(10:14) | 14 |
| `observation.root_orientation` | (0:4) | 4 |
| `observation.projected_gravity` | (0:3) | 3 |
| `observation.cpp_rotation_offset` | (0:4) | 4 |
| `observation.init_base_quat` | (0:4) | 4 |
| **合计** | | **72** |

**action 78D：**

| 原始 parquet 键 | 子键 | 维度 |
| --- | --- | --- |
| `action.motion_token` | motion_token(0:64) | 64 |
| `teleop.left_hand_joints` | left_hand_joints(0:7) | 7 |
| `teleop.right_hand_joints` | right_hand_joints(0:7) | 7 |
| **合计** | | **78** |

其他关键契约：

| 项 | 约定 | 验收检查 |
| --- | --- | --- |
| 视频 | 单路 `observation.images.ego_view`，resize 224×224 | 单相机、可解码 |
| 语言 | `annotation.human.task_description`（task_index → tasks.jsonl） | tasks.jsonl 存在对应 prompt |
| 归一化 | state/action 全部连续键 **q99**（非 min_max） | `dataset_statistics.json` 中 q01/q99 可读 |
| embodiment | `EmbodimentTag.NEW_EMBODIMENT` | 反归一化键 = `new_embodiment` |
| horizon | 8（YAML `action_horizon: 8`，registry `action_indices=list(range(8))`） | 训练/部署一致 |
| 模型输出 | 服务端返回**未归一化** `actions`，shape `(T, 78)` | P3 分片 64/7/7 |

## 2. P0：前置环境与当前代码缺口

### 2.1 环境检查（不下载、不训练）

在 `starVLA` 仓库根目录执行，保存到 `Project_Analysis/evidence/g1_wholebody/00_preflight/`：

```bash
git rev-parse HEAD && git status --short
nvidia-smi
df -h .
conda env list
```

激活 StarVLA 环境后执行依赖探针（`torch` / `accelerate` / `cv2` / `pyarrow` / `lerobot` / `huggingface_hub`）。通过标准与 RoboChallenge 计划一致：RTX 5090、可用空间 ≥ 30 GB、真实 CUDA toolkit（DeepSpeed 需要真实 nvcc）。

### 2.2 当前代码缺口（必须处理，否则 P3 无法执行）

| 文件/位置 | 现状 | 处理 |
| --- | --- | --- |
| `step3_deployment/eval_files/local_self_test.py` | **不存在**，但被 `run_starvla_eval.sh` 与 step3 README 的 smoke 命令引用 | P3 需自写（见 §5.1），镜像 RoboChallenge 的 `local_self_test.py` 模式 |
| `step2_training/train_files/run_starvla_qwenoft_g1_sonic_train.sh` | 定义了 `data_root_dir`/`data_mix` 两个 shell 变量但**未传入 accelerate**（死变量） | 确认以 YAML 为准：`data_root_dir: playground/Datasets/UnitreeG1_WholeBody/lerobot`、`data_mix: unitree_g1_test_sonic` |
| 同脚本 `PYTHONPATH="${ROOT_DIR}/starVLA"` | 指向包目录而非 repo root，`import starVLA` 可能失败（除非 starVLA 已 pip install -e） | P0 验证：`PYTHONPATH="$PWD" python -c "import starVLA"`；若失败，改为 `PYTHONPATH="$ROOT_DIR"` 或依赖 editable 安装 |
| `gr00t/policy/server_client.py`（compat shim） | `from eval_files.model2unitree_g1_interface import ...` | 仅当 `step3_deployment` 在 PYTHONPATH 下生效；P3 直接以 repo root + deploy 目录为 PYTHONPATH 验证 |

> 与 RoboChallenge 不同：G1 的脚本 `cd` 回退层级（`../../../../../`）是**正确**的（train_files→step2_training→UnitreeG1_WholeBody→realRobots→examples→repo root），且不硬编码 conda 环境名，无需迁移补丁。

## 3. P1：数据获取（三路线，按优先级）

### 路线 A：获取 `test_sonic` 数据集（首选，若可得）

`test_sonic`（49 episodes / 56919 frames / fps 50 / 单 ego_view）在 step1 README 中被描述为「本地示例数据集已存在」，但本 clone 无 `playground/`。**先确认其来源**：

```bash
test -d playground/Datasets/UnitreeG1_WholeBody/lerobot/test_sonic && echo 'test_sonic present'
```

若需从维护者/内部获取，落盘到 `playground/Datasets/UnitreeG1_WholeBody/lerobot/test_sonic/`（`data/` + `videos/` + `meta/`）。**这是外部资产 gate：若拿不到，不要伪造数据集存在，转路线 C。**

### 路线 B：MuJoCo 仿真遥操作采集（可选，需 GR00T-WholeBodyControl + 输入设备）

按 step0 README 在仿真中采集（`run_sim_loop.py` + `launch_data_collection.py`）。**即使无 PICO VR，SonicStar 参考提供 `send_keyboard_cmd.py` 键盘驱动**，可绕过 VR 硬件。此路线工程量大（外部仓库 + MuJoCo + SONIC sim 环境），仅在需要「真实语义」数据时选择。

### 路线 C：合成数据集（默认「无外部依赖」路线）

编写一个一次性生成器 `Project_Analysis/evidence/g1_wholebody/01_data/gen_synthetic_test_sonic.py`，产出一个 schema 完全一致的 LeRobot v2.1 数据集（目标路径 `playground/Datasets/UnitreeG1_WholeBody/lerobot/test_sonic_synthetic/`）：

```text
test_sonic_synthetic/
├── data/chunk-000/episode_NNNNNN.parquet   # 见下方 schema
├── videos/chunk-000/observation.images.ego_view/episode_NNNNNN.mp4
└── meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json, embodiment.json}
```

parquet 每帧必须包含（维度见 §1.2）：

| 字段 | dtype/shape |
| --- | --- |
| `observation.state` | float32 [43] |
| `observation.eef_state` | float32 [14] |
| `observation.root_orientation` | float32 [4] |
| `observation.projected_gravity` | float32 [3] |
| `observation.cpp_rotation_offset` | float32 [4] |
| `observation.init_base_quat` | float32 [4] |
| `action.motion_token` | float32 [64] |
| `teleop.left_hand_joints` | float32 [7] |
| `teleop.right_hand_joints` | float32 [7] |
| `timestamp` / `frame_index` / `episode_index` / `index` / `task_index` | float32[1] / int64[1]… |

- 视频用 `cv2.VideoWriter` 生成 224×224 H.264 短片段（或复用 RoboChallenge 转换器的 image/video 写法）
- `modality.json` 直接复制仓库里的 `step2_training/train_files/modality.json`（已与 registry 契约一致）
- 合成数据的动作**无任务语义**，仅用于验证训练/部署链路，不得表述为「模型会做某任务」

> 路线 C 是「无真机 + 无外部数据」下的最小可行复现：它证明整条数据→训练→服务→动作契约链路在工程上是通的。路线 A 的 `test_sonic` 能补上「真实 78D 分布」但语义仍非本计划考核点。

## 4. P2：dataloader、模型与 1000-step 训练

### 4.1 训练前两个 gate

```bash
conda activate <starvla env>
export PYTHONPATH="$PWD:${PYTHONPATH}"

python starVLA/dataloader/lerobot_datasets.py \
  --config_yaml examples/realRobots/UnitreeG1_WholeBody/step2_training/train_files/starvla_qwenoft_g1_sonic.yaml

python starVLA/model/framework/VLM4A/QwenOFT.py \
  --config_yaml examples/realRobots/UnitreeG1_WholeBody/step2_training/train_files/starvla_qwenoft_g1_sonic.yaml
```

通过标准：dataloader 能采样单路 224×224 图像、72D state、78D action，且不报 `modality.json`/registry/video symlink 错误；QwenOFT forward 不出现 key/shape/模型路径错误。**任何 gate 失败先修数据契约，不启动 DeepSpeed。**

### 4.2 1000-step 单卡训练

确认 YAML 的 `base_vlm` 指向实际 Qwen3.5-0.8B 目录后执行：

```bash
conda activate <starvla env>
export CUDA_VISIBLE_DEVICES=0
export BATCH=1
export MAX_STEPS=1000
export SAVE_EVERY=200
export EVAL_EVERY=500
export LOG_EVERY=10
export WANDB_MODE=disabled
# 仅当有真实 CUDA toolkit 时设置 CUDA_HOME

bash examples/realRobots/UnitreeG1_WholeBody/step2_training/train_files/run_starvla_qwenoft_g1_sonic_train.sh \
  2>&1 | tee tmp/logs/g1_wholebody_train_1000step.log
```

预期输出：

```text
results/Checkpoints/starvla_qwenoft_g1_sonic_smoke/
  config.yaml
  config.full.yaml
  dataset_statistics.json
  checkpoints/steps_1000_pytorch_model.pt
```

> 注意：G1 训练脚本 `run_root_dir=./results/Checkpoints`（不是 `playground/Checkpoints`），与 RoboChallenge 不同。检查 `results/Checkpoints/starvla_qwenoft_g1_sonic_smoke/`。

通过标准：loss 连续有限；无 CUDA OOM（0.8B + 78D MLP head 在 5090 上 ~5–7 GB，余量充足）；checkpoint 与 `dataset_statistics.json` 同目录。记录 wall time、step/s、峰值显存、最终 loss。

## 5. P3：策略服务与动作切分验证（无真机）

### 5.1 自写 `local_self_test.py`（交付物）

在 `step3_deployment/eval_files/local_self_test.py` 新建（镜像 RoboChallenge 模式），职责：

1. `ModelClient(policy_ckpt_path=..., port=...)` 连接策略服务
2. 构造合成 GR00T 风格观测 dict：
   - `image`：224×224×3 合成 ndarray（或 `video.ego_view`）
   - `state`：72D 合成向量（直接 ndarray，走 `_extract_state` 的 reshape 分支）
   - `lang`：固定 prompt，如 `"pick the toy on the table, and put it into the box."`
3. 调用 `get_action(observation)`，断言返回 dict 含三个键且维度正确：

```python
assert out["action.motion_token"].shape == (T, 64)
assert out["action.left_hand_joints"].shape == (T, 7)
assert out["action.right_hand_joints"].shape == (T, 7)
```

4. 打印延迟，退出码 0 表示通过。

### 5.2 启动策略服务 + 验证

终端 A（策略服务，StarVLA 环境）：

```bash
cd /absolute/path/to/starVLA
bash examples/realRobots/UnitreeG1_WholeBody/step3_deployment/run_policy_server.sh \
  results/Checkpoints/starvla_qwenoft_g1_sonic_smoke/checkpoints/steps_1000_pytorch_model.pt 5694
```

终端 B（验证客户端）：

```bash
cd /absolute/path/to/starVLA
export PYTHONPATH="$PWD:$PWD/examples/realRobots/UnitreeG1_WholeBody/step3_deployment:${PYTHONPATH}"
python examples/realRobots/UnitreeG1_WholeBody/step3_deployment/eval_files/local_self_test.py \
  --ckpt-path results/Checkpoints/starvla_qwenoft_g1_sonic_smoke/checkpoints/steps_1000_pytorch_model.pt \
  --server-host 127.0.0.1 --server-port 5694
```

### 5.3 通过标准

1. 服务端成功加载 checkpoint 并返回 metadata（`unnorm_key` 解析为 `new_embodiment`）
2. 客户端 `get_action` 返回 `(T, 78)` 未归一化动作，切分为 `64/7/7` 三个键，维度正确
3. 数值为有限浮点，无 NaN/维度错误
4. 记录单轮推理延迟

这是「无真机复现」的最终最低验收：**证明 StarVLA 策略服务能对接 G1 观测契约并输出语义正确的 78D 动作。** 它不验证 SONIC 解码、平衡控制或真实执行。

## 6. P4：结果归档

```text
Project_Analysis/evidence/g1_wholebody/
  00_preflight/              # GPU、磁盘、commit、依赖版本
  01_data/                   # 路线 A/C 数据 schema 证据、合成生成器、占用统计
  02_training/               # YAML、训练脚本副本、loss/显存/step-s、ckpt hash
  03_deploy_action_split/    # local_self_test.py、服务端/client 日志、(T,78)→64/7/7 证据
  README.md                  # 实测日期、命令、成功/失败与未完成范围
```

保留 `dataset_statistics.json`（q99 反归一化依赖它）。明确记录数据来自路线 A 还是 C；若为 C，标注「合成数据，无任务语义」。

## 7. 失败分流与停止条件

| 失败点 | 首先检查 | 停止/继续规则 |
| --- | --- | --- |
| 数据不可得（路线 A） | 维护者/内部来源是否提供 `test_sonic` | 不伪造；转路线 C 合成数据集 |
| 合成数据 dataloader 失败 | `modality.json` 子键、parquet 字段名、video symlink、维度 | 对照 §1.2 契约逐项修 |
| forward gate 失败 | `base_vlm` 路径、`action_dim=78`/`state_dim=72`、模型加载 | 未通过 gate 不得启动 DeepSpeed |
| 训练 OOM | batch、CUDA/DeepSpeed、显存 | 固定 0.8B，batch=1；不改用 3B/4B |
| `import starVLA` 失败 | P0 的 `PYTHONPATH` 指向、是否 editable 安装 | 改为 repo root 或 pip install -e |
| 服务端 metadata/unnorm_key 异常 | `dataset_statistics.json` 是否与 checkpoint 同 run 目录 | 缺失则重新训练或补拷 |
| 动作切分维度错 | `ModelClient._split_action` 的 64/7/7 边界、服务端返回 key（`data.actions`） | 抓包确认服务端响应结构 |

## 8. 最终交付标准

完成后应在干净 shell 中按序重新执行并通过：

1. 数据 metadata/shape 检查（72D state / 78D action / 单相机 / q99）；
2. dataloader gate；
3. QwenOFT forward gate；
4. 1000-step checkpoint 生成（`steps_1000_pytorch_model.pt` + `dataset_statistics.json`）；
5. 策略服务启动 + `ModelClient` 合成观测 → `(T,78)` 动作正确切分 `64/7/7`；
6. 汇总：数据来源（A/C）、峰值显存、step/s、单轮推理延迟、未完成范围。

向老师的建议表述：**「在无实体机器人的条件下，完成 StarVLA UnitreeG1_WholeBody 示例的数据契约确认、QwenOFT(78D SONIC 动作) 训练与 checkpoint 加载、策略服务启动，并验证 StarVLA→G1 适配器对合成观测输出语义正确的 64+7+7 动作块。」** 需同时注明：本计划未涉及实体 G1、SONIC/WBC 解码与 PICO 遥操作，因此不报告真实机器人闭环或任务成功率；数据若走合成路线，动作不具任务语义。

## 附：与 RoboChallenge 计划的关键差异

| 维度 | RoboChallenge | UnitreeG1_WholeBody |
| --- | --- | --- |
| 数据 | 公开 HF，`--only` 单任务可下 | **不随发/非公开**，需三路线决策 |
| 数据量级 | 9.4 GB/任务，全量 1 TB | `test_sonic` ~数 GB，合成可 <1 GB |
| 动作契约 | 8D ee_pose+gripper | **78D SONIC latent**（64+7+7） |
| 归一化 | min_max | **q99** |
| 模型/显存 | 0.8B ~5 GB | 0.8B ~5–7 GB（78D MLP head 略大） |
| 免真机验证 | mock server（上游 HTTP 协议） | **无 mock server**，改为自写合成观测 + 动作切分 |
| 代码缺口 | 2 处 import 路径 + 2 处 cd 层级 | 缺 `local_self_test.py` + `PYTHONPATH` 疑点 |
| 完成语义 | 协议闭环（GET/POST） | 动作契约切分（64/7/7） |
