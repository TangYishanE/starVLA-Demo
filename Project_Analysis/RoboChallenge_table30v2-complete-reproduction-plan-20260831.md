# RoboChallenge Table30v2 完整复现执行计划

**目标项目：** `examples/realRobots/RoboChallenge_table30v2`，固定先复现单任务 **UR5 / `shred_paper`**。  
**运行边界：** 远程单张 RTX 5090 32 GiB、无实体机器人。  
**本计划的完成定义：** Table30v2 公开数据下载并转换为 LeRobot v2.1；QwenOFT 100-step checkpoint 训练与重载通过；合成观测 self-test 产生合法 8×8 动作块；连接 RoboChallenge 官方 inference repository 的 mock server，连续读取状态并 POST 动作。

> 这是一份“公开数据—训练—模型—挑战协议 mock”完整流程复现计划。当前 StarVLA 目录明确将线上 production `job_loop` 标为 TODO；没有获得 RoboChallenge 线上任务和实体 UR5 时，**不得**表述为线上提交成功或真实机器人闭环成功。

## 0. 计划总览

### 0.1 固定的首轮配置

| 项目 | 首轮选择 | 原因 |
| --- | --- | --- |
| 任务 | `shred_paper` | 当前 YAML、registry 和训练脚本都为该单任务准备；避免一次下载/转换 30 个任务 |
| 机器人 | `ur5` | 两个 RGB 相机、7D joint+gripper state、8D EE-pose+gripper action，路径最完整 |
| 框架 | `QwenOFT` | 官方 walkthrough 配置，Qwen3.5-0.8B，单卡风险最低 |
| 模型配置 | 224×224，horizon=8，先 `BATCH=1`，`MAX_STEPS=100` | 先验证工程闭环；100 step 不是性能训练 |
| 评测 | local self-test + upstream mock server | 不需要机器人；能验证 checkpoint、归一化、图像/状态输入、HTTP action 输出 |
| 不做的内容 | 30 task 全下载、全参数大模型训练、线上 submission、实体 UR5 执行 | 超出当前算力/资产与本次复现目标 |

### 0.2 里程碑、时间与空间

| 阶段 | 主要工作 | 日历时间 | GPU 时间 | 峰值新增空间 | 可判定的通过条件 |
| --- | --- | ---: | ---: | ---: | --- |
| P0 | 服务器/代码路径/依赖检查和兼容补丁 | 0.5–1 天 | 0 | <5 GB | Python import、CUDA、脚本路径均正确 |
| P1 | 下载 `shred_paper` 并转 LeRobot | 2–8 h | 0 | **40–80 GB**，建议预留 100 GB | metadata、Parquet、两路视频、维度完整 |
| P2 | dataloader、模型 forward 与 100-step 训练 | 2–6 h | 1–3 h | +10–20 GB | checkpoint 与 `dataset_statistics.json` 生成 |
| P3 | checkpoint local self-test | 0.5–1 h | <1 h | 可忽略 | 3 次推理均输出 `(8,8)`，无 NaN/维度错误 |
| P4 | 上游 mock server 与连续协议验证 | 2–6 h | 1–2 h | +5–20 GB（若需上游回放数据） | 能连续 GET state / POST `(8,8)` action，正常结束 |
| P5 | 结果整理与可复现归档 | 2–4 h | 0 | <2 GB | 配置、日志、哈希、JSON、精选视频齐全 |

**首轮总计：1–3 天、约 2–6 GPU 小时、建议服务器空闲空间至少 100 GB。** 当前项目根目录的 100 GB 配额可勉强执行单任务；若下载后的原始任务超过预期或上游 mock 记录另需下载，应先扩容到 150 GB。不要运行 `download_table30v2.sh`：它会下载全部 30 个任务。

空间估算说明：转换器对视频创建符号链接，不复制 MP4；但转换完成前会同时保留 raw task、临时 tar 分片与少量 Parquet。确切数据集大小应在下载前通过 Hugging Face 文件清单记录；本计划不把未实测的网页大小当作确定值。

## 1. 预期输入、输出与模型数据契约

### 1.1 必须下载/取得的内容

| 资产 | 来源 | 首轮所需范围 | 预计空间/注意事项 |
| --- | --- | --- | --- |
| Table30v2 raw task | Hugging Face：`RoboChallenge/Table30v2` | 仅 `shred_paper` | 大小下载前动态确认；raw 视频必须保留，因为 LeRobot 输出会链接到它 |
| Qwen3.5-0.8B | YAML 指定的模型路径；按 StarVLA 的模型下载方式取得 | `./playground/Pretrained_models/Qwen3.5-0.8B` | 预留 5–10 GB（含下载/cache 裕量） |
| RoboChallengeInference | `RoboChallenge/RoboChallengeInference` 的 `cvpr` 分支 | 上游 mock server 与 `InterfaceClient` | 代码/依赖较小；必须独立 Python 环境 |
| 上游 mock 回放记录 | 上游仓库 `20260413/...` 或官方提供的对应 record 包 | 一个与 `ur5` 匹配的记录目录 | **单独 gate**，不能假设 Table30v2 raw task 一定符合 mock server 的回放目录格式 |

官方 upstream mock 设置示例使用 `20260413/ur5/arrange_fruits`，并要求在 `mock_settings.py` 中选择唯一的 `ROBOT_TAG` 与 `RECORD_DATA_DIR`。[上游 inference README](https://github.com/RoboChallenge/RoboChallengeInference/tree/cvpr)；[Table30v2 数据集页](https://huggingface.co/datasets/RoboChallenge/Table30v2)。

### 1.2 数据契约（必须逐项确认）

| 字段 | UR5 `shred_paper` 约定 | 验收检查 |
| --- | --- | --- |
| 图像 | `cam_global`、`cam_arm` 两路 RGB；训练 resize 至 224×224 | 两路 MP4 存在、可解码、相机顺序不变 |
| state | 7D = 6 个 joint position + 1 gripper width | `observation.state` shape 为 `[7]` |
| action | 8D = 7D quaternion EE pose + 1 gripper width | `action` shape 为 `[8]` |
| 对齐 | state 取 `t-1`、action 取 `t` | 使用项目转换器，不自行更改索引语义 |
| 语言 | `task_info.json` 内 prompt | `tasks.jsonl` 中存在 `shred_paper` 对应 task |
| 归一化 | state/action 均 `min_max` | 训练产生的 `dataset_statistics.json` 中 key 与维度可读 |
| 模型输出 | `normalized_actions`，首轮取 8 个步骤 | self-test 与 mock 均为 `(8,8)` |

## 2. P0：前置环境与当前 main 兼容补丁

### 2.1 服务器环境检查（不下载、不训练）

在 `starVLA` 仓库根目录执行并保存输出到 `Project_Analysis/evidence/robochallenge/00_preflight/`：

```bash
git rev-parse HEAD
git status --short
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h .
df -ih .
conda env list
which nvcc || true
nvcc --version || true
```

通过标准：GPU 是 RTX 5090、可见显存约 32 GiB；训练目录所在分区可用空间至少 100 GB；驱动可用；存在可用于 DeepSpeed 的真实 CUDA toolkit，而非仅有 stub `nvcc`。

激活 `starVLA_dev` 后执行依赖探针：

```bash
conda activate starVLA_dev
python - <<'PY'
import torch, accelerate, cv2, pyarrow
from huggingface_hub import HfApi
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0))
print('bf16:', torch.cuda.is_bf16_supported())
print('accelerate:', accelerate.__version__)
print('cv2:', cv2.__version__)
print('pyarrow:', pyarrow.__version__)
print('hf api:', type(HfApi()).__name__)
PY
```

若缺包，只在 `starVLA_dev` 环境补齐项目实际缺失的包；不要在同一环境安装上游 RoboChallengeInference 的 requirements，避免覆盖 StarVLA 的 PyTorch/CUDA 依赖。

### 2.2 当前 main 必做代码路径修复

当前目录从旧布局迁到 `examples/realRobots/` 后，RoboChallenge 评测层保留了旧路径。正式执行前建立一个独立提交或补丁，做以下四处更改：

| 文件 | 当前问题 | 应修改为 |
| --- | --- | --- |
| `eval_files/local_self_test.py` | 从 `examples.simBenchmarks.RoboChallenge_table30v2...` 导入 | `examples.realRobots.RoboChallenge_table30v2.eval_files.model2robochallenge_interface` |
| `eval_files/test_with_mock_server.py` | 同样从旧 `simBenchmarks` 路径导入 | 同上 |
| `eval_files/run_self_test.sh` | `cd "$(dirname "$0")/../../.."` 只到 `examples/` | `cd "$(dirname "$0")/../../../.."`，回到 repo root |
| `eval_files/run_test_with_mock.sh` | 同样少回退一级 | 同上 |

修改后运行：

```bash
bash -n examples/realRobots/RoboChallenge_table30v2/eval_files/run_self_test.sh
bash -n examples/realRobots/RoboChallenge_table30v2/eval_files/run_test_with_mock.sh
PYTHONPATH="$PWD" python -c "from examples.realRobots.RoboChallenge_table30v2.eval_files.model2robochallenge_interface import RoboChallengePolicy; print('import OK')"
```

通过标准：两项 `bash -n` 无输出且返回 0；Python import 成功。不要跳过此阶段，否则后续 self-test 会在加载模型之前因 import/path 失败。

### 2.3 上游 mock 环境准备

在 StarVLA 目录外创建隔离环境：

```bash
mkdir -p "$HOME/playground/Code"
cd "$HOME/playground/Code"
git clone -b cvpr https://github.com/RoboChallenge/RoboChallengeInference.git
cd RoboChallengeInference
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import fastapi, uvicorn, cv2, requests; print('mock env OK')"
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
```

上游 `requirements.txt` 当前包含 `requests numpy httpx loguru opencv-python fastapi pillow uvicorn`。保存分支与 commit，以免后续 API 行为不可复现。

## 3. P1：单任务数据下载、转换与验证

### 3.1 仅下载 `shred_paper`

从 `starVLA` 仓库根目录执行。所有路径必须位于有足够空间的服务器工作盘；如有单独数据盘，替换为绝对路径。

```bash
conda activate starVLA_dev
export RAW_ROOT="$PWD/playground/Datasets/RoboChallenge_table30v2/raw"
export LEROBOT_ROOT="$PWD/playground/Datasets/RoboChallenge_table30v2/lerobot"
mkdir -p "$RAW_ROOT" "$LEROBOT_ROOT" tmp/logs

python examples/realRobots/RoboChallenge_table30v2/train_files/download_table30v2.py \
  --raw-root "$RAW_ROOT" \
  --only shred_paper 2>&1 | tee tmp/logs/robochallenge_download_shred_paper.log
```

下载结束后先记录真实占用，不急于训练：

```bash
du -sh "$RAW_ROOT"
find "$RAW_ROOT/shred_paper" -type f | wc -l
test -f "$RAW_ROOT/shred_paper/meta/task_info.json"
```

失败处理：若 Hugging Face 鉴权/网络失败，先记录完整报错和 `huggingface_hub` 版本；不要改用全量 30-task 脚本。若空间不足，停止下载并清理**本次产生的 tar 分片**后扩容，再重新运行同一幂等下载器。

### 3.2 转换到 LeRobot v2.1

```bash
python examples/realRobots/RoboChallenge_table30v2/train_files/convert_robochallenge_to_lerobot.py \
  --raw-root "$RAW_ROOT" \
  --task shred_paper \
  --out-root "$LEROBOT_ROOT" 2>&1 | tee tmp/logs/robochallenge_convert_shred_paper.log
```

**重要：** 不删除 `$RAW_ROOT/shred_paper`。转换器为两路视频建立符号链接，raw 视频被删除后训练与评测会失效。

### 3.3 数据验收命令

```bash
export DATASET="$LEROBOT_ROOT/shred_paper"
python - <<'PY'
import json
from pathlib import Path
root = Path(__import__('os').environ['DATASET'])
info = json.loads((root/'meta/info.json').read_text())
modality = json.loads((root/'meta/modality.json').read_text())
print('episodes:', info['total_episodes'])
print('frames:', info['total_frames'])
print('fps:', info['fps'])
print('features:', sorted(info['features']))
print('state:', modality['state'])
print('action:', modality['action'])
print('video:', modality['video'])
assert info['total_episodes'] > 0 and info['total_frames'] > 0
assert info['features']['observation.state']['shape'] == [7]
assert info['features']['action']['shape'] == [8]
PY

find "$DATASET/videos" -type l -print -exec test -e {} \; | head -20
```

通过标准：episode/frame 为正；两个 video key 存在，链接目标有效；state/action 分别为 7/8 维；`tasks.jsonl` 里的 prompt 与训练 prompt 语义一致。将 `info.json`、`modality.json`、一个 Parquet 的 schema 和一帧解码截图作为报告证据。

## 4. P2：dataloader、模型与 100-step 训练

### 4.1 训练前两个 gate

```bash
conda activate starVLA_dev
export PYTHONPATH="$PWD:${PYTHONPATH}"

python starVLA/dataloader/lerobot_datasets.py \
  --config_yaml examples/realRobots/RoboChallenge_table30v2/train_files/starvla_qwenoft_robochallenge_table30v2.yaml

python starVLA/model/framework/VLM4A/QwenOFT.py \
  --config_yaml examples/realRobots/RoboChallenge_table30v2/train_files/starvla_qwenoft_robochallenge_table30v2.yaml
```

通过标准：dataloader 能采样两路 224×224 图像、7D state、8D action；模型 forward 不出现 key、shape、precision 或模型路径错误。任何 gate 失败都先修数据契约，不启动 DeepSpeed。

### 4.2 100-step 单卡训练

确认 YAML 的 `base_vlm` 指向实际 Qwen3.5-0.8B 目录后执行：

```bash
conda activate starVLA_dev
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
export BATCH=1
export MAX_STEPS=100
export SAVE_EVERY=100
export EVAL_EVERY=1000
export LOG_EVERY=5
export WANDB_MODE=disabled
# 仅当检测到真实 CUDA toolkit 时设置 CUDA_HOME；不要保留脚本里的集群示例路径。
export CUDA_HOME=/path/to/your/cuda

bash examples/realRobots/RoboChallenge_table30v2/train_files/run_robochallenge_table30v2.sh \
  2>&1 | tee tmp/logs/robochallenge_train_100step.log
```

首轮使用 batch=1 是为了优先排除显存问题；如果 100 step 稳定且峰值显存有余量，再把 batch 改为 2 做一次可选对照。不要因 0.8B 小模型可运行就切换到大模型或增加到完整训练日程。

预期输出：

```text
playground/Checkpoints/
  robochallenge_table30v2_qwenoft_shred_paper_100step/
    config.yaml
    config.full.yaml
    dataset_statistics.json
    checkpoints/steps_100_pytorch_model.pt
```

通过标准：loss 连续且为有限值；无 CUDA OOM；`steps_100_pytorch_model.pt` 与 `dataset_statistics.json` 同一 run 目录下。记录 wall time、平均 step/s、最大显存与最终 loss；这些是本机实测，不应预先填入报告。

## 5. P3：checkpoint 离线 self-test

使用 P0 修正后的 launcher，或先直接调用 Python：

```bash
export CKPT="$PWD/playground/Checkpoints/robochallenge_table30v2_qwenoft_shred_paper_100step/checkpoints/steps_100_pytorch_model.pt"
PYTHONPATH="$PWD" python examples/realRobots/RoboChallenge_table30v2/eval_files/local_self_test.py \
  --checkpoint "$CKPT" \
  --robot_tag ur5 \
  --prompt "shred the paper" \
  --n_warmup 1 \
  --n_runs 3 2>&1 | tee tmp/logs/robochallenge_local_self_test.log
```

预期结果：模型加载成功；三次推理均打印 action shape `(8, 8)`；输出的八维动作是有限浮点数；日志包含每次毫秒级 latency 及平均 latency。合成图像上的动作内容没有任务语义，不应解释为“模型会撕纸”；此步骤只验证 checkpoint、图像/state 解析、min-max normalization 和 action unnormalization。

## 6. P4：上游 mock server 协议闭环

### 6.1 mock 回放数据 gate

先在上游仓库检查是否有可用 UR5 回放目录：

```bash
cd "$HOME/playground/Code/RoboChallengeInference"
test -d 20260413/ur5/arrange_fruits && echo 'upstream UR5 record found'
find 20260413/ur5/arrange_fruits -maxdepth 3 -type f | head
```

若该目录不存在，**停止在此 gate，不要伪造 mock 成功。** 需要从 RoboChallenge 官方渠道取得与 `mock_robot_server.py` 匹配的 `20260413` 记录包，或与维护方确认如何将 Table30v2 raw 数据转换成该 mock server 所需布局。当前 StarVLA Example 虽引用该目录，但未提供自动下载器。

数据存在时，编辑 `$RC_REPO/mock_server/mock_settings.py`，只保留一组：

```python
ROBOT_TAG = 'ur5'
RECORD_DATA_DIR = '../20260413/ur5/arrange_fruits'
```

注意：训练 prompt 是 `shred the paper`，而 mock 回放可先用官方提供的 `arrange_fruits` UR5 记录来验证协议；这时只能证明**跨任务协议连通**，不能评估任务策略。若获得 `shred_paper` 对应 mock 记录，才将 prompt 和记录都统一为 `shred_paper`。

### 6.2 启动 mock 与 StarVLA client

终端 A（上游独立虚拟环境）：

```bash
cd "$HOME/playground/Code/RoboChallengeInference"
source .venv/bin/activate
cd mock_server
python3 mock_robot_server.py
```

终端 B（StarVLA 环境）：

```bash
cd /absolute/path/to/starVLA
conda activate starVLA_dev
export CKPT="$PWD/playground/Checkpoints/robochallenge_table30v2_qwenoft_shred_paper_100step/checkpoints/steps_100_pytorch_model.pt"
export RC_REPO="$HOME/playground/Code/RoboChallengeInference"
PYTHONPATH="$PWD" python examples/realRobots/RoboChallenge_table30v2/eval_files/test_with_mock_server.py \
  --checkpoint "$CKPT" \
  --robot_tag ur5 \
  --prompt "shred the paper" \
  --rc_repo "$RC_REPO" \
  --duration 0.05 \
  --max_wait 60 2>&1 | tee tmp/logs/robochallenge_mock_60s.log
```

### 6.3 mock 阶段的预期结果与判定

每次循环应满足：

1. `GET /state.pkl` 返回 `state=normal`、两个 PNG 相机和 7D joint/gripper state；
2. StarVLA 推理成功，日志打印 `(8,8)` 的 action chunk 和单轮 inference latency；
3. `POST /action` 使用 `leftpos`，body 为 8D EE pose+gripper 动作序列，`duration=0.05`；
4. 在 `max_wait=60` 前至少完成连续多轮，无 `pending_actions` 永久堆积、HTTP error、NaN 或 action shape 错误；
5. 结束时 client 调用 `end_motion()`，日志包含迭代数与 elapsed time。

这是最终“项目复现成功”的最低验收。它证明 StarVLA policy 能接入 RoboChallenge I/O 协议；不证明动作在真实 UR5 上安全或能完成任务。

## 7. P5：结果归档与汇报材料

每个阶段结束即归档下列内容：

```text
Project_Analysis/evidence/robochallenge/
  00_preflight/                  # GPU、磁盘、commit、依赖版本
  01_data/                       # info/modality/schema、占用统计、转换日志
  02_training/                   # YAML、训练脚本副本、loss/显存/step-s 日志、ckpt hash
  03_local_self_test/             # 三次 latency、(8,8) 输出、错误为零证明
  04_mock_protocol/               # mock 配置、两端 commit、60s client/server 日志
  README.md                       # 实测日期、命令、成功/失败和未完成范围
```

保留 `dataset_statistics.json`，因为 self-test 和 action unnormalization 依赖它。视频只保留 1–2 个代表 episode；不要删除 raw 视频链接目标。训练完成后可删除 Hugging Face 的临时 tar 分片和不再需要的下载 cache 副本，但清理前先以 `readlink -f` 检查所有 LeRobot 视频链接仍有效。

## 8. 失败分流与停止条件

| 失败点 | 首先检查 | 停止/继续规则 |
| --- | --- | --- |
| 下载失败或空间不足 | Hugging Face 权限、网络、`df -h`、raw task 实际大小 | 空间不足立刻停止；不要转全量下载 |
| 转换失败 | `task_info.json`、`states.jsonl`、MP4、`cv2`、`pyarrow` | 修数据/依赖后可幂等重跑单任务 |
| dataloader/forward 失败 | `modality.json`、registry、video symlink、模型路径 | 未通过 gate 不得启动 DeepSpeed |
| 训练 OOM | batch、CUDA/DeepSpeed、显存占用 | 固定 0.8B，batch 降至 1；不改用全参数 3B/4B |
| self-test import/path 错 | P0 的四处迁移补丁、`PYTHONPATH=$PWD` | 修复后重跑，无需重训 |
| mock 服务器无记录数据 | `20260413` record 目录是否存在 | 记录为外部资产阻塞；完成 P0–P3 并报告 mock 未验证 |
| mock action 错误 | `ur5`、两相机顺序、`leftjoint` state 与 `leftpos` post、8D 输出 | 先做单次 GET/POST 抓包，再恢复循环 |

## 9. 最终交付标准

完成后应能在干净 shell 中按以下顺序重新执行并通过：

1. 数据 metadata/shape 检查；
2. dataloader gate；
3. QwenOFT forward gate；
4. checkpoint self-test（三次 `(8,8)` 动作）；
5. 60 秒 mock protocol loop；
6. 汇总实际下载大小、峰值空间、GPU 峰值显存、平均 step/s、平均推理延迟。

向老师的建议表述：**“在公开 Table30v2 单任务数据上完成 StarVLA QwenOFT 的训练、checkpoint 加载、RoboChallenge 本地动作推理与官方 mock 协议闭环复现。”** 需同时注明：线上 submission adapter 仍是 StarVLA 当前 README 的 TODO，且未拥有实体 UR5，因此未报告物理任务成功率。
