# EgoVLA 无真机复现执行计划

**目标项目：** `examples/realRobots/EgoVLA`，在**无实体 G1 机器人、无 Isaac-Lab 仿真、无 GR00T-WBC-Bridge** 条件下复现其「框架推理 → 48D 相机帧动作 → 解码 → G1 关节目标动作」链路（模型侧闭环），并评估「数据契约 → 训练」链路（训练侧，二期）。

**运行边界：** 远程单张 RTX 5090 32 GiB（共享）、无实体机器人、无遥操作采集设备、无仿真/桥接器。

**本计划的完成定义（首轮）：** 用 EgoVLA **已发布的开源权重**（`rchal97/*` 的 VILA 式 checkpoint）加载 starVLA 的 `EgoVLA` 框架并输出 `(T,48)`；`decode.py` 将 48 维解码为骨盆系腕部 EE 位姿 + Inspire 手部；`server_egovla_g1.py` 在 `:5555` 上对合成 GR00T 风格观测输出形状正确、IK 收敛、手部可articulate 的 G1 关节目标动作 dict。

> 这是一份「框架推理—解码—GR00T 协议动作契约」的无真机复现计划。**没有实体 G1、Isaac-Lab G1 仿真和 GR00T-WBC-Bridge 时，不得**表述为真实机器人闭环成功、抓取成功率或平衡控制验证。本计划最多验证到「EgoVLA 策略服务能对 G1 观测契约输出语义正确的 48 维相机帧动作，并解码为结构正确的 G1 关节目标动作」，这是部署链路的模型侧闭环。
>
> ⚠️ 与 RoboChallenge / UnitreeG1 / Franka / Realman 四个示例**根本不同**：EgoVLA 是 `realRobots/` 里**唯一有开源权重**的示例（`rchal97/egovla_base_vlm` 等），且 `eval_files/` 已经是一套**完整、可直接运行的 GR00T(ZMQ) 协议部署链**（server + policy + decode + pinocchio IK + 相机变换 + URDF）。因此本计划把「**零训练部署复现**」作为 P1 主交付（这是其它示例做不到的），而训练侧反而是一个**空骨架**（占位 YAML、无 `data_registry/`、无 `modality.json`、无训练脚本、无公开 48 维 bimanual 相机帧 MANO 数据），作为 P2 二期、且难度最高。

## 0. 计划总览

### 0.1 固定的首轮配置

| 项目 | 首轮选择 | 原因 |
| --- | --- | --- |
| 权重 | **EgoVLA 开源 checkpoint**（VILA 式 `ego_vla_checkpoint/ckpt-6720`，来源 `rchal97/*`），**不训练** | EgoVLA 是唯一有开源权重的 realRobots 示例；框架已 vendored 且 README 声明 6/6 projector、98/98 traj_decoder 键严格匹配 |
| 框架 | `EgoVLA`（SigLIP-384 + Qwen2-1.5B + `mlp_downsample` projector + `EgoVLATrajDecoder`） | `@FRAMEWORK_REGISTRY.register("EgoVLA")`，`framework.name: EgoVLA` |
| 动作契约 | **48 维相机帧动作** = 双手 × (3 wrist-trans + 6 rot6d + 15 MANO)，`action_horizon=30` | `action_dim: 48`、`action_horizon: 30`，与 decode.py 切片一致 |
| 部署 | `server_egovla_g1.py`（GR00T **ZMQ** 协议 `:5555`）+ 合成 GR00T 风格观测 → G1 关节目标动作 | 不需要实体机器人/仿真/桥接器；`ZmqGr00tPolicyServer` 契约已对齐 |
| 评测 | 三层验证阶梯：① FK-IK 自测（纯 pinocchio）→ ② `decode.py` 合成 `(T,48)` 解码 → ③ 完整 server `get_action` | 逐层递进，隔离 checkpoint / MANO / IK 三类依赖 |
| 训练（二期） | freeze VLM / LoRA，Qwen2-1.5B（~2B），先契约归一 + registry/modality 补丁 + 合成 48 维数据 | 训练侧是空骨架；2B 全参 FT 16–20GB，仅 5090 可做，4070 Ti 必须 freeze/LoRA |
| 不做的内容 | 真实 G1 执行、Isaac-Lab G1 red-ball 仿真、GR00T-WBC-Bridge 联调、PICO 遥操作采集、任务成功率 | 超出「无真机」边界；仿真/桥接器为外部组件 |

### 0.2 里程碑、时间与空间

| 阶段 | 主要工作 | 日历时间 | GPU 时间 | 峰值新增空间 | 可判定的通过条件 |
| --- | --- | ---: | ---: | ---: | --- |
| P0 | 环境检查 + license-gated 资产获取 + 依赖补装 + 代码缺口审计 | 0.5–1 天（+MANO 审批 1–3 天） | 0 | checkpoint 2–4 GB | import、CUDA、pinocchio/smplx/chumpy、资产齐全 |
| P1 | **部署侧零训练模型闭环**（主交付） | 0.5–1 天 | <1 h（纯推理） | 可忽略 | `(T,48)` 输出 + FK-IK 收敛 + 解码 + server `get_action` 结构正确 |
| P2 | 训练侧（二期，硬缺口）：契约归一 + registry/modality 补丁 + 合成数据 + freeze/LoRA smoke | 1–3 天 | 1–4 h | +5–10 GB | dataloader/forward gate 通过、N-step checkpoint + `dataset_statistics.json` |
| P3 | 结果归档 | 1–2 h | 0 | <2 GB | 配置、日志、哈希、JSON 齐全 |

**首轮总计（仅 P0+P1+P3，不含二期训练）：1–3 天日历（含 MANO 审批等待）、<1 GPU 小时（纯推理）、建议服务器空闲空间 ≥ 10 GB。** 这是所有 realRobots 示例里 **GPU 成本最低** 的首轮（零训练），但其首要 gate 是 **license-gated 资产（MANO 模型审批 1–3 天）**，而非其它示例的数据 gate。空间大头是 EgoVLA checkpoint（SigLIP-384 + Qwen2-1.5B + traj decoder，约 2–4 GB）。

## 1. 预期输入、输出与数据契约

### 1.1 必须取得/准备的内容

| 资产 | 来源 | 首轮范围 | 说明 |
| --- | --- | --- | --- |
| EgoVLA checkpoint（VILA 式） | `rchal97/egovla_base_vlm`（或 `rchal97/ego_vla_human_video_pretrained` / `rchal97/egovla`）；亦可用 `EgoVLA_Release/checkpoints/ego_vla_checkpoint/ckpt-6720` | 部署侧必需 | 含 `{llm, vision_tower, mm_projector, traj_decoder}` 四个 HF 子模型；`framework.qwenvl.base_vlm` 指向该目录 |
| MANO 手模型 | <https://mano.is.tue.mpg.de>（研究许可，需注册审批） | 部署侧 decode 必需 | `mano_v1_2/models/MANO_{LEFT,RIGHT}.pkl`，放 `<EgoVLA_Release>/mano_v1_2/models/` |
| hand-retarget net | EgoVLA release | 部署侧 decode 必需 | `hand_actuation_net.pth`（input 30 → output 24），复制到 `eval_files/assets/`（已被 `.gitignore` 排除 `*.pth`） |
| G1 URDF | `eval_files/assets/g1_29dof_with_hand.urdf`（已随仓库分发） | pinocchio 只需要文本 | 无需 mesh，29 DoF（Dex3 手） |
| 训练数据（二期） | 三路线（§4.1） | 仅 P2 训练需要 | H1 数据私有；**无公开 48 维相机帧 MANO bimanual 数据集**，默认合成 |

> **首要外部 gate = MANO 模型审批（1–3 天）**。checkpoint 与 hand net 走 `rchal97/*` HF 仓库（国内 `HF_ENDPOINT=https://hf-mirror.com`）或 `EgoVLA_Release` checkout。在 MANO 审批完成前，P1 的 ① FK-IK 自测 与 ② checkpoint→`(T,48)` 两条子路可以先行（均不依赖 MANO），只有 decode 手部段与完整 server 需要 MANO + hand net。

### 1.2 模型动作契约（48 维，相机帧，来源 = `EgoVLA.py` + `EgoVLA_ActionHeader.py` + `decode.py`）

**action 48 维（每步）：**

| 切片（decode.py 为准） | 内容 | 维度 |
| --- | --- | --- |
| `pred[:, 0:6]` | 双手 wrist 平移 `wrist_trans`（左 3 + 右 3） | 2×3 |
| `pred[:, 6:36]` | 双手 MANO 手姿 `mano_hand`（左 15 + 右 15） | 2×15 |
| `pred[:, 36:48]` | 双手 wrist 旋转 `rot6d`（左 6 + 右 6，HMR2 约定） | 2×6 |
| **合计** | | **48** |

> ⚠️ **48 维内部排布一致性必须在 P1 用真实 checkpoint 实测确认**（本计划列为 P1 第一等验证项）。`EgoVLA_ActionHeader.py` 文档写「decoder 每步发射 `[left(3+6+15), right(3+6+15)]`」，即每只手内部是 (trans, rot6d, mano)；而 `decode.py` 按**跨手模态分组** `[trans(2×3) | mano(2×15) | rot6d(2×6)]` 切片。二者排布**不一致**，且 `EgoVLA.py` 的 `_run` 直接返回 `result["pred"]` **没有做重排**。若框架确实未重排，则 `decode.py` 的切片语义是错的，需在 P1 修正或确认（这是上游 vendoring 时最可能遗留的坑）。

**其他关键契约：**

| 项 | 约定 | 验收检查 |
| --- | --- | --- |
| 图像 | `List[PIL]`，384×384（SigLIP-384），单路 head-cam；YAML `obs_image_size: [384,384]` | 单相机、可经 SigLIP processor |
| 语言 | `lang: str`（无 `<image>` 前置，框架自拼 text + image token） | prompt 进入 tokenizer |
| proprio（相机帧，`sep_proprio: true`） | `proprio_3d` 2×3、`proprio_rot` 2×3（rotvec）、`proprio_hand_finger_tip` 2×5×3；plain `proprio` 16 维在 sep 分支不参与 | 缺失默认全零，模型仍可跑 |
| horizon | 30（`action_horizon: 30`，forward 取 `actions[:, -30:, :]`） | 训练/部署一致 |
| action 语义 | `action_type: abs_qpos`、`action_mode: abs`；相机帧（非关节目标） | 与 decode 的相机→骨盆变换一致 |
| 归一化 | MANO 15 维用 decode.py 的 denorm 范围 `_MANO_MIN/_MANO_MAX`；wrist trans / rot6d 的归一化尺度需 P2 从 checkpoint 的 `dataset_statistics.json` / 上游 EgoVLA 训练配置确认 | P1 只做部署（框架直接输出原始 decoder 值），P2 训练时对齐 |

### 1.3 部署观测/动作契约（GR00T N1.7 / REAL_G1 profile，来源 = `egovla_g1_policy.py`）

**观测（`_obs_to_example` 消费）：**

| 键 | 内容 |
| --- | --- |
| `video["ego_view"]` | `(1,T,H,W,3)` uint8，取 `[0,-1]` 帧 → resize 384 |
| `state.left_arm` / `right_arm` | 7-DoF 关节 |
| `state.left_wrist_eef_9d` / `right_wrist_eef_9d` | 9 维（pos 3 + rot6d 6），骨盆系 → 相机系 proprio |
| `language["annotation.human.task_description"]` | 语言指令（缺省回退 `place the red ball in the box`） |

**动作输出（`_to_g1_action` 产出，`(1,T,D)`）：**

| 键 | 维度 | 来源 |
| --- | --- | --- |
| `left_arm` / `right_arm` | (1,T,7) | EE 位姿 → pinocchio 阻尼最小二乘 IK（warm-start） |
| `left_hand` / `right_hand` | (1,T,7) | Inspire-12 → Dex3-7（`_inspire12_to_dex7`） |
| `base_height_command` | (1,T,1) | 中性（固定基座桌面任务） |
| `navigate_command` | (1,T,3) | 中性 |

## 2. P0：前置环境、license-gated 资产与代码缺口

### 2.1 环境检查（不下载、不训练）

在 `starVLA` 仓库根目录执行，保存到 `Project_Analysis/evidence/egovla/00_preflight/`：

```bash
git rev-parse HEAD && git status --short
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h .
conda env list
```

激活 StarVLA 环境后执行依赖探针（**EgoVLA 特有**：部署 decode 需要 `pinocchio`、`smplx`、`chumpy`；`pytorch3d` 已是 starVLA 依赖）：

```bash
python - <<'PY'
import torch, cv2, numpy
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))
for m in ('pinocchio', 'smplx', 'chumpy', 'transformers'):
    try:
        mod = __import__(m); print(m, getattr(mod, '__version__', 'ok'))
    except ImportError as e:
        print(m, 'MISSING ->', e)
try:
    import zmq, msgpack; print('zmq/msgpack ok')
except ImportError as e:
    print('zmq/msgpack MISSING ->', e)
PY
```

缺失则补装（README 指令）：`pip install pinocchio smplx chumpy`（`zmq`/`msgpack` 通常已在，若缺失一并装）。通过标准：RTX 5090 可见、可用空间 ≥ 10 GB、`pinocchio`/`smplx`/`chumpy` 可 import、transformers 为 4.57（README 声明）。

### 2.2 license-gated 资产获取（P0 的核心 gate）

```bash
# 1) EgoVLA checkpoint（VILA 式，二选一：HF 或 EgoVLA_Release checkout）
export HF_ENDPOINT=https://hf-mirror.com
hf download rchal97/egovla_base_vlm --local-dir playground/Pretrained_models/ego_vla_checkpoint/ckpt-6720
#    或直接 git clone EgoVLA_Release 后，把 checkpoints/ego_vla_checkpoint/ckpt-6720 指过去

# 2) MANO（研究许可，浏览器注册下载）→ <EgoVLA_Release>/mano_v1_2/models/MANO_{LEFT,RIGHT}.pkl
# 3) hand_actuation_net.pth → eval_files/assets/（.gitignore 已排除 *.pth，不会误提交）
```

> **资产不可得时不伪造**：MANO 审批未通过 → decode 手部段 / 完整 server 阻塞，但 FK-IK 自测 + checkpoint→`(T,48)` 仍可先行；checkpoint 下载失败 → 转 HF mirror 重试。所有 license-gated 资产均**不进入** `starVLA-Demo` 仓库（`assets/.gitignore` 已排除 `*.pth`；MANO `.pkl` 与 checkpoint 落服务器本地，不入 git）。

### 2.3 当前代码缺口（P1/P2 前必须处理）

| # | 文件/位置 | 现状 | 处理 |
| --- | --- | --- | --- |
| G1 | `decode.py:30`、`egovla_g1_policy.py:52` | `EGOVLA_RELEASE` 默认硬编码 `/home/dhy/Projects/EgoVLA_Release`（Linux 路径） | P0 用 `--egovla_release` 或 `EGOVLA_RELEASE` 覆盖；不改仓库默认值 |
| G2 | `egovla_g1_policy.py:53` | 默认 `ckpt_dir = <egovla_release>/checkpoints/ego_vla_checkpoint/ckpt-6720` | P0 确认实际目录后 `--ckpt_dir` 覆盖 |
| G3 | `EgoVLA_ActionHeader.py` vs `decode.py` | **48 维排布文档不一致**（decoder `[left(3+6+15),right(3+6+15)]` vs decode `[trans|mano|rot6d]`），`EgoVLA.py._run` 未重排 | P1 用真实 checkpoint 实测对齐，必要时修 `decode.py` 切片（见 §3.1） |
| G4 | `train_files/` | **无 `data_registry/data_config.py`、无 `modality.json`、无训练脚本**；YAML `data_mix: your_egovla_data_mix`、`data_root_dir: /home/dhy/Projects/datasets` 均为占位 | P2 自写 DataConfig + modality.json + 训练脚本 + 注册 mixture（见 §4） |
| G5 | `starvla_egovla.yaml` | `freeze_modules: ''`（= 不冻结） | 12GB 卡上必须设 `freeze_modules` 或走 LoRA；5090 全参 FT 可保留 |
| G6 | 手部对应（校准点，非代码 bug） | `_inspire12_to_dex7` 假设 EgoVLA Inspire-12 顺序 ≈ Isaac URDF-12 顺序；H1→G1 相机/EE/手部映射 | README 已声明为「能力完备、抓取是否成功取决于模型」；P1 记录为已知校准不确定项，不在本计划内解决 |

> 与 RoboChallenge/G1/Franka 不同：EgoVLA 的**部署侧 import 与 server 契约基本完整**——`server_egovla_g1.py` 的 `_bootstrap.py` 正确把 repo root 挂上 `sys.path`，`ZmqGr00tPolicyServer` 契约（`get_action(observation, options)` / `reset` / `get_modality_config`）与 `EgoVLAG1Policy` 逐一对齐。真正的缺口集中在 **G3（48 维排布）** 与 **G4（训练侧空骨架）**。

## 3. P1：部署侧零训练模型闭环（主交付，无需训练）

### 3.1 子路 A：checkpoint → framework 加载 → `(T,48)`

不依赖 MANO，可先行。验证 `EgoVLA` 框架能用开源权重重建端到端前向（顶层 README 已声明 6/6 projector、98/98 traj_decoder 键严格匹配，此处在实际服务器环境复验）：

```bash
cd /absolute/path/to/starVLA
export PYTHONPATH="$PWD:${PYTHONPATH}"
python - <<'PY'
from omegaconf import OmegaConf
from starVLA.model.framework.base_framework import build_framework
import numpy as np
from PIL import Image
cfg = OmegaConf.load("examples/realRobots/EgoVLA/train_files/starvla_egovla.yaml")
cfg.framework.qwenvl.base_vlm = "playground/Pretrained_models/ego_vla_checkpoint/ckpt-6720"
fw = build_framework(cfg).to("cuda").eval()
obs = {"image": [Image.fromarray(np.random.randint(0,255,(384,384,3),np.uint8))],
       "lang": "place the red ball in the box"}
out = fw.predict_action([obs])["normalized_actions"]   # (1,30,48)
print("pred shape", out.shape, "finite", np.isfinite(out).all())
PY
```

通过标准：`build_framework` 实例化 `EgoVLA`，checkpoint 加载不报 key 错误，`normalized_actions` shape `(1,30,48)` 且有限。**同时在此步核对 48 维排布（G3）**：用一个已知小扰动（如只改第 0:6 维）观察 decode 后 EE 平移是否响应，或用上游 EgoVLA 的排布注释交叉验证，确认 `decode.py` 的 `[trans|mano|rot6d]` 切片是否与 checkpoint 输出对齐；不对齐则修 `decode.py`。

### 3.2 子路 B：G1 运动学 FK-IK 自测（纯 pinocchio，无 GPU/MANO）

`g1_kinematics.py` 已内置 `__main__` FK→IK 往返自测：

```bash
python examples/realRobots/EgoVLA/eval_files/g1_kinematics.py
# 期望输出：FK-IK round trip: 20/20 within 1mm
```

通过标准：20/20 收敛在 1mm 内，证明 `g1_29dof_with_hand.urdf` 能建 pinocchio 模型、`left/right_wrist_yaw_link` 帧存在、7-DoF 阻尼最小二乘 IK 有效。

### 3.3 子路 C：`decode.py` 合成 `(T,48)` 解码（依赖 MANO + hand net）

```bash
export EGOVLA_RELEASE=/path/to/EgoVLA_Release
python - <<'PY'
from examples.realRobots.EgoVLA.eval_files.g1_kinematics import G1Kinematics
from examples.realRobots.EgoVLA.eval_files.decode import EgoVLADecoder
import numpy as np
dec = EgoVLADecoder(G1Kinematics(), device="cuda")
pred = np.random.randn(8, 48).astype(np.float32)   # (T,48)
ego = dec.decode(pred)
for k, v in ego.items():
    print(k, v.shape, np.isfinite(v).all())        # ee_pose (T,7), inspire12 (T,12)
PY
```

通过标准：`left/right_ee_pose` `(T,7)`、`left/right_inspire12` `(T,12)` 有限无 NaN；MANO `.pkl` 能在 numpy≥1.24 下通过 `_install_numpy_aliases` shim 正常 unpickle。

### 3.4 子路 D：完整 server + 合成 GR00T 观测 → G1 关节目标动作（最终最低验收）

终端 A（策略服务，StarVLA 环境）：

```bash
cd /absolute/path/to/starVLA
python examples/realRobots/EgoVLA/eval_files/server_egovla_g1.py \
    --port 5555 --egovla_release /path/to/EgoVLA_Release \
    --ckpt_dir /path/to/ego_vla_checkpoint/ckpt-6720
```

终端 B（合成观测验证客户端，自写 `Project_Analysis/evidence/egovla/03_deploy_action_check/client_smoke.py`，镜像 `egovla_g1_policy._obs_to_example` 的输入构造，但不接真实 bridge）：

```python
# 构造 GR00T 风格观测：video.ego_view (1,T,H,W,3) 合成 + state{left_arm, right_arm,
#   left_wrist_eef_9d, right_wrist_eef_9d, ...} + language 固定 prompt
# 走 ZMQ REQ/REP（deployment.model_server.tools.zmq_policy_server 的 pack/unpack）
# 调 get_action，断言返回动作 dict：
#   left_arm/right_arm (1,T,7)、left_hand/right_hand (1,T,7)、
#   base_height_command (1,T,1)、navigate_command (1,T,3)，且数值有限、无 NaN
```

### 3.5 P1 通过标准

1. 框架加载 checkpoint 并输出 `(1,30,48)`（子路 A）；
2. FK-IK 20/20 收敛（子路 B）；
3. decode 输出 `(T,7)` EE + `(T,12)` Inspire 有限（子路 C）；
4. server `get_action` 对合成观测返回结构正确的 G1 关节目标动作 dict，臂 IK 收敛、手部 7 维 articulate（子路 D）；
5. 记录单轮推理延迟、48 维排布核对结论（G3 处理结果）。

这是「无真机复现」的最终最低验收：**证明 EgoVLA 策略服务能用开源权重对接 G1 观测契约、输出语义正确的 48 维相机帧动作，并解码为结构正确的 G1 关节目标动作。** 它不验证真实 G1 执行、不验证仿真/桥接器、不验证抓取成功率（README 声明 H1 训练 → G1 映射为「能力完备，抓取是否成功取决于模型」）。

## 4. P2：训练侧（二期，硬缺口）

> 仅当需要「在 G1 数据上微调 EgoVLA 以获得可靠抓取」时才进入 P2（eval README 明确指此为其路径）。P2 的工程量与 Franka 相当（缺 data registry + modality.json + 训练脚本 + 数据），且 EgoVLA 的 48 维 bimanual 相机帧 MANO 契约**无干净公开数据**，合成数据是默认路线。

### 4.1 训练数据三路线（按优先级）

| 路线 | 来源 | 结论 |
| --- | --- | --- |
| A：H1 训练数据 | EgoVLA 发布方私有 | 不可得，不伪造 |
| B：公开 bimanual 数据替代 | GR00T G1 bimanual 等 | 契约是关节目标，非 48 维相机帧 MANO，需重投影/重标注，工程量大且语义对不上，仅备选 |
| C：合成数据集（**默认**） | 自写生成器 | 最小可行，证明训练/部署链路工程闭环 |

### 4.2 契约归一 + registry/modality 补丁（G4 的落地）

1. **注册 mixture**：新建 `train_files/data_registry/data_config.py`，在 `DATASET_NAMED_MIXTURES` 注册 `egovla_*` mixture，并提供一个 `EgoVLA...DataConfig`（48 维 action、`proprio_3d/proprio_rot/proprio_hand_finger_tip` 三组 proprio、单路 384 相机）。`registry.py` 会 `glob("examples/**/train_files/data_registry")` 自动发现。
2. **自写 `train_files/modality.json`**：action 单列 `action`（48 维）或三段子键；proprio 三段（`proprio_3d`、`proprio_rot`、`proprio_hand_finger_tip`）；video 单路 `observation.images.ego_view`；annotation 扁平键 `human.action.task_description`。**这是 P2 最易错环节**（EgoVLA 示例 repo 内无现成 modality.json 可复制）。
3. **YAML 占位符替换**：`data_root_dir`、`data_mix`、`wandb_entity`。
4. **归一化对齐**：48 维里 MANO 15 维的 denorm 范围必须与 `decode.py` 的 `_MANO_MIN/_MANO_MAX` 一致；wrist trans/rot6d 的归一化尺度需与部署解码假设对齐（部署侧不经 `dataset_statistics.json` 反归一化，直接消费 decoder 输出）。

### 4.3 合成数据集生成器

`Project_Analysis/evidence/egovla/02_training/gen_synthetic_egovla.py`，产出 schema 一致的 LeRobot v2.1 数据集（目标路径 `playground/Datasets/EgoVLA/lerobot/egovla_synthetic/`）：

```text
egovla_synthetic/
├── data/chunk-000/episode_NNNNNN.parquet     # action[48] + proprio 三组 + timestamp 等
├── videos/chunk-000/observation.images.ego_view/episode_NNNNNN.mp4
└── meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json, embodiment.json}
```

- 每 episode ≥ 64 帧（> horizon=30，留采样余量）；action 走平滑轨迹，MANO 段 `_MANO_MIN.._MANO_MAX` 内采样；
- 视频用 **PyAV**（`av`）编码 384×384 H.264（服务器 OpenCV 缺 libx264，`cv2.VideoWriter` 会静默写空文件）；
- `meta/embodiment.json` 的 `embodiment_tag` 写 `new_embodiment`；`tasks.jsonl` 至少一条 prompt；
- 合成数据动作**无任务语义**，仅验证链路。

### 4.4 dataloader/forward gate + freeze/LoRA smoke

```bash
conda activate <starvla env>
export PYTHONPATH="$PWD:${PYTHONPATH}"
python starVLA/dataloader/lerobot_datasets.py \
  --config_yaml examples/realRobots/EgoVLA/train_files/starvla_egovla.yaml

python starVLA/model/framework/VLM4A/EgoVLA.py \
  --config_yaml examples/realRobots/EgoVLA/train_files/starvla_egovla.yaml
```

通过标准：dataloader 采样单路 384 图像 + 48 维 action + 三组 proprio，不报 modality/registry/video 错误；EgoVLA forward 不报 key/shape 错误（`EgoVLA.py` 是否有 `__main__` gate 需 P2 确认，若无则用 `build_framework` 自检）。

smoke 训练（单卡）：

```bash
export CUDA_VISIBLE_DEVICES=0
export BATCH=1
export MAX_STEPS=1000
export WANDB_MODE=disabled
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/realRobots/EgoVLA/train_files/starvla_egovla.yaml \
  --datasets.vla_data.data_root_dir playground/Datasets/EgoVLA/lerobot \
  --datasets.vla_data.data_mix egovla_synthetic \
  --framework.qwenvl.base_vlm playground/Pretrained_models/ego_vla_checkpoint/ckpt-6720 \
  --trainer.max_train_steps ${MAX_STEPS} \
  --run_root_dir ./results/Checkpoints --run_id egovla_smoke \
  2>&1 | tee tmp/logs/egovla_train_1000step.log
```

> 显存预算：SigLIP-384 + Qwen2-1.5B（~2B）全参 FT 16–20GB，仅 5090 可行；4070 Ti 12GB 必须 `--framework.freeze_modules`（冻结 VLM）或 LoRA。5000 step 全参 FT 约 2.5–4.5h（5090）。`gradient_checkpointing: true` + `gradient_accumulation_steps: 4` 已在 YAML，batch=1 优先。

预期输出（`run_root_dir=./results/Checkpoints`）：

```text
results/Checkpoints/egovla_smoke/
  config.yaml  config.full.yaml  dataset_statistics.json
  checkpoints/steps_1000_pytorch_model.pt
```

通过标准：loss 连续有限、无 OOM、checkpoint + `dataset_statistics.json` 同目录。

## 5. P3：结果归档

```text
Project_Analysis/evidence/egovla/
  00_preflight/              # GPU、磁盘、commit、依赖版本（pinocchio/smplx/chumpy）
  01_checkpoint/             # checkpoint 来源（rchal97/*）、HF 文件清单、sha256
  02_kinematics/             # FK-IK 自测日志
  03_deploy_action_check/    # (T,48) 输出、decode 证据、client_smoke.py、server 日志、G3 排布核对结论
  04_training/               # （仅 P2）registry/modality 补丁、合成生成器、loss/显存/step-s、ckpt hash
  README.md                  # 实测日期、命令、成功/失败与未完成范围
```

license-gated 资产（MANO `.pkl`、`hand_actuation_net.pth`、checkpoint 权重）**不归档入仓库**（`assets/.gitignore` 已排除 `*.pth`；MANO 有研究许可，不传播）。记录 48 维排布核对结论与已知校准不确定项（Inspire→Dex3、H1→G1）。P2 的 code 补丁（registry/modality）→ fork `TangYishanE/starVLA` 分支 `repro/egovla`；若产出自训练 checkpoint → HF 私有 repo（必须带 `config.yaml` + `dataset_statistics.json`，见 `starvla-demo-archive` 约定）。

## 6. 失败分流与停止条件

| 失败点 | 首先检查 | 停止/继续规则 |
| --- | --- | --- |
| MANO 审批未通过 | mano.is.tue.mpg.de 注册状态、研究许可 | 不伪造；FK-IK + `(T,48)` 子路先行，decode/server 阻塞至审批通过 |
| checkpoint 下载失败 | HF mirror（`HF_ENDPOINT=https://hf-mirror.com`）、`rchal97/*` 仓库名 | 换镜像/仓库重试；权重不可得则 P1 主交付无法完成，如实报告 |
| `build_framework` 报框架未注册 | `_auto_import_framework_modules` 是否覆盖 `EgoVLA.py`、`FRAMEWORK_REGISTRY` 是否含 `EgoVLA` | 未注册禁止训练/部署，先修注册 |
| checkpoint 加载 key 错误 | `base_vlm` 目录是否含 `{llm,vision_tower,mm_projector,traj_decoder}`、traj_decoder 键是否与 vendored `EgoVLATrajDecoder` 对齐（98/98） | 核对 `EgoVLA_ActionHeader` 的 `decoder.*` 嵌套 |
| **48 维排布对不上（G3）** | `decode.py` 的 `[trans|mano|rot6d]` vs action header 的 `[left(3+6+15),right(3+6+15)]` | P1 实测后修 `decode.py` 切片（或确认框架重排），未对齐不得宣称「语义正确」 |
| decode MANO 加载失败 | `EGOVLA_RELEASE` 路径、`mano_v1_2/models/*.pkl`、numpy 版本（`_install_numpy_aliases` shim） | 对照 §1.1 资产清单逐项修 |
| server `get_action` 返回 `{"error":...}` | server 日志、合成观测键名（`video.ego_view`/`state.*`/`language.*`）、ZMQ pack/unpack | 抓包确认请求结构，对照 §1.3 契约 |
| IK 不收敛 | `ik_arm` 的 seed、目标 EE 位姿是否可达、damping/iters | 调 iters=100、max_step；仍不收敛则检查 EE 帧名与 URDF |
| 训练 dataloader 失败（P2） | modality.json 子键（尤其 proprio 三段）、parquet 字段、video symlink、`<your_dataset>` 占位 | 对照 §4.2 契约逐项修 |
| 训练 OOM（P2） | batch、freeze_modules、DeepSpeed | 固定 2B，batch=1；12GB 卡强制 freeze/LoRA，不改用 4B/8B |

## 7. 最终交付标准

完成后应在干净 shell 中按序重新执行并通过：

1. 依赖探针（`pinocchio`/`smplx`/`chumpy`/`zmq`/`msgpack`）与 license-gated 资产齐全；
2. checkpoint → framework 加载 → `(1,30,48)` 有限输出（含 48 维排布核对结论）；
3. FK-IK 自测 20/20 收敛；
4. `decode.py` 合成 `(T,48)` → `(T,7)` EE + `(T,12)` Inspire 有限；
5. `server_egovla_g1.py` + 合成 GR00T 观测 → G1 关节目标动作 dict 结构正确、臂 IK 收敛、手部 articulate、无 NaN；
6. 汇总：checkpoint 来源、峰值显存、单轮推理延迟、G3 排布结论、未完成范围（P2 训练是否开展）。

向老师的建议表述：**「在无实体机器人、无仿真与桥接器的条件下，用 EgoVLA 已发布的开源权重完成 starVLA EgoVLA 示例的模型侧闭环：加载 EgoVLA 框架输出 48 维相机帧动作，经 decode 解码为 G1 骨盆系腕部位姿与 Inspire 手部，pinocchio IK 解算为 7-DoF 臂关节目标，并通过 GR00T ZMQ 协议对合成观测返回结构正确的 G1 关节目标动作。」** 需同时注明：本计划未涉及实体 G1、Isaac-Lab 仿真与 GR00T-WBC-Bridge，因此不报告真实机器人闭环或任务成功率；EgoVLA 为 H1 训练，H1→G1 映射为「能力完备、抓取是否成功取决于模型」，手部 Inspire→Dex3 对应为已知校准点；训练侧（P2）为空骨架、无公开 48 维 bimanual 相机帧数据，是否开展微调需另行决策。

## 附：与其他 realRobots 计划的关键差异

| 维度 | RoboChallenge | UnitreeG1_WholeBody | Franka | Realman | **EgoVLA（本计划）** |
| --- | --- | --- | --- | --- | --- |
| 开源权重 | ❌ | ❌ | ❌ | ❌ | ✅ `rchal97/*`（**唯一有**） |
| 首轮是否训练 | 是 | 是 | 是 | 是 | **否（零训练部署）** |
| 首要外部 gate | 公开数据 `--only` | 数据不随发/非公开 | 数据+转换脚本皆无 | 公开数据可替代 | **MANO 模型审批（license）** |
| 动作契约 | 8D ee_pose+gripper | 78D SONIC latent（64+7+7） | 7D delta-EE | 8D 关节 delta+gripper | **48D 相机帧 = 2×(3 trans+6 rot6d+15 MANO)** |
| 模型/显存 | 0.8B ~5GB | 0.8B ~5–7GB | 0.8B+DiT ~8–12GB | ResNet-18 ~2–3GB | **SigLIP+Qwen2-1.5B（~2B）FT 16–20GB / freeze·LoRA** |
| 部署协议 | 上游 HTTP mock | 自写合成观测+切分 | WebSocket `server_policy.py` | 无部署层（in-process） | **GR00T ZMQ `:5555`（`ZmqGr00tPolicyServer`）** |
| 免真机验证 | mock server | 动作切分 64/7/7 | 反归一化 (16,7) | in-process 重载 | **三层阶梯：FK-IK→decode→完整 server get_action** |
| 代码缺口 | 2 import + 2 cd | 缺 `local_self_test.py` | 6 处 | registry 占位 + mixture 默认值 | **G3（48 维排布）+ G4（训练侧空骨架）** |
| 完成语义 | 协议闭环 | 动作切分 | 反归一化 | 双基线重载 | **零训练模型闭环 + 48D 解码为 G1 关节目标** |
