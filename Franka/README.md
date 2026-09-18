# Franka（Real Robot）无真机复现 — 完成归档

**日期：** 2026-08-31（服务器时间 2026-08-30 UTC）
**目标项目：** `examples/realRobots/Franka`（StarVLA）
**运行环境：** 远程单张 RTX 5090 32GB（共享）、无实体 Franka
**完成定义：** 数据契约确认 → QwenOFT 训练与 checkpoint 重载 → 策略服务 → 合成观测输出语义正确的 7D delta-EE 动作块

---

## 1. 复现成功要求核对（对照计划 §8）

| # | 验收项 | 状态 | 证据 |
| --- | --- | --- | --- |
| 1 | 数据 metadata/shape（6D state / 7D action / 双相机 / min_max+binary） | ✅ | `evidence/01_data/info.json`、`modality.json`；180 集、10,351 帧 |
| 2 | mixture 注册命中（`franka_eef_joints`） | ✅ | dataloader gate 输出 `Using mixture 'franka_eef_joints'` |
| 3 | dataloader gate | ✅ | `EmbodimentTag.NEW_EMBODIMENT`；action keys = `delta_eef_position/delta_eef_rotation/gripper_close` |
| 4 | QwenOFT forward gate | ✅ | `Predicted Action shape: (1, 16, 7)`（exit=0） |
| 5 | 1000-step checkpoint + `dataset_statistics.json` | ✅ | `steps_1000_pytorch_model.pt`（2.2GB，sha256 `7552b04d...`）+ stats 同 run 目录 |
| 6 | 策略服务 + 合成观测 → `(16,7)` 动作 | ✅ | 服务端 `actions (1,16,7)`，全有限，gripper 二值化 [0,1]，延迟 3.03s |
| 7 | 汇总指标 | ✅ | 见 §3 |

**结论：全部 7 项通过，满足计划的「无真机复现」完成定义。**

---

## 2. 数据与契约

- **数据来源：** RoboMimic "Lift"（rt_benchmark `lift_real/ph`，**真实 Franka Panda** 采集，200 demo 中的 train split 180 集），HDF5 1.9GB 直链下载 → 按 `franka2lerobot/README.md` 规范自写转换器（`scripts/convert_robomimic_to_lerobot.py`）转 LeRobot v2.1（PyAV H.264，224×224）。
- **契约（`SingleFrankaRobotiqDeltaEefDataConfig`）：**
  - action 7D = `delta_eef_position(3) + delta_eef_rotation(3) + gripper_close(1)`（RoboMimic OSC_POSE 原生即 7D delta-EE，零转换）
  - state 6D = `eef_position(3) + eef_rotation(3, axis-angle)`（`include_state: false`，不进模型）
  - 双相机 `base_view` + `ego_view`；归一化 min_max + binary(gripper)；embodiment = `new_embodiment`；horizon = 16
- **数据规模：** 180 集 / 10,351 帧 / 33MB（H.264 压缩后）。

## 3. 关键指标（实测）

| 指标 | 值 |
| --- | --- |
| 训练 | QwenOFT + Qwen3.5-0.8B + MLP 头，batch=1，1000 步 |
| 训练时长 / 速度 | 38 min 22 s / 2.30 s/it |
| 峰值显存 | ~22.3 GB（DeepSpeed zero2 单卡） |
| 推理延迟 | 3.03 s / 16 步动作块 |
| 输出 | `(16, 7)` 有限浮点，gripper ∈ {0, 1}（0=关 1=开） |
| checkpoint | `steps_1000_pytorch_model.pt`，sha256 `7552b04d156381fc6e6b5dc6d3ec33d63bf3ede44881ea52b9f66289ddb2480f` |

## 4. 交付物

- **Hugging Face：** [`TangYishan/starvla-qwenoft-franka-robomimic-lift`](https://huggingface.co/TangYishan/starvla-qwenoft-franka-robomimic-lift)（public，5 个 checkpoint + config + dataset_statistics.json + summary.jsonl + final_model + README）
- **代码缺口修复（6 处）：**
  - G1 mixture 注册（`franka_eef_joints` → `franka_pick_and_place_lerobot` / `custom_robot_config`）
  - G3 client import（`websocketclient` → `deployment.model_server.tools.websocket_policy_client`）
  - G6 框架统一（`QwenGR00T/2.5VL-3B/flash_attn2/DiT-B` → `QwenOFT/Qwen3.5-0.8B/sdpa/MLP`）
  - G8 embodiment key（`franka` → `new_embodiment`，实际服务端已自行反归一化，client 只验证）
  - G4/G7 由启动方式规避（`accelerate launch --num_processes 1` + 全路径 PATH）

## 5. 边界与未完成范围

- **无实体 Franka、无真实相机**：只验证到「策略服务对合成观测输出语义正确的 7D delta-EE 动作」，未做真实机器人闭环、运动学执行或任务成功率。
- **1000 步为工程冒烟**（非性能训练）：动作无任务语义，不可用于实际部署。
- 双 14D 臂（`dual_franka_eef_joints`）未做（需新写 DualFranka DataConfig）。
- 训练期间 5090 被其他任务占用导致两次 OOM，通过等待 + 终止泄漏进程解决。

## 6. 目录结构

```text
Franka/
├── README.md                  # 本文件
├── plans/                     # 执行计划文档
├── evidence/
│   ├── 00_preflight/          # GPU、commit、环境版本
│   ├── 01_data/               # info/modality/schema、转换日志、占用
│   ├── 02_training/           # config、stats、summary、ckpt 哈希、日志尾
│   └── 03_deploy_action_check/# forward gate、P3 验证、服务端、链日志
└── scripts/                   # 转换器、训练链、P3 验证、HF 上传脚本
```
