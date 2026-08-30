# Realman RM-75 无真机复现

StarVLA `examples/realRobots/Realman` 示例的**无实体机器人复现**：VM4A 的 ACT 与 DiffusionPolicy 两个基线在公开数据集上完成「数据契约 → 训练 → checkpoint 重载 → 合成观测推理」闭环。

**状态：✅ 完成（2026-08-31）**

## 数据源

`nvidia/PhysicalAI-Robotics-Manipulation-SingleArm` → `panda-open-drawer` 子集（CC-BY-4.0 可商用）

- Franka Panda（7-DoF）IsaacSim 仿真数据，**LeRobot v2.1**，1273 ep / 154256 帧 / 30fps
- action **8D = 7 关节 delta + 1 gripper**（与 Realman 契约逐维一致）
- 双相机 `world_camera` + `hand_camera`（512×512），另带 depth（忽略）
- 落地路径：`playground/Datasets/Realman/lerobot/panda-open-drawer/`（1.3 GB）

## 复现结果

| 项 | 结果 |
| --- | --- |
| dataloader gate | ✅ `state=['joints','gripper']` / `action=['delta_joints','gripper_close']`，154256 帧可读 |
| forward gate | ✅ ACT + DiffusionPolicy 均实例化 |
| ACT 100-step | ✅ `realman_act_smoke`（5.74 it/s） |
| DiffusionPolicy 100-step | ✅ `realman_dp_smoke` |
| self-test | ✅ ACT `(50,8)` finite、DP `(8,8)` finite（反归一化 `new_embodiment`） |
| HF 上传 | ✅ 两个私有 repo |

## Checkpoint（HF 私有 repo）

- ACT → `TangYishan/starvla-act-realman-panda-open-drawer`
- DiffusionPolicy → `TangYishan/starvla-diffusionpolicy-realman-panda-open-drawer`

（均含 `config.yaml` + `dataset_statistics.json` + `checkpoints/steps_100_pytorch_model.pt`）

## 关键适配（code patches → fork 分支 `repro/realman`）

1. `data_config.py`：`<your_dataset>` → `panda-open-drawer`
2. 两个 YAML：`action_mode: delta` → `abs`（NVIDIA 数据本身已是 delta，避免双重差分）；`data_root_dir` 指向真实路径
3. `train_realman_dp.sh`：默认 `data_mix` → `realman_example_dp`
4. 自写 `meta/modality.json`（state 切片 `joints[11:18]`/`gripper[7:8]`，相机 `cam0_rgb→world_camera`）

## 环境要点

- `lerobot` 必须 pin **0.3.3**（0.4.x 移除 `lerobot.constants`，ACT.py 无法 import）
- 服务器需装 `draccus pyserial deepdiff einops` 等 lerobot 运行时依赖

## 目录

- `plans/` — 复现执行计划
- `scripts/` — `realman_adapt.py` / `realman_repro_chain.sh` / `realman_self_test.py` / `realman_upload_hf.py`
- `evidence/` — 数据契约（info/modality）、训练 config、self-test 日志、summary、链日志

## 边界

IsaacSim 仿真数据、无实体 RM-75。本复现验证到「checkpoint 重载 + 合成观测输出语义正确的 8D 动作」，不报告真实机器人闭环或任务成功率。
