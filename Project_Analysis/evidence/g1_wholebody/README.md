# UnitreeG1_WholeBody 无真机复现 — 实测结果归档

**实测日期：** 2026-08-31（服务器 UTC 2026-08-30 20:32 → 21:15，总耗时 43 分钟）
**服务器：** `c03cm3nlal8v-0`（10.27.130.23:30647），单张 RTX 5090 32GB
**仓库：** `/225010261/StarVLA/src/starVLA` @ `48e5881`（main）
**环境：** `envs/starvla-cu128`（Python 3.10.20 / torch 2.7.1+cu128 / DeepSpeed 0.16.9 / CUDA 12.5）

---

## 1. 数据（P1）— 路线：公开真实数据（非合成）

| 项 | 值 |
|---|---|
| 数据集 | `cloudwalk-research/gr00t-g1-grab-bottle-right-hand-v11`（LeRobot v2.1） |
| 规模 | 355 episodes / 87148 frames / fps 50 / 单路 ego_view（480×640） / 472MB |
| 接线 | `lerobot/test_sonic -> gr00t-g1-grab-bottle-right-hand-v11`（symlink，registry 的 `unitree_g1_test_sonic` mixture 直接生效，零配置改动） |
| Schema 核对 | **state 72D** = observation.state(43) + eef_state(14) + root_orientation(4) + projected_gravity(3) + cpp_rotation_offset(4) + init_base_quat(4)；**action 78D** = motion_token(64) + left/right_hand_joints(7+7) — 与 `data_config.py`/`modality.json` 逐字段一致 |

## 2. 训练（P2）— QwenOFT / Qwen3.5-0.8B / 1000-step 冒烟

| 项 | 值 |
|---|---|
| 启动 | `run_starvla_qwenoft_g1_sonic_train.sh`（BATCH=1, MAX_STEPS=1000, SAVE_EVERY=200, WANDB_MODE=disabled） |
| 耗时 | **40:20**（1000 步，~2.42 s/it；`data_times≈0`，dataloader worker 预取） |
| 峰值显存 | **27.3 / 32.6 GB**（DeepSpeed ZeRO-2 单卡，bf16 + fp32 优化器状态） |
| 最终 loss（step 1000） | `action_dit_loss=0.598`，`mse_score=0.0093`（学习率已衰减至 min_lr；1000 步为工程冒烟，未收敛属预期） |
| 产物 | `checkpoints/steps_{200,400,600,800,1000}_pytorch_model.pt` + `final_model/` + `dataset_statistics.json` + `config.yaml` + `summary.jsonl`（13GB 含中间档） |
| checkpoint sha256 | `2612cf7fbcbabafba9efdb1a29ac7e6c5c7e29fd6c1afdd90cd402b88b477d56`（steps_1000，2.24GB） |
| 归一化 | `dataset_statistics.json` 顶层键 = `new_embodiment`（= `EmbodimentTag.NEW_EMBODIMENT`，q99） |

**前置 gate：** dataloader gate ✅（355 轨迹/87148 帧，72D/78D 键重排正确）；QwenOFT forward gate ✅（推理输出 `(1,8,78)`，loss 有限）。

## 3. 部署（P3）— 策略服务 + 动作切分（无真机）

| 项 | 值 |
|---|---|
| 服务 | `deployment/model_server/server_policy.py`（port 5694, `--use_bf16`），模型加载至端口就绪 ~63s |
| Server metadata | `action_chunk_size=8`、`default_unnorm_key=new_embodiment`、`training_obs_image_size=[224,224]`、action_keys=`[motion_token, left_hand_joints, right_hand_joints]`、state_keys=15 个（72D） |
| 客户端 | 自写 `local_self_test.py`（`ModelClient` + 合成 224×224 图像 + 72D state + prompt） |
| **输出分片** | `motion_token=(8,64)` + `left_hand=(8,7)` + `right_hand=(8,7)` — 3 次运行一致，数值有限无 NaN |
| **平均推理延迟** | **1947.6 ms** |
| 断言 | `OK ✅ policy returns 8 steps -> 64+7+7 (78D), split correct` |

## 4. 完成范围与未做内容

**✅ 已完成：** 数据契约确认（72D/78D）→ dataloader gate → forward gate → 1000-step checkpoint → 策略服务启动 → `ModelClient` 对合成观测输出并正确切分 64/7/7 动作块。

**❌ 未涉及（超出无真机边界）：** 实体 G1 执行、SONIC/WBC 解码、PICO 遥操作采集、真实机器人闭环、任务成功率。

**⚠️ 说明：** 数据来自公开 grab-bottle 数据集（非官方私有 `test_sonic`），动作分布真实但任务语义仅限该数据集；1000-step 为冒烟训练（未收敛），本计划验证的是「数据—训练—策略服务—动作契约」工程链路，不承诺策略性能。

## 5. 证据文件索引

- `00_preflight/` — （环境信息见本 README；GPU/依赖已在会话内核实）
- `01_data/` — 数据集 schema 核对结果（见 §1）
- `02_training/` — `config.yaml`、`dataset_statistics.json`、`summary.jsonl`、checkpoint sha256
- `03_deploy_action_split/` — `local_self_test.py`、`self_test.log`、`summary.log`、`server.log`、`run_g1_full_chain.sh`
- 远程完整日志：`/225010261/StarVLA/src/starVLA/results/g1_wholebody_run/`
