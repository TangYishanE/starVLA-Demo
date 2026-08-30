# RoboChallenge Table30v2 `shred_paper` 复现结果

**日期**：2026-08-31（服务器时间 UTC 2026-08-30）
**机器人**：UR5（单臂）
**框架**：QwenOFT（Qwen3.5-0.8B）
**GPU**：RTX 5090 32GB（远程单卡）

## P1 数据

- raw：8.9GB（`RoboChallenge/Table30v2` 仅 `shred_paper`）
- LeRobot v2.1：1016 episodes / 971436 frames，fps=30
- state `[7]`（6 joint + gripper）、action `[8]`（7 ee_pose + gripper）
- 双相机 `cam_global` + `cam_arm`（480×640×3 h264）
- 数据契约见 `evidence/robochallenge/data/info.json`、`modality.json`

## P2 训练（100-step 冒烟）

- 全参数微调（`freeze_modules: ''`），batch=1，bf16 + DeepSpeed Zero-2
- 实际训练 19:36:53 → 19:40:53（约 4 分钟），~2.4s/step
- GPU 峰值 22074 MiB / 32607（68%）
- 配置见 `evidence/robochallenge/training/config.yaml`

### Loss 轨迹（action_dit_loss，每 5 步）

```
step  5: 0.557   step 30: 0.466   step 55: 0.397   step 80: 0.439
step 10: 0.630   step 35: 0.450   step 60: 0.434   step 85: 0.331
step 15: 0.952   step 40: 0.941   step 65: 0.395   step 90: 0.438
step 20: 0.652   step 45: 0.563   step 70: 0.407   step 95: 0.309
step 25: 0.435   step 50: 0.524   step 75: 0.559   step 100: 0.315
```

- 全部有限、无 NaN/Inf；整体 0.56→0.31 下降（尖峰为 batch=1 正常噪声）
- 产物：`steps_100_pytorch_model.pt`（2.24GB）+ `final_model/pytorch_model.pt`（2.24GB）+ `dataset_statistics.json`

## P3 checkpoint self-test

- 3 次推理均输出 `(8,8)`，无 NaN/维度错误
- 输出 `[-0.546, 0.026, 0.433, 0.724, 0.682, -0.003, -0.005, 0.020]`（接近动作均值，符合 100-step 模型预期，证明 un-normalization 链路正确）
- latency：1742.8 / 1999.2 / 2106.5 ms，avg 1949.5 ms

## P4 上游 mock 协议闭环

- 上游 `RoboChallenge/RoboChallengeInference`（cvpr 分支），回放数据 `20260413/ur5/arrange_fruits` 仓库自带
- **28 次完整迭代 / 62s**，clock jitter 0.0004s
- 每轮：`GET /state.pkl`（leftjoint, 7D）→ 推理 → `POST /action`（leftpos, 8D, duration=0.05）
- 推理 ~1.65–2.82s，网络 ~0.6–1.0ms，`pending_actions` 始终为 0

### 修复的上游 bug（见 `evidence/robochallenge/mock/rc_mock_patch.diff`）

1. `interface_client.py`：`mock_url + "/"` 产生双斜杠 URL → 全部 404 + clock-sync 死循环；去掉尾斜杠
2. `interface_client.py`：`end_motion()` 调 `/stop_motion`，server 路由是 `/end_motion`；改名

## 边界说明

- 跨任务协议验证：checkpoint 训练 `shred_paper`，mock 回放用 `arrange_fruits`（官方唯一 UR5 记录）
- 动作输出不具任务语义；未做线上 submission、未使用实体 UR5
