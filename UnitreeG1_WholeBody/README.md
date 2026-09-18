# UnitreeG1 WholeBody

StarVLA `examples/realRobots/UnitreeG1_WholeBody` 示例的**无实体机器人复现**：在无实体 G1 机器人条件下完成「数据契约 → 训练 → 策略服务 → StarVLA→G1 动作适配」链路。

**状态：✅ 完成（2026-08-31）**

## 数据源

官方 `test_sonic` 不公开，改用公开数据集 `cloudwalk-research/gr00t-g1-grab-bottle-right-hand-v11`（LeRobot v2.1）：

- 355 episodes / 87148 frames / fps 50 / 单路 ego_view（480×640）/ 472MB
- 通过 symlink `test_sonic → gr00t-g1-grab-bottle-right-hand-v11` 接入 registry 的 `unitree_g1_test_sonic` mixture，**零配置改动**
- **state 72D** = observation.state(43) + eef_state(14) + root_orientation(4) + projected_gravity(3) + cpp_rotation_offset(4) + init_base_quat(4)
- **action 78D** = motion_token(64) + left_hand_joints(7) + right_hand_joints(7)，q99 归一化

## 复现结果

| 项 | 结果 |
| --- | --- |
| dataloader gate | ✅ 355 轨迹 / 87148 帧，72D/78D 键重排正确 |
| forward gate | ✅ QwenOFT（Qwen3.5-0.8B）输出 `(1,8,78)`，loss 有限 |
| 1000-step 训练 | ✅ 40:20 / ~2.42 s/it / 峰值 27.3GB / loss 0.598 |
| self-test | ✅ `motion_token(8,64) + left_hand(8,7) + right_hand(8,7)`，avg 1947.6ms |
| dry-run step 2 | ✅ 真实 episode 0 回放（30 帧），motion_token MSE 0.0126 |
| dry-run step 3 | ✅ mock 控制器 64/7/7 切分（5 次迭代，形状/有限性全通过） |

## 产物位置

- 代码补丁：本轮无仓库改动（symlink 接线，`local_self_test.py`/`mock_g1_controller.py`/`replay_lerobot_episode.py` 为自写交付物）
- checkpoint：`steps_1000_pytorch_model.pt`（2.24GB，sha256 `2612cf7fbcbabafba9efdb1a29ac7e6c5c7e29fd6c1afdd90cd402b88b477d56`）
- 实测证据：`Project_Analysis/evidence/g1_wholebody/`（README + 02_training + 03_deploy_action_split，含 dry-run step 2/3 日志）

## 目录

- `plans/` — 无实体机器人复现计划
- `scripts/run_g1_full_chain.sh` — 无人值守训练→服务→self-test 链
- 实测证据见 `Project_Analysis/evidence/g1_wholebody/`

## 边界

- 无实体 G1、SONIC/WBC 解码、PICO 遥操作采集，不报告真实机器人闭环或任务成功率
- 数据来自公开 grab-bottle 数据集（非官方私有 `test_sonic`），动作分布真实但任务语义仅限该数据集
- 1000-step 为冒烟训练（未收敛），验证的是「数据—训练—策略服务—动作契约」工程链路，不承诺策略性能
