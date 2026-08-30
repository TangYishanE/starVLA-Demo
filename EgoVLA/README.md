# EgoVLA — 无真机复现

**目标：** 复现 `starVLA/examples/realRobots/EgoVLA` 的「框架推理 → 48D 相机帧动作 → 解码 → G1 关节目标动作」模型侧闭环，无实体 G1 / 无 Isaac-Lab 仿真 / 无 GR00T-WBC-Bridge。

## 状态

| 阶段 | 状态 | 说明 |
| --- | --- | --- |
| P0 环境 + 资产 | ✅ | 依赖补装（pinocchio 4.1.0 / smplx 0.1.28 / chumpy 0.70 / pyzmq）；EgoVLA checkpoint 3.9GB 下载并拍平；`RchalYang/EgoVLA_Release@09645b9` 克隆（含 hand nets） |
| P1a checkpoint | ✅ | `rchal97/egovla` → `ego_vla_checkpoint/ckpt-6720`（VILA 式 `{llm,vision_tower,mm_projector,traj_decoder}`），sha256 见 evidence |
| P1b FK-IK 自测 | ✅ | `g1_kinematics.py`：**20/20 收敛在 1mm 内** |
| P1c 框架 → (1,30,48) | ✅ | `build_framework(EgoVLA)` 加载开源权重，输出 `(1,30,48)` 有限，推理延迟 ~0.8–1.9s |
| G3 48D 排布审计 | ✅ | **无 bug**：decode.py 切片与原始 `ik_eval_single_step` 逐行一致；vendored decoder 与原始构造一致 + 98/98 权重严格匹配（`evidence/03_deploy_action_check/G3_layout_audit.md`） |
| P1d 部分链路（无 MANO） | ✅ | 48D → rot6d/trans 切片 → 骨盆系 EE 位姿（有限）→ `ik_arm` 执行；EE 位置在桌面工作空间量级 |
| P1d/e 手部 decode + 完整 server | ⛔ **阻塞** | **MANO 手模型（研究许可）缺失**：需在 [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) 注册下载 `MANO_LEFT.pkl`/`MANO_RIGHT.pkl`，放到服务器 `EgoVLA_Release/mano_v1_2/models/` 后，运行 `scripts/client_smoke.py` 完成验收 |
| P2 训练侧（二期） | ⏸ 未执行 | 训练侧为空骨架（占位 YAML，无 data_registry/modality.json/脚本/公开数据），另立一轮 |

## 关键路径与数字

- **远程服务器：** `starvla-hpc`（RTX 5090 32GB），仓库 `48e5881`，环境 `starvla-cu128`（py3.10 / torch 2.7.1+cu128 / numpy 2.2.6）
- **EgoVLA checkpoint：** `playground/Pretrained_models/ego_vla_checkpoint/ckpt-6720`（3.9GB，含 `dataset_statistics` 无关，部署不经反归一化）
- **EgoVLA_Release：** `/225010261/StarVLA/EgoVLA_Release`（含 `hand_actuation_net.pth`、`hand_mano_retarget_net.pth`、`human_plan/` MANO FK）
- **PyPI 坑：** 真 pinocchio 包名是 `pin`（`pip install pinocchio` 装的是无关 ORM 库）；chumpy 0.70 需 `--no-build-isolation` 且要 numpy 别名 shim（`decode.py` 已内置）

## 文件

- `plans/EgoVLA-no-robot-reproduction-plan-20260831.md` — 执行计划（样板）
- `scripts/client_smoke.py` — GR00T ZMQ 合成观测验证客户端（待 MANO 就绪运行）
- `evidence/` — preflight、FK-IK 日志、框架 (1,30,48) 日志、EE/IK 部分链路日志、G3 审计

## 链接

- fork 分支：`TangYishanE/starVLA`（本轮无代码改动，G3 审计结论为无需修改）
- HF checkpoint 备份：`TangYishan/egovla-ckpt-6720`（私有，EgoVLA 开源权重镜像 + 模型卡）
- 上游权重：`rchal97/egovla`；发布仓库：`RchalYang/EgoVLA_Release`
