# RoboChallenge Table30v2：当前服务器执行条件评估

评估对象是此前已记录的远程服务器，而不是当前 Windows 本地工作区。现有硬件证据来自 2026-07-30 的测试汇总；因此“通过”表示该日期已实测，“待确认”表示切换到新 `main@48e5881`、开始下载前必须在远程服务器重新运行命令确认。

## 结论

**服务器适合启动 RoboChallenge Table30v2 的单任务 `shred_paper` 复现，条件是先通过磁盘、CUDA toolkit、模型下载和当前 main 路径补丁四个 gate。** RTX 5090 32 GiB 对默认 Qwen3.5-0.8B + QwenOFT、batch=1、100 step 的训练和本地推理具有充分余量；GPU 不是首要风险。

当前最可能阻塞的是：

1. **磁盘：** 服务器总配额只有 100 GB，不能保证既能保留现有工作、下载 raw 视频、临时解包，又保留 LeRobot 视频链接目标和 checkpoint。单任务开始前必须有至少 80 GB 可用空间，推荐 100 GB；否则应使用更大的独立数据盘。
2. **当前 main 的迁移回归：** RoboChallenge 的两个评测 Python import 仍指向旧 `examples.simBenchmarks`，两个 launcher 也少返回一级目录；不补丁，self-test/mock 将在加载 checkpoint 前失败。
3. **上游 mock 回放记录：** `RoboChallengeInference` 的 `20260413/ur5/...` 记录资源是否可取得尚未确认。它是 mock 协议闭环的外部资产，不应假定已由 Table30v2 数据集提供。

## 已有实测条件

| 维度 | 已有证据 | 对 RoboChallenge 的判断 |
| --- | --- | --- |
| GPU | 单张 RTX 5090，32 GiB VRAM | **通过。** 默认 0.8B walkthrough 应以 `BATCH=1` 首跑；比此前已测的 4B OFT 推理显存负载轻得多 |
| Python/深度学习栈 | Python 3.10、PyTorch 2.7.1+cu128、BF16、SDPA 已通过 | **基本通过。** 仍须在新 `starVLA_dev` 上复核 `accelerate`、DeepSpeed、OpenCV、PyArrow |
| 已测大模型资源边界 | 4B 全参 AdamW 在约 32,116 MiB OOM；LoRA+动作头峰值约 10,025 MiB；4B OFT 推理峰值约 10,643 MiB | **策略明确。** 不做 4B 全参；首轮固定 0.8B、100 step、batch=1 |
| CPU | 4 CPU 核 | **条件通过。** 训练可运行，但视频解码、tar 解包、Parquet 转换和 mock server 不宜与训练高并发 |
| 内存 | 64 GiB RAM | **通过但不宽裕。** 单任务转换可做；避免同时解压多任务、建立大量 video cache 或运行重型仿真 |
| 总存储配额 | 100 GB | **风险/待现场确认。** 这是总额，不是剩余空间；全量 30 task 明确不适合 |
| 无头仿真经验 | 已实测 EGL/MuJoCo LIBERO rollout | 对 RoboChallenge 不构成要求；仅说明后台服务器运行方式成熟 |

## 与 RoboChallenge 配置的匹配度

官方 Example 的首轮 YAML 使用 Qwen3.5-0.8B、QwenOFT、224×224、8D action、7D state、horizon=8、100 training steps、gradient checkpointing。该模型规模与训练长度明显低于此前因资源失败的 4B 全参数实验，因此在单卡 5090 上的首要风险不是显存，而是环境和数据 I/O。

预期的资源策略：

| 环节 | 服务器适配结论 | 执行约束 |
| --- | --- | --- |
| raw 数据下载/解包 | 可做，但受磁盘与网络限制 | 仅下载 `shred_paper`，不运行全量下载 shell |
| LeRobot 转换 | 可做 | 使用 1 个 CPU 任务；转换输出的视频是 raw 视频的符号链接，不能删 raw task |
| dataloader/forward | 可做 | 先检查两路视频解码；避免第一次就在 DeepSpeed 中定位数据错误 |
| 100-step QwenOFT | 可做 | `NUM_GPUS=1`、`BATCH=1`、`WANDB_MODE=disabled`；记录峰值显存 |
| local self-test | 可做 | GPU 独占运行 3 次推理，输出 `(8,8)` 与 latency |
| upstream mock | 可做，取决于外部 record | mock server 用独立 venv；确认 9098 端口和 record 目录 |
| 全量 30 task / 长训 | 不建议 | 磁盘、CPU 和评测资产都不匹配当前配额 |

## 必须在服务器执行的现场检查

以下命令输出应保存到 `Project_Analysis/evidence/robochallenge/00_preflight/`。在它们通过前，不开始下载。

```bash
git -C /path/to/starVLA rev-parse HEAD
git -C /path/to/starVLA status --short
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
nvidia-smi
df -h /path/to/starVLA
df -ih /path/to/starVLA
free -h
nproc

conda activate starVLA_dev
which nvcc || true
nvcc --version || true
python - <<'PY'
import torch, accelerate, cv2, pyarrow
print('torch=', torch.__version__)
print('cuda=', torch.cuda.is_available())
print('gpu=', torch.cuda.get_device_name(0))
print('bf16=', torch.cuda.is_bf16_supported())
print('accelerate=', accelerate.__version__)
print('cv2=', cv2.__version__)
print('pyarrow=', pyarrow.__version__)
PY
```

### 判定门槛

| Gate | 必须满足的条件 | 未满足时的动作 |
| --- | --- | --- |
| G1：磁盘 | `Avail ≥ 80 GB`；100 GB 更稳妥 | 释放旧 cache/无关产物，或改用独立数据盘；不开始下载 |
| G2：GPU | RTX 5090 可见、BF16 为 `True`、无其他长期显存占用 | 清理 GPU 作业或调整作业排队 |
| G3：CUDA | nvcc 对应真实 toolkit，DeepSpeed 可编译/加载 | 设置实际 `CUDA_HOME`；不要沿用项目中的集群示例路径 |
| G4：依赖 | `accelerate`、`cv2`、`pyarrow` 可 import | 仅向 `starVLA_dev` 补项目缺包，并记录版本 |
| G5：网络 | 可访问 Hugging Face 与 GitHub | 先解决代理/令牌；数据与 mock repo 都依赖网络 |
| G6：代码 | RoboChallenge 四处迁移补丁已应用，import 和 `bash -n` 通过 | 修路径后重跑 gate |
| G7：mock 资产 | 上游 UR5 record 目录可定位 | 没有该资产时完成训练/self-test，但把 mock 标为外部阻塞 |

## 当前可执行范围

在 G1–G6 通过、G7 尚未确认的情况下，服务器仍可完成以下 80% 的项目工作：

1. `shred_paper` 单任务下载与 LeRobot 转换；
2. metadata、视频、7D state/8D action 检查；
3. dataloader 与 QwenOFT forward gate；
4. 100-step checkpoint；
5. synthetic-observation 的 3 次 self-test 和 latency 记录。

这可称为“**RoboChallenge 数据—训练—本地 policy 推理复现**”。只有 G7 也通过并完成连续 HTTP GET/POST 后，才能升级为“**RoboChallenge mock 协议闭环复现**”。

## 最终判定

| 目标 | 当前服务器条件 | 建议 |
| --- | --- | --- |
| 0.8B 单任务训练与 self-test | **可执行，等待现场 gate 确认** | 立刻按完整计划 P0→P3 推进 |
| mock server 协议闭环 | **条件可执行** | 先确认上游 `20260413` record 资产；不存在则向官方/维护方索取 |
| 全任务 Table30v2 复现 | **不适合** | 不在 100 GB/4 CPU 配额上尝试 |
| 4B 全参数训练 | **不适合** | 已有 OOM 实测；不作为本项目路线 |

综上，服务器不是项目的根本障碍；它适合做单任务、0.8B、短训练和协议验证。启动前唯一不可省略的实际检查是**剩余磁盘**，其次是新 main 的路径补丁与上游 mock 数据资产。
