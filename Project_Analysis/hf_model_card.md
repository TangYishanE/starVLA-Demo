---
license: mit
tags:
- starvla
- vla
- robotics
- unitree-g1
- wholebody
- qwen-oft
- sonic
---

# StarVLA QwenOFT — Unitree G1 WholeBody (SONIC 78D action)

Smoke checkpoint for the StarVLA `examples/realRobots/UnitreeG1_WholeBody` workflow.

## What this is
- **Model**: QwenOFT — Qwen3.5-0.8B backbone + L1-regression MLP action head
- **Action contract**: SONIC-style **78D** = 64D `motion_token` + 7D `left_hand_joints` + 7D `right_hand_joints`
- **State**: **72D** proprioception = 43D `observation.state` + 14D `observation.eef_state` + 4D root_orientation + 3D projected_gravity + 4D cpp_rotation_offset + 4D init_base_quat
- **Normalization**: q99 (`dataset_statistics.json`; unnorm_key = `new_embodiment`)
- **Image**: single ego view, 224×224
- **Horizon**: 8

## Scope / caveats
- **1000-step SMOKE run** — not converged (final L1 loss ≈ 0.60). Do not expect task-level performance.
- Trained on the public `cloudwalk-research/gr00t-g1-grab-bottle-right-hand-v11` dataset (355 episodes), **not** the private `test_sonic` example data.
- Validates the data → train → policy-server → 78D-action-split engineering chain. It does **not** imply real-robot closed-loop control or task success (no physical G1 / SONIC / WBC used).

## Files
- `checkpoints/steps_1000_pytorch_model.pt` — StarVLA checkpoint (2.24 GB)
- `config.yaml` / `config.full.yaml` — model + training config
- `dataset_statistics.json` — q99 normalization stats (required for serving)
- `summary.jsonl` — training checkpoint-save log

## Serving (StarVLA)
Requires the StarVLA repo and the Qwen3.5-0.8B backbone (`base_vlm` in `config.yaml`):

```bash
cd starVLA
export PYTHONPATH=$PWD:${PYTHONPATH}
CUDA_VISIBLE_DEVICES=0 python deployment/model_server/server_policy.py \
  --ckpt_path <this_repo>/checkpoints/steps_1000_pytorch_model.pt \
  --port 5694 \
  --use_bf16
```

## Reproduction
Training config / scripts live in the StarVLA example:
`examples/realRobots/UnitreeG1_WholeBody/step2_training/train_files/`.

- Dataset: `cloudwalk-research/gr00t-g1-grab-bottle-right-hand-v11` (LeRobot v2.1)
- Backbone: `Qwen/Qwen3.5-0.8B`
- `dataset_statistics.json` top-level key `new_embodiment` = `EmbodimentTag.NEW_EMBODIMENT`
