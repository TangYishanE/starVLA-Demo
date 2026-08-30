#!/usr/bin/env python3
"""Realman -> NVIDIA panda-open-drawer adaptations (idempotent).

1. Write meta/modality.json (registry logical keys -> NVIDIA physical keys).
2. Patch data_registry/data_config.py: <your_dataset> -> panda-open-drawer.
3. Patch both YAMLs: data_root_dir + action_mode=abs (NVIDIA action already delta).
4. Patch train_realman_dp.sh: default data_mix -> realman_example_dp.
"""
import json
from pathlib import Path

ROOT = Path("/225010261/StarVLA/src/starVLA")
DATASET = ROOT / "playground/Datasets/Realman/lerobot/panda-open-drawer"

# NVIDIA panda-open-drawer state (25D):
#   eef_pos(0:3) eef_quat(3:7) finger_joint1_pos(7) finger_joint2_pos(8)
#   finger_vel(9:11) panda_joint1..7_pos(11:18) panda_joint1..7_vel(18:25)
# action (8D): panda_joint1..7_delta_pos(0:7) + gripper(7)
MODALITY = {
    "state": {
        "joints": {"start": 11, "end": 18, "original_key": "observation.state"},
        "gripper": {"start": 7, "end": 8, "original_key": "observation.state"},
    },
    "action": {
        "delta_joints": {"start": 0, "end": 7, "original_key": "action"},
        "gripper_close": {"start": 7, "end": 8, "original_key": "action"},
    },
    "video": {
        "cam0_rgb": {"original_key": "observation.images.world_camera"},
        "cam1_rgb": {"original_key": "observation.images.hand_camera"},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"},
    },
}


def write_modality() -> None:
    meta = DATASET / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "modality.json").write_text(json.dumps(MODALITY, indent=2) + "\n")
    print("wrote modality.json")


def patch_registry() -> None:
    p = ROOT / "examples/realRobots/Realman/train_files/data_registry/data_config.py"
    s = p.read_text()
    n = s.count("<your_dataset>")
    s = s.replace("<your_dataset>", "panda-open-drawer")
    p.write_text(s)
    print(f"patched data_config.py ({n} placeholder(s) replaced)")


def patch_yaml(rel: str) -> None:
    p = ROOT / rel
    s = p.read_text()
    s = s.replace("data_root_dir: /path/to/lerobot_datasets",
                  "data_root_dir: playground/Datasets/Realman/lerobot")
    # NVIDIA action is already delta -> do NOT apply an extra delta conversion.
    s = s.replace('action_mode: "delta"',
                  'action_mode: "abs"  # NVIDIA action already delta')
    p.write_text(s)
    print(f"patched {rel}")


def patch_dp_sh() -> None:
    p = ROOT / "examples/realRobots/Realman/train_files/train_realman_dp.sh"
    s = p.read_text()
    s = s.replace("data_mix=${DATA_MIX:-realman_example}",
                  "data_mix=${DATA_MIX:-realman_example_dp}")
    p.write_text(s)
    print("patched train_realman_dp.sh")


if __name__ == "__main__":
    write_modality()
    patch_registry()
    patch_yaml("examples/realRobots/Realman/train_files/train_realman_act.yaml")
    patch_yaml("examples/realRobots/Realman/train_files/train_realman_dp.yaml")
    patch_dp_sh()
    print("ADAPT DONE")
