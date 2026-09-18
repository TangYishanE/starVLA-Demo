#!/usr/bin/env python3
"""Convert RoboMimic `lift_real` (real Franka, rt_benchmark) `demo.hdf5` into a
LeRobot v2.1 dataset matching the StarVLA Franka example contract.

Source HDF5 structure (rt_benchmark / lift_real):
    data/demo_N/
        obs.image          (T, 120, 120, 3) uint8   -> base_view
        obs.image_wrist    (T, 120, 120, 3) uint8   -> ego_view
        obs.ee_pose        (T, 7) float64 = [x,y,z,qx,qy,qz,qw]
        obs.gripper_position (T, 1)
        obs.q / obs.dq / obs.ee_vel   (not used)
        actions             (T, 7) = [dx,dy,dz,droll,dpitch,dyaw,gripper]  (OSC_POSE delta-EE)
    mask/train             (180,) demo names
    mask/valid             (20,)  demo names

Target contract (SingleFrankaRobotiqDeltaEefDataConfig):
    observation.state (6,)  = eef_position(3) + eef_rotation(3, axis-angle)
    action            (7,)  = delta_eef_position(3) + delta_eef_rotation(3) + gripper_close(1)
    video: observation.images.base_view + observation.images.ego_view (224x224)

The RoboMimic OSC_POSE action is already a 7D delta-EE command, so action maps
with no arithmetic. State rotation is converted quat -> axis-angle.

Usage:
    python convert_robomimic_to_lerobot.py \
        --hdf5 /path/to/demo.hdf5 \
        --out-root /path/to/franka_pick_and_place_lerobot \
        [--split train] [--task "lift the red cube"] [--max-episodes N] [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# RoboMimic obs key -> Franka video subkey
CAMERA_MAP = {
    "image": "base_view",
    "image_wrist": "ego_view",
    # robosuite-style fallbacks
    "agentview_image": "base_view",
    "robot0_eye_in_hand_image": "ego_view",
}

STATE_DIM = 6   # eef_position(3) + eef_rotation(3, axis-angle)
ACTION_DIM = 7  # delta_eef_position(3) + delta_eef_rotation(3) + gripper_close(1)
TARGET_H, TARGET_W = 224, 224
FPS = 20  # RoboMimic real data recorded at 20 Hz


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """(N,4) quaternions [x,y,z,w] -> (N,3) axis-angle."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sin(angle / 2.0)
    safe = np.where(np.abs(sin_half) > 1e-8, sin_half, 1.0)
    axis = np.stack([x, y, z], axis=-1) / safe[..., None]
    aa = angle[..., None] * axis
    aa = np.where(np.abs(angle[..., None]) < 1e-8, 0.0, aa)
    return aa.astype(np.float32)


def resolve_cameras(obs: h5py.Group) -> dict[str, str]:
    """Return {video_subkey: hdf5_obs_key} for base_view + ego_view."""
    found: dict[str, str] = {}
    for key in obs.keys():
        if key in CAMERA_MAP:
            subkey = CAMERA_MAP[key]
            if subkey not in found:
                found[subkey] = key
    return found


def write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    """Write (T,H,W,3) uint8 RGB frames to an H.264 mp4 via PyAV (same backend
    as the StarVLA dataloader's torchvision_av)."""
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, C = frames.shape
    container = av.open(str(path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = W
    stream.height = H
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23", "preset": "veryfast"}
    for i in range(T):
        frame = av.VideoFrame.from_ndarray(frames[i], format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _build_features(h: int, w: int) -> dict:
    feats: dict = {}
    for cam in ("base_view", "ego_view"):
        feats[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channel"],
            "info": {
                "video.height": h, "video.width": w, "video.channels": 3,
                "video.fps": FPS, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False, "has_audio": False,
            },
        }
    feats["observation.state"] = {"dtype": "float32", "shape": [STATE_DIM], "names": ["state"]}
    feats["action"] = {"dtype": "float32", "shape": [ACTION_DIM], "names": ["action"]}
    for k in ("timestamp",):
        feats[k] = {"dtype": "float32", "shape": [1], "names": None}
    for k in ("frame_index", "episode_index", "index", "task_index"):
        feats[k] = {"dtype": "int64", "shape": [1], "names": None}
    return feats


def convert(hdf5_path: Path, out_root: Path, task: str, split: str,
            max_episodes: int | None, overwrite: bool) -> None:
    import cv2

    if out_root.exists():
        if overwrite:
            shutil.rmtree(out_root)
        else:
            raise FileExistsError("%s exists (pass --overwrite)" % out_root)
    (out_root / "meta").mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        all_demos = sorted([k for k in f["data"].keys()], key=lambda x: int(x.split("_")[1]))

        # select demos by split (rt_benchmark stores train/valid as name lists)
        if split and f"mask/{split}" in f:
            sel = {x.decode() for x in f[f"mask/{split}"][:]}
            demo_names = [d for d in all_demos if d in sel]
            print(f"[convert] split={split}: {len(demo_names)} demos selected from {len(all_demos)}")
        else:
            demo_names = all_demos
        if max_episodes:
            demo_names = demo_names[:max_episodes]

        total_frames = 0
        new_ep_idx = 0
        episodes_lines: list[str] = []

        for demo_name in demo_names:
            grp = f["data"][demo_name]
            obs = grp["obs"]
            actions = grp["actions"][:]          # (T,7) delta-EE

            cams = resolve_cameras(obs)
            if set(cams) != {"base_view", "ego_view"}:
                print("[skip] %s: cameras=%s" % (demo_name, cams))
                continue

            # --- state: eef_position(3) + eef_rotation(3, axis-angle) ---
            ee = obs["ee_pose"][:]                 # (T,7) pos+quat
            eef_pos = ee[:, 0:3]
            eef_rot = quat_to_axis_angle(ee[:, 3:7])
            state = np.concatenate([eef_pos, eef_rot], axis=-1).astype(np.float32)  # (T,6)

            action = actions.astype(np.float32)    # (T,7) already delta-EE
            n = state.shape[0]

            ts = np.arange(n, dtype=np.float32) / FPS

            episode_chunk = new_ep_idx // 1000
            df_path = out_root / f"data/chunk-{episode_chunk:03d}/episode_{new_ep_idx:06d}.parquet"
            df_path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.table({
                "observation.state": pa.array([r.tolist() for r in state], type=pa.list_(pa.float32(), STATE_DIM)),
                "action": pa.array([r.tolist() for r in action], type=pa.list_(pa.float32(), ACTION_DIM)),
                "timestamp": pa.array(ts, type=pa.float32()),
                "frame_index": pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
                "episode_index": pa.array(np.full(n, new_ep_idx, dtype=np.int64), type=pa.int64()),
                "index": pa.array(np.arange(total_frames, total_frames + n, dtype=np.int64), type=pa.int64()),
                "task_index": pa.array(np.full(n, 0, dtype=np.int64), type=pa.int64()),
            })
            pq.write_table(table, df_path)

            for subkey in ("base_view", "ego_view"):
                frames = obs[cams[subkey]][:]                     # (n,120,120,3)
                resized = np.stack([cv2.resize(fr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
                                    for fr in frames])
                vpath = out_root / f"videos/chunk-{episode_chunk:03d}/observation.images.{subkey}/episode_{new_ep_idx:06d}.mp4"
                write_video(vpath, resized, FPS)

            episodes_lines.append(json.dumps({"episode_index": new_ep_idx, "tasks": [task], "length": n}))
            total_frames += n
            new_ep_idx += 1
            if new_ep_idx % 25 == 0:
                print("[convert] %d/%d episodes, %d frames" % (new_ep_idx, len(demo_names), total_frames))

    (out_root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")
    (out_root / "meta/episodes.jsonl").write_text("\n".join(episodes_lines) + "\n")

    modality = {
        "state": {
            "eef_position": {"original_key": "observation.state", "start": 0, "end": 3},
            "eef_rotation": {"original_key": "observation.state", "start": 3, "end": 6},
        },
        "action": {
            "delta_eef_position": {"original_key": "action", "start": 0, "end": 3},
            "delta_eef_rotation": {"original_key": "action", "start": 3, "end": 6},
            "gripper_close": {"original_key": "action", "start": 6, "end": 7},
        },
        "video": {
            "base_view": {"original_key": "observation.images.base_view"},
            "ego_view": {"original_key": "observation.images.ego_view"},
        },
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }
    (out_root / "meta/modality.json").write_text(json.dumps(modality, indent=2))

    (out_root / "meta/embodiment.json").write_text(json.dumps({
        "robot_name": "franka", "robot_type": "franka",
        "record_frequency": FPS, "body_controller_frequency": FPS,
        "hand_controller_frequency": FPS, "embodiment_tag": "new_embodiment",
    }, indent=2))

    chunks = (new_ep_idx + 1000 - 1) // 1000
    info = {
        "codebase_version": "v2.1", "robot_type": "franka",
        "total_episodes": new_ep_idx, "total_frames": total_frames,
        "total_tasks": 1, "total_videos": new_ep_idx * 2, "total_chunks": chunks,
        "chunks_size": 1000, "fps": FPS,
        "splits": {"train": f"0:{new_ep_idx}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": _build_features(TARGET_H, TARGET_W),
    }
    (out_root / "meta/info.json").write_text(json.dumps(info, indent=4))

    print("[convert] done. episodes=%d frames=%d" % (new_ep_idx, total_frames))
    print("[convert] output -> %s" % out_root)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--task", default="lift the red cube")
    p.add_argument("--split", default="train")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    convert(args.hdf5, args.out_root, args.task, args.split, args.max_episodes, args.overwrite)


if __name__ == "__main__":
    main()
