"""Step-2 dry-run: replay a real LeRobot episode through the StarVLA policy server.

Loads one episode (parquet state + ego-view mp4), rebuilds the TRAINING-TIME
observation (224x224 resized image + 72D state concatenated in registry order +
task text), feeds strided frames to the policy server, and reports the predicted
78D -> 64/7/7 action groups against the recorded ground-truth action.

This is dry-run step 2 of step3_deployment/README.md ("replay_lerobot_episode.py
with a real recorded episode"). It needs no robot / controller, only the dataset
and a running policy server.

Usage (server must already be running via run_policy_server.sh):
    python eval_files/replay_lerobot_episode.py \
        --dataset-path playground/Datasets/UnitreeG1_WholeBody/lerobot/test_sonic \
        --ckpt-path results/Checkpoints/starvla_qwenoft_g1_sonic_smoke/checkpoints/steps_1000_pytorch_model.pt \
        --episode 0 --stride 8 --max-frames 30
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from eval_files.model2unitree_g1_interface import ModelClient

# 72D state = concatenation of these parquet keys in the order defined by
# step2_training/train_files/data_registry/data_config.py `state_keys`.
STATE_KEYS = [
    ("observation.state", 43),
    ("observation.eef_state", 14),
    ("observation.root_orientation", 4),
    ("observation.projected_gravity", 3),
    ("observation.cpp_rotation_offset", 4),
    ("observation.init_base_quat", 4),
]  # total 72
assert sum(d for _, d in STATE_KEYS) == 72


def build_state(row) -> np.ndarray:
    pieces = []
    for key, dim in STATE_KEYS:
        v = np.asarray(row[key], dtype=np.float32).reshape(-1)
        assert v.shape[0] == dim, f"{key}: expected {dim}, got {v.shape[0]}"
        pieces.append(v)
    return np.concatenate(pieces).astype(np.float32)


def load_task(tasks_path: Path, task_index: int) -> str:
    fallback = "pick the toy on the table, and put it into the box."
    if not tasks_path.exists():
        return fallback
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    for t in tasks:
        if int(t.get("task_index", 0)) == int(task_index):
            return t["task"]
    return tasks[0]["task"] if tasks else fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True, help="LeRobot v2.1 dataset dir")
    parser.add_argument("--ckpt-path", required=True, help="checkpoint path (loaded by the server)")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--stride", type=int, default=8, help="sample every Nth frame")
    parser.add_argument("--max-frames", type=int, default=30, help="cap on sampled frames")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5694)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("replay_lerobot_episode")

    ds = Path(args.dataset_path)
    ep = f"{args.episode:06d}"
    parquet_path = ds / "data" / "chunk-000" / f"episode_{ep}.parquet"
    video_path = ds / "videos" / "chunk-000" / "observation.images.ego_view" / f"episode_{ep}.mp4"
    tasks_path = ds / "meta" / "tasks.jsonl"

    df = pq.read_table(parquet_path).to_pandas()
    n_rows = len(df)
    logger.info("episode %d: %d rows", args.episode, n_rows)
    task = load_task(tasks_path, int(np.asarray(df["task_index"].iloc[0]).reshape(-1)[0]))
    logger.info("task: %s", task)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("video frames: %d", n_video_frames)

    client = ModelClient(policy_ckpt_path=args.ckpt_path, host=args.server_host, port=args.server_port)
    logger.info("connected; unnorm_key=%s", client.unnorm_key)

    idxs = list(range(0, n_rows, args.stride))[: args.max_frames]
    times, mse_mt, mse_lh, mse_rh = [], [], [], []
    n_inf = 0
    for i in idxs:
        # seek by the row's recorded frame index (1:1 with video for unfiltered data)
        fi = int(np.asarray(df["frame_index"].iloc[i]).reshape(-1)[0])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            logger.warning("frame index %d not readable; skipping", fi)
            continue
        # cv2 decodes BGR; the training backend (torchvision_av) uses RGB
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

        row = df.iloc[i]
        obs = {"image": img, "state": build_state(row), "lang": task}

        t0 = time.time()
        out, _info = client.get_action(obs)
        dt = time.time() - t0
        times.append(dt)

        mt = np.asarray(out["action.motion_token"])
        lh = np.asarray(out["action.left_hand_joints"])
        rh = np.asarray(out["action.right_hand_joints"])

        # recorded ground-truth action for the current frame (compare to step 0)
        rec_mt = np.asarray(row["action.motion_token"], dtype=np.float32).reshape(-1)  # 64
        rec_lh = np.asarray(row["teleop.left_hand_joints"], dtype=np.float32).reshape(-1)  # 7
        rec_rh = np.asarray(row["teleop.right_hand_joints"], dtype=np.float32).reshape(-1)  # 7
        mse_mt.append(float(np.mean((mt[0] - rec_mt) ** 2)))
        mse_lh.append(float(np.mean((lh[0] - rec_lh) ** 2)))
        mse_rh.append(float(np.mean((rh[0] - rec_rh) ** 2)))

        n_inf += 1
        if n_inf % 10 == 0:
            logger.info("...%d/%d frames (avg %.1f ms)", n_inf, len(idxs), 1e3 * float(np.mean(times)))

    cap.release()
    if not times:
        raise RuntimeError("no frames were replayed")

    logger.info("replay done: %d inferences, avg latency %.1f ms", n_inf, 1e3 * float(np.mean(times)))
    logger.info(
        "pred-vs-recorded MSE: motion_token=%.4f left_hand=%.4f right_hand=%.4f",
        float(np.mean(mse_mt)), float(np.mean(mse_lh)), float(np.mean(mse_rh)),
    )
    logger.info("OK ✅  replay produced 64/7/7 splits for %d real frames", n_inf)
    client.close()


if __name__ == "__main__":
    main()
