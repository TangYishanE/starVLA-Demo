"""Step-3 dry-run: mock G1 controller consuming StarVLA 78D action chunks.

Inspects the 64/7/7 action groups (motion_token / left_hand / right_hand) the
way the GR00T-WholeBodyControl / SONIC consumer side would, but WITHOUT a real
robot: validate shape + finiteness + value ranges, report per-group summary
stats, and simulate a publish loop with a stale-action (frequency) check and
optional hand-joint clipping.

This is dry-run step 3 of step3_deployment/README.md ("mock_g1_controller.py
to inspect action groups").

Usage (server must already be running via run_policy_server.sh):
    python eval_files/mock_g1_controller.py \
        --ckpt-path results/Checkpoints/starvla_qwenoft_g1_sonic_smoke/checkpoints/steps_1000_pytorch_model.pt \
        --server-host 127.0.0.1 --server-port 5694
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from eval_files.model2unitree_g1_interface import ModelClient

MOTION_TOKEN_DIM = 64
HAND_DIM = 7


def _make_synthetic_observation() -> dict:
    image = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    state = np.random.uniform(-1.0, 1.0, 72).astype(np.float32)
    return {
        "image": image,
        "state": state,
        "lang": "pick the toy on the table, and put it into the box.",
    }


def inspect_chunk(mt: np.ndarray, lh: np.ndarray, rh: np.ndarray) -> dict:
    """Validate and summarize one 78D -> 64/7/7 action chunk."""
    assert mt.ndim == 2 and mt.shape[1] == MOTION_TOKEN_DIM, f"motion_token shape {mt.shape}"
    assert lh.ndim == 2 and lh.shape[1] == HAND_DIM, f"left_hand shape {lh.shape}"
    assert rh.ndim == 2 and rh.shape[1] == HAND_DIM, f"right_hand shape {rh.shape}"
    T = mt.shape[0]
    assert lh.shape[0] == T and rh.shape[0] == T, "group horizon mismatch"
    for name, arr in (("motion_token", mt), ("left_hand", lh), ("right_hand", rh)):
        assert np.isfinite(arr).all(), f"non-finite values in {name}"

    def stats(arr: np.ndarray):
        return (float(arr.min()), float(arr.max()), float(np.mean(arr)), float(np.std(arr)))

    return {
        "T": T,
        "motion_token": stats(mt),
        "left_hand": stats(lh),
        "right_hand": stats(rh),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True, help="checkpoint path (loaded by the server)")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5694)
    parser.add_argument("--n_iter", type=int, default=5, help="number of publish-loop iterations")
    parser.add_argument("--publish_hz", type=float, default=2.0, help="simulated controller frequency")
    parser.add_argument("--clip_hand", type=float, default=None, help="optional hand-joint clip bound")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("mock_g1_controller")

    client = ModelClient(policy_ckpt_path=args.ckpt_path, host=args.server_host, port=args.server_port)
    logger.info("connected; unnorm_key=%s", client.unnorm_key)

    obs = _make_synthetic_observation()
    period = 1.0 / args.publish_hz
    for i in range(args.n_iter):
        t0 = time.time()
        out, _info = client.get_action(obs)
        dt = time.time() - t0
        mt = np.asarray(out["action.motion_token"])
        lh = np.asarray(out["action.left_hand_joints"])
        rh = np.asarray(out["action.right_hand_joints"])
        s = inspect_chunk(mt, lh, rh)
        logger.info(
            "iter %d: T=%d | motion_token[min=%.3f max=%.3f] left_hand[min=%.3f max=%.3f] "
            "right_hand[min=%.3f max=%.3f] | latency=%.1fms",
            i, s["T"],
            s["motion_token"][0], s["motion_token"][1],
            s["left_hand"][0], s["left_hand"][1],
            s["right_hand"][0], s["right_hand"][1],
            dt * 1e3,
        )
        if dt > period:
            logger.warning(
                "inference (%.0fms) slower than publish period (%.0fms) -> stale-action risk",
                dt * 1e3, period * 1e3,
            )
        if args.clip_hand is not None:
            # simulate the adapter-side hand clipping before publishing
            _ = np.clip(lh, -args.clip_hand, args.clip_hand)
            _ = np.clip(rh, -args.clip_hand, args.clip_hand)

    logger.info("OK ✅  mock controller consumed %d chunks -> 64/7/7 split valid", args.n_iter)
    client.close()


if __name__ == "__main__":
    main()
