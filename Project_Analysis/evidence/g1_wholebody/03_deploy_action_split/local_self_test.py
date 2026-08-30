"""Step-1 self test for UnitreeG1_WholeBody: connect to StarVLA policy server
with a synthetic GR00T-style observation and validate the 78D action split.

This exercises the full deployment path WITHOUT a real robot / SONIC / WBC:
  * ModelClient -> WebsocketClientPolicy -> StarVLA policy server
  * synthetic 224x224 ego image + 72D state + language prompt
  * get_action -> action split into 64 (motion_token) + 7 (left) + 7 (right)

The checkpoint is loaded by the SERVER (run_policy_server.sh); this client only
validates the observation contract and the returned 78D action split.

Example:
    python eval_files/local_self_test.py \\
        --ckpt-path results/Checkpoints/starvla_qwenoft_g1_sonic_smoke/checkpoints/steps_1000_pytorch_model.pt \\
        --server-host 127.0.0.1 --server-port 5694
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from eval_files.model2unitree_g1_interface import ModelClient


def _make_synthetic_observation() -> dict:
    image = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    state = np.random.uniform(-1.0, 1.0, 72).astype(np.float32)
    return {
        "image": image,
        "state": state,
        "lang": "pick the toy on the table, and put it into the box.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True, help="checkpoint path (loaded by the server)")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5694)
    parser.add_argument("--n_warmup", type=int, default=1)
    parser.add_argument("--n_runs", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("g1_self_test")

    client = ModelClient(
        policy_ckpt_path=args.ckpt_path,
        host=args.server_host,
        port=args.server_port,
    )
    meta = client.get_server_metadata()
    logger.info("connected to policy server; unnorm_key=%s", client.unnorm_key)
    logger.info(
        "server_meta=%s",
        {k: meta[k] for k in list(meta)[:10]} if isinstance(meta, dict) else type(meta),
    )

    obs = _make_synthetic_observation()
    logger.info("synthetic obs: image=%s state=%s lang=%r", obs["image"].shape, obs["state"].shape, obs["lang"])

    # warmup (jit/compile, autotune)
    for _ in range(args.n_warmup):
        client.get_action(obs)

    times = []
    for i in range(args.n_runs):
        t0 = time.time()
        out, _info = client.get_action(obs)
        dt = time.time() - t0
        times.append(dt)
        logger.info(
            "run %d: %.1f ms | motion_token=%s left_hand=%s right_hand=%s",
            i, dt * 1e3,
            out["action.motion_token"].shape,
            out["action.left_hand_joints"].shape,
            out["action.right_hand_joints"].shape,
        )

    # assertions: (T, 64) + (T, 7) + (T, 7)
    out, _info = client.get_action(obs)
    mt = np.asarray(out["action.motion_token"])
    lh = np.asarray(out["action.left_hand_joints"])
    rh = np.asarray(out["action.right_hand_joints"])
    T = mt.shape[0]
    assert mt.shape == (T, 64), f"motion_token shape {mt.shape} != ({T}, 64)"
    assert lh.shape == (T, 7), f"left_hand shape {lh.shape} != ({T}, 7)"
    assert rh.shape == (T, 7), f"right_hand shape {rh.shape} != ({T}, 7)"
    assert np.isfinite(mt).all() and np.isfinite(lh).all() and np.isfinite(rh).all(), "non-finite values in action"

    logger.info("avg latency over %d runs: %.1f ms", len(times), 1e3 * float(np.mean(times)))
    logger.info("OK ✅  policy returns %d steps -> 64+7+7 (78D), split correct", T)
    client.close()


if __name__ == "__main__":
    main()
