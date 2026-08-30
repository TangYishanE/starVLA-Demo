"""client_smoke.py — EgoVLA G1 server no-robot verification client (GR00T ZMQ).

Mirrors EgoVLAG1Policy._obs_to_example's input contract (bridge N1.7 / REAL_G1)
with synthetic observations. Run against server_egovla_g1.py on :5555.

Usage:
    python client_smoke.py [--host 127.0.0.1] [--port 5555] [--horizon 30]
"""
import argparse
import os
import sys
import time

import numpy as np

# Bootstrap: repo root is the dir containing pyproject.toml above this file.
_root = os.path.dirname(os.path.abspath(__file__))
while _root != os.path.dirname(_root) and not os.path.exists(os.path.join(_root, "pyproject.toml")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--horizon", type=int, default=30)
    args = ap.parse_args()

    from deployment.model_server.tools.zmq_policy_server import pack, unpack
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://{args.host}:{args.port}")

    def call(endpoint, data=None, needs_input=False):
        sock.send(pack({"endpoint": endpoint, "data": data or {}}))
        return unpack(sock.recv())

    # 0) ping
    print("ping:", call("ping", needs_input=False))

    # 1) modality config
    mc = call("get_modality_config", needs_input=False)
    print("modality_config:", {k: (v if not isinstance(v, list) else v) for k, v in mc.items()})

    # 2) synthetic GR00T-style observation (bridge N1.7 / REAL_G1 profile)
    T = args.horizon
    H = W = 640
    obs = {
        "video": {"ego_view": np.random.randint(0, 255, (1, T, H, W, 3), dtype=np.uint8)},
        "state": {
            "left_arm": np.zeros(7, np.float32),
            "right_arm": np.zeros(7, np.float32),
            "left_hand": np.zeros(7, np.float32),
            "right_hand": np.zeros(7, np.float32),
            "waist": np.zeros(1, np.float32),
            "left_wrist_eef_9d": np.array([0.3, 0.0, 0.6, 1, 0, 0, 0, 1, 0], np.float32),
            "right_wrist_eef_9d": np.array([0.3, 0.0, 0.6, 1, 0, 0, 0, 1, 0], np.float32),
        },
        "language": {"annotation.human.task_description": np.array(
            [["place the red ball in the box"]], dtype=object)},
    }

    # 3) get_action
    t0 = time.time()
    act, info = call("get_action", {"observation": obs, "options": {}}, needs_input=True)
    dt = time.time() - t0
    print(f"get_action latency: {dt:.3f}s")

    # 4) assert structure (action dict of (1,T,D), finite)
    expected = {
        "left_arm": (1, T, 7), "right_arm": (1, T, 7),
        "left_hand": (1, T, 7), "right_hand": (1, T, 7),
        "base_height_command": (1, T, 1), "navigate_command": (1, T, 3),
    }
    for k, shp in expected.items():
        v = np.asarray(act[k])
        assert v.shape == shp, f"{k}: {v.shape} != {shp}"
        assert np.isfinite(v).all(), f"{k}: non-finite"
        print(f"[OK] {k} {v.shape} finite, range [{v.min():.3f}, {v.max():.3f}]")
    print("client_smoke PASSED")


if __name__ == "__main__":
    main()
