#!/usr/bin/env python3
"""P3 no-robot validation: connect to the StarVLA policy server with synthetic
observations (2x224x224 + lang) and verify the Franka 7D delta-EE action chunk.

The current server (policy_wrapper.py) returns already-unnormalized
``actions`` of shape [B, T, action_dim]; the client just validates shape,
finiteness and prints latency. (G3 fix: import from the real module path.)
"""
import time

import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

HOST, PORT = "127.0.0.1", 5694
LANG = "lift the red cube"
EXPECT_SHAPE = (16, 7)  # action_horizon=16, action_dim=7


def make_synthetic_obs():
    rng = np.random.default_rng(42)
    base = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    ego = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    return [base, ego]


def main() -> None:
    images = make_synthetic_obs()
    client = WebsocketClientPolicy(host=HOST, port=PORT)
    print("connected to policy server %s:%s" % (HOST, PORT))

    t0 = time.time()
    result = client.predict_action({"examples": [{"image": images, "lang": LANG}]})
    latency = time.time() - t0
    print("response top-level keys:", list(result.keys()) if isinstance(result, dict) else type(result))

    data = result.get("data", result) if isinstance(result, dict) else result
    if isinstance(data, dict):
        print("data keys:", list(data.keys()))

    if isinstance(data, dict) and "actions" in data:
        acts = np.asarray(data["actions"])
        print("returned key: actions (already unnormalized)")
    elif isinstance(data, dict) and "normalized_actions" in data:
        acts = np.asarray(data["normalized_actions"])
        print("returned key: normalized_actions (client unnorm would be needed)")
    else:
        raise RuntimeError("no actions key in response: %s" % (list(data.keys()) if isinstance(data, dict) else "?"))

    print("raw shape:", acts.shape, "dtype:", acts.dtype)
    if acts.ndim == 3:
        acts = acts[0]
    print("chunk shape:", acts.shape)

    finite = bool(np.isfinite(acts).all())
    print("all finite:", finite)
    print("action[0]:", np.round(acts[0], 4))
    print("gripper col values:", np.unique(np.round(acts[:, 6], 3))[:8])

    assert acts.shape == EXPECT_SHAPE, "shape mismatch: %s != %s" % (acts.shape, EXPECT_SHAPE)
    assert finite, "NaN/Inf in predicted actions"
    assert np.isfinite(latency) and latency >= 0
    print("LATENCY(s): %.3f" % latency)
    print("P3 VALIDATION PASSED")
    client.close()


if __name__ == "__main__":
    main()
