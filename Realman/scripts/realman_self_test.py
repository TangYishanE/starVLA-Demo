#!/usr/bin/env python3
"""Realman no-robot self-test: reload an ACT/DP checkpoint and validate the 8D chunk.

Usage: realman_self_test.py <ckpt.pt> <T_expected> <label>
Loads the framework in-process (no policy server), feeds one synthetic obs
(2x224x224 + 8D state + lang), and asserts the unnormalized action chunk is
(T_expected, 8) and finite.
"""
import sys

import numpy as np

from starVLA.model.framework.base_framework import baseframework
from deployment.model_server.policy_norm_processor import PolicyNormProcessor

LANG = "open the drawer"


def make_obs() -> dict:
    rng = np.random.default_rng(42)
    cam0 = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    cam1 = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    return {"image": [cam0, cam1], "state": np.zeros(8, dtype=np.float32), "lang": LANG}


def run(ckpt: str, t_expected: int, label: str) -> np.ndarray:
    proc = PolicyNormProcessor(ckpt, unnorm_key="new_embodiment")
    fw = baseframework.from_pretrained(ckpt).to("cuda").eval()
    out = fw.predict_action(examples=[make_obs()])
    norm = np.asarray(out["normalized_actions"])  # (B=1, T, 8)
    assert norm.ndim == 3 and norm.shape[0] == 1, f"{label}: bad normalized shape {norm.shape}"
    acts = proc.unapply_actions(norm[0])          # (T, 8)
    assert acts.shape == (t_expected, 8), f"{label}: {acts.shape} != ({t_expected}, 8)"
    assert np.isfinite(acts).all(), f"{label}: NaN/Inf in actions"
    print(f"[PASS] {label}: {acts.shape} finite, action[0]={np.round(acts[0], 4)}")
    return acts


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: realman_self_test.py <ckpt.pt> <T_expected> <label>")
        sys.exit(2)
    run(sys.argv[1], int(sys.argv[2]), sys.argv[3])
