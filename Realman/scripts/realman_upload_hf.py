#!/usr/bin/env python3
"""Upload Realman ACT/DP smoke checkpoints to private HF model repos.

Mirrors tmp/franka_upload_hf.py. The server is already logged in as TangYishan.
"""
from huggingface_hub import HfApi, create_repo

REPOS = {
    "results/Checkpoints/realman_act_smoke": "TangYishan/starvla-act-realman-panda-open-drawer",
    "results/Checkpoints/realman_dp_smoke": "TangYishan/starvla-diffusionpolicy-realman-panda-open-drawer",
}

api = HfApi()
for folder, repo in REPOS.items():
    print(f"creating repo {repo} (private)")
    create_repo(repo, repo_type="model", exist_ok=True, private=True)
    print(f"uploading {folder} -> {repo} ...")
    api.upload_large_folder(folder_path=folder, repo_id=repo, repo_type="model")
    print(f"UPLOAD DONE: {repo}")
