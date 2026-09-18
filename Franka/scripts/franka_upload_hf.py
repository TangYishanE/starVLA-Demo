#!/usr/bin/env python3
"""Upload the Franka QwenOFT smoke checkpoint to Hugging Face.

Mirrors deployment/upload/push_model_to_hf.py (create_repo + upload_large_folder).
Run from the starVLA repo root; the server is already logged in as TangYishan.
"""
from huggingface_hub import HfApi, create_repo

REPO = "TangYishan/starvla-qwenoft-franka-robomimic-lift"
FOLDER = "results/Checkpoints/franka_qwenoft_smoke"

print("creating repo %s (exist_ok)" % REPO)
create_repo(REPO, repo_type="model", exist_ok=True)
api = HfApi()
print("uploading folder %s -> %s ..." % (FOLDER, REPO))
api.upload_large_folder(folder_path=FOLDER, repo_id=REPO, repo_type="model")
print("UPLOAD DONE: %s" % REPO)
