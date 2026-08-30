#!/bin/bash
# Realman RM-75 no-robot reproduction chain (queued after franka + UnitreeG1 tasks).
# wait -> adapt -> dataloader gate -> forward gates -> ACT+DP 100-step train
# -> in-process self-test -> HF upload -> COMPLETE marker + summary.json.
set -uo pipefail
cd /225010261/StarVLA/src/starVLA || exit 1
export PYTHONPATH="$PWD"
export PATH="/225010261/StarVLA/envs/starvla-cu128/bin:/usr/local/cuda-12.5/bin:${PATH}"
export CUDA_HOME=/usr/local/cuda-12.5
export WANDB_MODE=disabled
PY=/225010261/StarVLA/envs/starvla-cu128/bin/python
RUN=results/realman_repro
LOG=tmp/logs/realman_repro_chain.log
mkdir -p "$RUN"
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
mark(){ echo "$*" > "$RUN/COMPLETE"; }

log "chain started; waiting for franka + G1 tasks"

# 1. wait for franka + G1 (training / post-train / upload / dry-run / servers) to exit
while pgrep -f "franka_qwenoft_smoke|franka_post_train_chain|franka_upload_hf|g1_dryrun_wait_chain|mock_g1_controller|replay_lerobot_episode|starvla_qwenoft_g1_sonic_smoke" >/dev/null 2>&1; do
  sleep 60
done
log "franka + G1 finished"

# 1b. GPU free (safety)
while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')" -gt 0 ]; do
  sleep 30
done
log "GPU free"

# 2. dataset present
DS=playground/Datasets/Realman/lerobot/panda-open-drawer
if [ ! -f "$DS/meta/info.json" ]; then
  log "FAIL: dataset missing"; mark "FAIL dataset_missing $(date -u +%FT%TZ)"; exit 1
fi
log "dataset OK"

# 3. adaptations
if ! $PY tmp/realman_adapt.py > tmp/logs/realman_adapt.log 2>&1; then
  log "FAIL: adapt"; mark "FAIL adapt $(date -u +%FT%TZ)"; exit 1
fi
log "adapt OK"

# 4. dataloader gate
if ! $PY starVLA/dataloader/lerobot_datasets.py \
  --config_yaml examples/realRobots/Realman/train_files/train_realman_act.yaml \
  > tmp/logs/realman_dataloader_gate.log 2>&1; then
  log "FAIL: dataloader gate"; mark "FAIL dataloader_gate $(date -u +%FT%TZ)"; exit 1
fi
log "dataloader gate OK"

# 5. forward gates (build_framework)
if ! $PY - <<'PY' > tmp/logs/realman_forward_gate.log 2>&1
from omegaconf import OmegaConf
from starVLA.model.framework.base_framework import build_framework
for y in ["examples/realRobots/Realman/train_files/train_realman_act.yaml",
          "examples/realRobots/Realman/train_files/train_realman_dp.yaml"]:
    cfg = OmegaConf.load(y)
    cfg.datasets.vla_data.data_root_dir = "playground/Datasets/Realman/lerobot"
    fw = build_framework(cfg)
    print(y, "->", type(fw).__name__)
print("FORWARD GATES OK")
PY
then
  log "FAIL: forward gate"; mark "FAIL forward_gate $(date -u +%FT%TZ)"; exit 1
fi
log "forward gates OK"

# 6. ACT 100-step training
log "ACT 100-step training"
export CUDA_VISIBLE_DEVICES=0
if ! accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/realRobots/Realman/train_files/train_realman_act.yaml \
  --framework.name ACT \
  --datasets.vla_data.data_root_dir playground/Datasets/Realman/lerobot \
  --datasets.vla_data.data_mix realman_example \
  --datasets.vla_data.per_device_batch_size 1 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.max_train_steps 100 \
  --trainer.save_interval 100 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100000 \
  --run_root_dir ./results/Checkpoints \
  --run_id realman_act_smoke \
  --wandb_project starVLA_realman \
  > tmp/logs/realman_act_train.log 2>&1; then
  log "FAIL: ACT train"; mark "FAIL act_train $(date -u +%FT%TZ)"; exit 1
fi
log "ACT train OK"

# 7. DP 100-step training
log "DP 100-step training"
if ! accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/realRobots/Realman/train_files/train_realman_dp.yaml \
  --framework.name DiffusionPolicy \
  --datasets.vla_data.data_root_dir playground/Datasets/Realman/lerobot \
  --datasets.vla_data.data_mix realman_example_dp \
  --datasets.vla_data.per_device_batch_size 1 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.max_train_steps 100 \
  --trainer.save_interval 100 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100000 \
  --run_root_dir ./results/Checkpoints \
  --run_id realman_dp_smoke \
  --wandb_project starVLA_realman \
  > tmp/logs/realman_dp_train.log 2>&1; then
  log "FAIL: DP train"; mark "FAIL dp_train $(date -u +%FT%TZ)"; exit 1
fi
log "DP train OK"

# 8. self-test (reload + unnormalize)
ACT_CKPT=results/Checkpoints/realman_act_smoke/checkpoints/steps_100_pytorch_model.pt
DP_CKPT=results/Checkpoints/realman_dp_smoke/checkpoints/steps_100_pytorch_model.pt
ST_OK=1
$PY tmp/realman_self_test.py "$ACT_CKPT" 50 ACT > tmp/logs/realman_act_self_test.log 2>&1 || ST_OK=0
$PY tmp/realman_self_test.py "$DP_CKPT" 8 DiffusionPolicy > tmp/logs/realman_dp_self_test.log 2>&1 || ST_OK=0
log "self-test pass=$ST_OK"

# 9. upload (best-effort, does not fail the chain)
UP="skipped"
if [ "$ST_OK" -eq 1 ]; then
  if $PY tmp/realman_upload_hf.py > tmp/logs/realman_hf_upload.log 2>&1; then UP="ok"; else UP="fail"; fi
  log "HF upload=$UP"
fi

# 10. summary + marker
cat > "$RUN/summary.json" <<JSON
{
  "status": "$([ "$ST_OK" -eq 1 ] && echo pass || echo fail)",
  "upload": "$UP",
  "act_ckpt": "$ACT_CKPT",
  "dp_ckpt": "$DP_CKPT",
  "ts": "$(date -u +%FT%TZ)"
}
JSON
mark "DONE realman self_test=$ST_OK upload=$UP $(date -u +%FT%TZ)"
log "chain done"
