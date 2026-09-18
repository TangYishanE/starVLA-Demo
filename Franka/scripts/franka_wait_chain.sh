#!/bin/bash
# Wait for the UnitreeG1 training (PID 19925) to finish, then run the Franka
# forward gate, and (if the gate passes) launch the 1000-step smoke training.
cd /225010261/StarVLA/src/starVLA || exit 1
export PYTHONPATH="$PWD"
PY=/225010261/StarVLA/envs/starvla-cu128/bin/python
CHAIN_LOG=tmp/logs/franka_wait_chain.log

echo "[$(date)] chain started; waiting for G1 training (PID 19925)..." >> "$CHAIN_LOG"

# 1. wait for the G1 training process to exit
while kill -0 19925 2>/dev/null; do sleep 60; done
echo "[$(date)] G1 training finished" >> "$CHAIN_LOG"

# 2. forward gate
$PY starVLA/model/framework/VLM4A/QwenOFT.py \
  --config_yaml examples/realRobots/Franka/train_files/starvla_cotrain_franka_single.yaml \
  > tmp/logs/franka_forward_gate.log 2>&1
GATE_RC=$?
echo "[$(date)] forward gate exit=$GATE_RC" >> "$CHAIN_LOG"

# 3. training only if the gate passed
if [ "$GATE_RC" -eq 0 ]; then
  echo "[$(date)] forward gate PASSED -> launching 1000-step training" >> "$CHAIN_LOG"
  export CUDA_VISIBLE_DEVICES=0
  export WANDB_MODE=disabled
  nohup $PY -u starVLA/training/train_starvla.py \
    --config_yaml examples/realRobots/Franka/train_files/starvla_cotrain_franka_single.yaml \
    --datasets.vla_data.per_device_batch_size 1 \
    --trainer.max_train_steps 1000 \
    --trainer.save_interval 200 \
    --trainer.logging_frequency 10 \
    --trainer.eval_interval 100000 \
    --run_root_dir ./results/Checkpoints \
    --run_id franka_qwenoft_smoke \
    --wandb_project starVLA_franka \
    > tmp/logs/franka_train_1000step.log 2>&1 &
  echo "[$(date)] training launched (pid $!)" >> "$CHAIN_LOG"
else
  echo "[$(date)] forward gate FAILED (exit=$GATE_RC) -> SKIPPING training" >> "$CHAIN_LOG"
fi
echo "[$(date)] chain done" >> "$CHAIN_LOG"
