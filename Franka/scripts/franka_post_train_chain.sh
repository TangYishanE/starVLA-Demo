#!/bin/bash
# Post-training chain: wait for smoke training -> P3 validation (policy server
# + synthetic obs) -> if PASSED, upload checkpoint to HF -> stop server.
set -uo pipefail
cd /225010261/StarVLA/src/starVLA
export PATH="/225010261/StarVLA/envs/starvla-cu128/bin:${PATH}"
export PYTHONPATH="${PWD}:${PWD}/starVLA:${PYTHONPATH:-}"
PY=/225010261/StarVLA/envs/starvla-cu128/bin/python

RUN_DIR=results/Checkpoints/franka_qwenoft_smoke
CKPT="${RUN_DIR}/checkpoints/steps_1000_pytorch_model.pt"
CHAIN_LOG=tmp/logs/franka_post_train_chain.log

echo "[$(date)] post-train chain started" >> "$CHAIN_LOG"

# 1. wait for training to finish
while [ ! -f "$CKPT" ]; do
  if ! pgrep -f "train_starvla.py.*franka_qwenoft_smoke" >/dev/null 2>&1; then
    echo "[$(date)] FATAL: training process exited WITHOUT steps_1000 checkpoint" >> "$CHAIN_LOG"
    exit 1
  fi
  sleep 30
done
echo "[$(date)] training complete: $CKPT" >> "$CHAIN_LOG"

# 2. start policy server
if curl -s --max-time 2 http://127.0.0.1:5694/health >/dev/null 2>&1; then
  echo "[$(date)] port 5694 already in use - aborting" >> "$CHAIN_LOG"
  exit 2
fi
CUDA_VISIBLE_DEVICES=0 nohup "$PY" deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" --port 5694 --use_bf16 \
  > tmp/logs/franka_policy_server.log 2>&1 &
SERVER_PID=$!
echo "[$(date)] policy server pid=$SERVER_PID" >> "$CHAIN_LOG"

# 3. wait for server ready
READY=0
for i in $(seq 1 60); do
  if grep -q "server running" tmp/logs/franka_policy_server.log 2>/dev/null; then
    READY=1; echo "[$(date)] server ready after ~$((i*5))s" >> "$CHAIN_LOG"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[$(date)] server process died; log tail:" >> "$CHAIN_LOG"
    tail -15 tmp/logs/franka_policy_server.log >> "$CHAIN_LOG"
    exit 3
  fi
  sleep 5
done
if [ "$READY" -ne 1 ]; then
  echo "[$(date)] server not ready in 300s" >> "$CHAIN_LOG"
  kill "$SERVER_PID" 2>/dev/null
  exit 3
fi
sleep 3

# 4. P3 validation
"$PY" tmp/franka_p3_validate.py > tmp/logs/franka_p3_validate.log 2>&1
P3_RC=$?
echo "[$(date)] P3 validation exit=$P3_RC" >> "$CHAIN_LOG"
tail -20 tmp/logs/franka_p3_validate.log >> "$CHAIN_LOG"

# 5. if passed -> upload to HF
if [ "$P3_RC" -eq 0 ]; then
  echo "[$(date)] P3 PASSED -> uploading checkpoint to HF" >> "$CHAIN_LOG"
  "$PY" tmp/franka_upload_hf.py > tmp/logs/franka_hf_upload.log 2>&1
  UP_RC=$?
  echo "[$(date)] HF upload exit=$UP_RC" >> "$CHAIN_LOG"
  tail -8 tmp/logs/franka_hf_upload.log >> "$CHAIN_LOG"
else
  echo "[$(date)] P3 FAILED -> SKIPPING HF upload" >> "$CHAIN_LOG"
fi

# 6. stop server
kill "$SERVER_PID" 2>/dev/null
sleep 2
echo "[$(date)] post-train chain done" >> "$CHAIN_LOG"
