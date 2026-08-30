#!/bin/bash
# Wait for the Franka training (run_id=franka_qwenoft_smoke) to finish, then
# auto-run the G1 WholeBody dry-run step 2/3 against the G1 policy server:
#   mock_g1_controller.py (step 3) + replay_lerobot_episode.py (step 2).
# Mirrors tmp/franka_wait_chain.sh (the established wait-chain pattern).

cd /225010261/StarVLA/src/starVLA || exit 1
export PATH="/225010261/StarVLA/envs/starvla-cu128/bin:/usr/local/cuda-12.5/bin:${PATH}"
export CUDA_HOME=/usr/local/cuda-12.5
export PYTHONPATH="$PWD:$PWD/examples/realRobots/UnitreeG1_WholeBody/step3_deployment:${PYTHONPATH:-}"
PY=/225010261/StarVLA/envs/starvla-cu128/bin/python
RUN=$PWD/results/g1_wholebody_dryrun
CHAIN_LOG=tmp/logs/g1_dryrun_wait_chain.log

mkdir -p "$RUN"
echo "[$(date)] g1 dry-run chain started; waiting for franka training..." >> "$CHAIN_LOG"

# 1. wait for the Franka training process (accelerate + train_starvla, run_id=franka_qwenoft_smoke) to exit
while pgrep -f "franka_qwenoft_smoke" > /dev/null 2>&1; do sleep 60; done
echo "[$(date)] franka training finished" >> "$CHAIN_LOG"
sleep 15   # brief grace for GPU release / checkpoint finalize

# 2. locate the G1 checkpoint
CKPT=$PWD/results/Checkpoints/starvla_qwenoft_g1_sonic_smoke/checkpoints/steps_1000_pytorch_model.pt
if [[ ! -f "$CKPT" ]]; then
  echo "[$(date)] FAIL: G1 checkpoint missing: $CKPT" >> "$CHAIN_LOG"
  echo "FAIL g1_ckpt_missing $(date -u +%FT%TZ)" > "$RUN/COMPLETE"
  exit 1
fi

# 3. start the G1 policy server
PORT=5694
echo "[$(date)] starting G1 policy server on :$PORT" >> "$CHAIN_LOG"
CUDA_VISIBLE_DEVICES=0 python deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" --port "$PORT" --use_bf16 \
  > "$RUN/server.log" 2>&1 &
SERVER_PID=$!

# wait for the server to bind the port (model loaded before bind)
"$PY" - "$PORT" <<'PYEOF'
import socket, sys, time
port = int(sys.argv[1])
for i in range(180):
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=2); s.close()
        print(f"[server] port open after ~{i*2}s"); break
    except Exception:
        time.sleep(2)
else:
    print("[server] TIMEOUT: port never opened"); raise SystemExit(2)
PYEOF
echo "[$(date)] server ready" >> "$CHAIN_LOG"

# 4. dry-run step 3: mock controller (inspect 64/7/7 action groups)
echo "[$(date)] running mock_g1_controller" >> "$CHAIN_LOG"
python examples/realRobots/UnitreeG1_WholeBody/step3_deployment/eval_files/mock_g1_controller.py \
  --ckpt-path "$CKPT" --server-host 127.0.0.1 --server-port "$PORT" \
  > "$RUN/mock_controller.log" 2>&1
MOCK_EC=$?
echo "[$(date)] mock exit=$MOCK_EC" >> "$CHAIN_LOG"

# 5. dry-run step 2: replay a real recorded episode
echo "[$(date)] running replay_lerobot_episode" >> "$CHAIN_LOG"
python examples/realRobots/UnitreeG1_WholeBody/step3_deployment/eval_files/replay_lerobot_episode.py \
  --dataset-path playground/Datasets/UnitreeG1_WholeBody/lerobot/test_sonic \
  --ckpt-path "$CKPT" --server-host 127.0.0.1 --server-port "$PORT" \
  --episode 0 --stride 8 --max-frames 30 \
  > "$RUN/replay_episode.log" 2>&1
REPLAY_EC=$?
echo "[$(date)] replay exit=$REPLAY_EC" >> "$CHAIN_LOG"

# 6. teardown + completion marker
kill "$SERVER_PID" 2>/dev/null || true
if [[ "$MOCK_EC" -eq 0 && "$REPLAY_EC" -eq 0 ]]; then
  echo "DONE dryrun_pass $(date -u +%FT%TZ)" > "$RUN/COMPLETE"
else
  echo "FAIL mock=$MOCK_EC replay=$REPLAY_EC $(date -u +%FT%TZ)" > "$RUN/COMPLETE"
fi
echo "[$(date)] g1 dry-run chain done (mock=$MOCK_EC replay=$REPLAY_EC)" >> "$CHAIN_LOG"
