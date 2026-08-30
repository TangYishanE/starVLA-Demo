#!/usr/bin/env bash
# Unattended UnitreeG1_WholeBody reproduction chain:
#   train (1000 steps) -> locate checkpoint -> policy server -> local_self_test -> marker.
set -euo pipefail

ENV=/225010261/StarVLA/envs/starvla-cu128
REPO=/225010261/StarVLA/src/starVLA
RUN=$REPO/results/g1_wholebody_run

export PATH="$ENV/bin:/usr/local/cuda-12.5/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.5
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled

mkdir -p "$RUN"
cd "$REPO"
echo "=== CHAIN START $(date -u +%FT%TZ) ===" | tee "$RUN/summary.log"

# ---- Phase 1: training (1000 steps) ----
echo "[train] launch $(date -u +%T)" | tee -a "$RUN/summary.log"
BATCH=1 MAX_STEPS=1000 SAVE_EVERY=200 EVAL_EVERY=500 LOG_EVERY=10 \
  bash examples/realRobots/UnitreeG1_WholeBody/step2_training/train_files/run_starvla_qwenoft_g1_sonic_train.sh \
  > "$RUN/train.log" 2>&1
echo "[train] done $(date -u +%T)" | tee -a "$RUN/summary.log"

# ---- Phase 1.5: locate checkpoint ----
CKPTDIR=$REPO/results/Checkpoints/starvla_qwenoft_g1_sonic_smoke
CKPT=$(ls -t "$CKPTDIR"/checkpoints/steps_*.pt 2>/dev/null | head -1)
if [[ -z "$CKPT" ]]; then
  echo "[FAIL] no checkpoint under $CKPTDIR/checkpoints" | tee -a "$RUN/summary.log"
  echo "FAIL no_checkpoint $(date -u +%FT%TZ)" > "$RUN/COMPLETE"
  exit 1
fi
echo "[train] checkpoint: $CKPT" | tee -a "$RUN/summary.log"

# ---- Phase 2: policy server + self test ----
export PYTHONPATH="$REPO:$REPO/examples/realRobots/UnitreeG1_WholeBody/step3_deployment:${PYTHONPATH:-}"
PORT=5694
echo "[server] launch on :$PORT $(date -u +%T)" | tee -a "$RUN/summary.log"
python deployment/model_server/server_policy.py --ckpt_path "$CKPT" --port "$PORT" --use_bf16 \
  > "$RUN/server.log" 2>&1 &
SERVER_PID=$!

"$ENV/bin/python" - "$PORT" <<'PYEOF'
import socket, sys, time
port = int(sys.argv[1])
for i in range(180):
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=2)
        s.close()
        print(f"[server] port open after ~{i*2}s")
        break
    except Exception:
        time.sleep(2)
else:
    print("[server] TIMEOUT: port never opened")
    raise SystemExit(2)
PYEOF
echo "[server] ready $(date -u +%T)" | tee -a "$RUN/summary.log"

echo "[selftest] run $(date -u +%T)" | tee -a "$RUN/summary.log"
python examples/realRobots/UnitreeG1_WholeBody/step3_deployment/eval_files/local_self_test.py \
  --ckpt-path "$CKPT" --server-host 127.0.0.1 --server-port "$PORT" \
  > "$RUN/self_test.log" 2>&1
SELFTEST_EC=$?
echo "[selftest] exit=$SELFTEST_EC $(date -u +%T)" | tee -a "$RUN/summary.log"

kill "$SERVER_PID" 2>/dev/null || true

if [[ "$SELFTEST_EC" -eq 0 ]]; then
  echo "DONE selftest_pass $(date -u +%FT%TZ)" > "$RUN/COMPLETE"
else
  echo "FAIL selftest_exit_$SELFTEST_EC $(date -u +%FT%TZ)" > "$RUN/COMPLETE"
fi
echo "=== CHAIN END $(date -u +%FT%TZ) ===" | tee -a "$RUN/summary.log"
echo "chain finished; self_test exit=$SELFTEST_EC"
