#!/usr/bin/env bash
set -euo pipefail

# Example models - replace with your local HF model ids or paths
OPTIMIZE_MODEL="/path/to/local/opt"
EVALUATE_MODEL="/path/to/local/eval"
EXECUTE_MODEL="/path/to/local/exec"

OPTIMIZE_PORT="${OPTIMIZE_PORT:-8000}"
EVALUATE_PORT="${EVALUATE_PORT:-8001}"
EXECUTE_PORT="${EXECUTE_PORT:-8002}"

# Optional: served alias names (short names you will use in client 'model' field)
OPTIMIZE_NAME="${OPTIMIZE_NAME:-opt-llm}"
EVALUATE_NAME="${EVALUATE_NAME:-judge-llm}"
EXECUTE_NAME="${EXECUTE_NAME:-exec-llm}"

echo "[*] Launching vLLM servers..."
echo "    Optimize: ${OPTIMIZE_MODEL} on :${OPTIMIZE_PORT} as ${OPTIMIZE_NAME}"
echo "    Evaluate: ${EVALUATE_MODEL} on :${EVALUATE_PORT} as ${EVALUATE_NAME}"
echo "    Execute : ${EXECUTE_MODEL} on :${EXECUTE_PORT} as ${EXECUTE_NAME}"

# GPU 0 Opt
CUDA_VISIBLE_DEVICES=0 nohup vllm serve "${OPTIMIZE_MODEL}" \
  --host 0.0.0.0 --port "${OPTIMIZE_PORT}" \
  --served-model-name "${OPTIMIZE_NAME}" \
  --gpu-memory-utilization 0.90 \
  > "opt_${OPTIMIZE_PORT}.log" 2>&1 &

# GPU 1 Eval
CUDA_VISIBLE_DEVICES=1 nohup vllm serve "${EVALUATE_MODEL}" \
  --host 0.0.0.0 --port "${EVALUATE_PORT}" \
  --served-model-name "${EVALUATE_NAME}" \
  --gpu-memory-utilization 0.90 \
  > "eval_${EVALUATE_PORT}.log" 2>&1 &

# GPU 2 Exe
CUDA_VISIBLE_DEVICES=2 nohup vllm serve "${EXECUTE_MODEL}" \
  --host 0.0.0.0 --port "${EXECUTE_PORT}" \
  --served-model-name "${EXECUTE_NAME}" \
  --gpu-memory-utilization 0.90 \
  > "exec_${EXECUTE_PORT}.log" 2>&1 &

echo "[*] Launched. Use 'tail -f *_*.log' to monitor."

# pkill -9 -f "vllm serve"
