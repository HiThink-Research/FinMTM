#!/bin/bash
set -euo pipefail

MODEL_PATHS=(
  "/cpfs01/NLP/models/Qwen3-VL-30B-A3B-Instruct"
)

PY_SCRIPT="val.py"

BASE_PORT=8080
NUM_GPUS=1
MAX_MODEL_LEN=60720
HOST="0.0.0.0"
TARGET_GPUS="7"

WAIT_LISTEN_TIMEOUT=120   # 等端口监听
WAIT_READY_TIMEOUT=900    # 等模型真正ready（按需调大）
COOLDOWN_TIME=20
FAIL_FAST=0

log() { echo "[$(date '+%F %T')] $*"; }

port_listening() {
  local port="$1"
  lsof -i :"${port}" -sTCP:LISTEN -t >/dev/null 2>&1
}

wait_listen() {
  local port="$1" timeout="$2"
  local t=0
  while ! port_listening "${port}"; do
    sleep 1; t=$((t+1))
    if [ "${t}" -ge "${timeout}" ]; then return 1; fi
  done
  return 0
}

# 用 /v1/models 确认服务可用且模型名出现
wait_ready() {
  local port="$1" model_name="$2" timeout="$3"
  local t=0
  while true; do
    if curl -s "http://127.0.0.1:${port}/v1/models" | grep -q "\"${model_name}\""; then
      return 0
    fi
    sleep 2; t=$((t+2))
    if [ "${t}" -ge "${timeout}" ]; then return 1; fi
  done
}

stop_by_pid() {
  local pid="$1"
  if [ -z "${pid}" ] || ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi
  # 杀进程组（更干净），避免残留 worker
  log "Stopping process group for PID=${pid}"
  kill -TERM "-${pid}" 2>/dev/null || true
  sleep 3
  kill -KILL "-${pid}" 2>/dev/null || true
}

stop_by_port_best_effort() {
  local port="$1"
  local pid
  pid="$(lsof -i :"${port}" -sTCP:LISTEN -t 2>/dev/null || true)"
  if [ -n "${pid}" ]; then
    log "Best-effort stop: port ${port} PID=${pid}"
    kill -TERM "${pid}" 2>/dev/null || true
    sleep 2
    kill -KILL "${pid}" 2>/dev/null || true
  fi
}

idx=0
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  log "=================================================="
  MODEL_NAME="$(basename "${MODEL_PATH}")"
  PORT=$((BASE_PORT + idx))
  LOG_FILE="vllm_${MODEL_NAME}_${PORT}.log"

  export CUDA_VISIBLE_DEVICES="${TARGET_GPUS}"
  export MODEL_NAME="${MODEL_NAME}"
  export CURRENT_VLM_MODEL="${MODEL_NAME}"
  export CURRENT_VLM_PORT="${PORT}"

  # 启动前清理端口残留
  stop_by_port_best_effort "${PORT}"

  log "Model: ${MODEL_NAME}"
  log "Path : ${MODEL_PATH}"
  log "Port : ${PORT}"
  log "GPUs : ${CUDA_VISIBLE_DEVICES}"
  log "Log  : ${LOG_FILE}"

  log "Starting vLLM server..."
  # 用 setsid 让它成为一个新的进程组，便于后面 kill -PGID
  setsid nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --trust-remote-code \
    --served-model-name "${MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${NUM_GPUS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    > "${LOG_FILE}" 2>&1 &

  VLLM_PID=$!
  log "vLLM PID=${VLLM_PID}"

  if ! wait_listen "${PORT}" "${WAIT_LISTEN_TIMEOUT}"; then
    log "ERROR: vLLM did not listen on port ${PORT} in ${WAIT_LISTEN_TIMEOUT}s"
    tail -n 80 "${LOG_FILE}" || true
    stop_by_pid "${VLLM_PID}"
    exit 1
  fi
  log "Port ${PORT} is listening."

  if ! wait_ready "${PORT}" "${MODEL_NAME}" "${WAIT_READY_TIMEOUT}"; then
    log "ERROR: vLLM not ready in ${WAIT_READY_TIMEOUT}s (model=${MODEL_NAME})"
    tail -n 120 "${LOG_FILE}" || true
    stop_by_pid "${VLLM_PID}"
    exit 1
  fi
  log "vLLM is READY."

  log "Running evaluation script: ${PY_SCRIPT}"
  set +e
  python3 "${PY_SCRIPT}"
  EXIT_CODE=$?
  set -e

  log "Stopping vLLM..."
  stop_by_pid "${VLLM_PID}"

  if [ "${EXIT_CODE}" -ne 0 ]; then
    log "WARN: Evaluation failed (exit=${EXIT_CODE}). See ${LOG_FILE}"
    if [ "${FAIL_FAST}" -eq 1 ]; then
      exit "${EXIT_CODE}"
    fi
  else
    log "OK: Evaluation finished successfully"
  fi

  log "Cooldown: ${COOLDOWN_TIME}s"
  sleep "${COOLDOWN_TIME}"
  idx=$((idx + 1))
done

log "All models finished."
