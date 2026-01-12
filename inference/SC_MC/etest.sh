#!/bin/bash
set -euo pipefail

###########################################
#               参数配置
###########################################

MODEL_PATHS=(
  "/cpfs01/NLP/models/Qwen3-VL-30B-A3B-Instruct"
)

PY_SCRIPT="val.py"

BASE_PORT=8080
NUM_GPUS=1
MAX_MODEL_LEN=60720
WAIT_TIME=400                 # 等待模型加载时间（秒）
COOLDOWN_TIME=20              # 每个模型完成后的冷却时间（秒）
HOST="0.0.0.0"
TARGET_GPUS="7"

###########################################
#               工具函数
###########################################

log() {
  echo "[$(date '+%F %T')] $*"
}

wait_port_ready() {
  local port="$1"
  local timeout="${2:-120}"   # 最多等 120 秒端口监听起来
  local t=0
  while ! lsof -i :"${port}" -sTCP:LISTEN -t >/dev/null 2>&1; do
    sleep 1
    t=$((t + 1))
    if [ "${t}" -ge "${timeout}" ]; then
      return 1
    fi
  done
  return 0
}

stop_vllm_by_port() {
  local port="$1"

  local main_pid
  main_pid="$(lsof -i :"${port}" -sTCP:LISTEN -t 2>/dev/null || true)"

  if [ -z "${main_pid}" ]; then
    log "WARN: No process is listening on port ${port}"
    return 0
  fi

  log "Stopping vLLM process on port ${port} (PID=${main_pid})"
  kill -9 "${main_pid}" >/dev/null 2>&1 || true

  # 清理子进程（如果还有）
  local child_pids
  child_pids="$(pgrep -P "${main_pid}" 2>/dev/null || true)"
  if [ -n "${child_pids}" ]; then
    log "Stopping worker processes: ${child_pids}"
    kill -9 ${child_pids} >/dev/null 2>&1 || true
  fi
}

###########################################
#               主循环
###########################################

idx=0
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  log "=================================================="

  MODEL_NAME="$(basename "${MODEL_PATH}")"
  PORT=$((BASE_PORT + idx))
  LOG_FILE="vllm_${MODEL_NAME}_${PORT}.log"

  log "Model: ${MODEL_NAME}"
  log "Path : ${MODEL_PATH}"
  log "Port : ${PORT}"
  log "Log  : ${LOG_FILE}"

  export CUDA_VISIBLE_DEVICES="${TARGET_GPUS}"
  export MODEL_NAME="${MODEL_NAME}"
  export CURRENT_VLM_MODEL="${MODEL_NAME}"
  export CURRENT_VLM_PORT="${PORT}"

  ###########################################
  # 1) 启动 vLLM 服务
  ###########################################
  log "Starting vLLM server..."
  nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --trust-remote-code \
    --served-model-name "${MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${NUM_GPUS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    > "${LOG_FILE}" 2>&1 &

  # 给进程一点点启动时间
  sleep 5

  log "Waiting for model warmup: ${WAIT_TIME}s"
  sleep "${WAIT_TIME}"

  # 可选：确认端口已监听（更稳健）
  if wait_port_ready "${PORT}" 60; then
    log "vLLM is listening on port ${PORT}"
  else
    log "ERROR: vLLM did not start listening on port ${PORT} within timeout"
    log "Check log: ${LOG_FILE}"
    stop_vllm_by_port "${PORT}"
    exit 1
  fi

  ###########################################
  # 2) 执行 Python 评测任务
  ###########################################
  log "Running evaluation script: ${PY_SCRIPT}"
  set +e
  python3 "${PY_SCRIPT}"
  EXIT_CODE=$?
  set -e

  ###########################################
  # 3) Kill vLLM 服务
  ###########################################
  log "Stopping vLLM server on port ${PORT}"
  stop_vllm_by_port "${PORT}"

  ###########################################
  # 4) 打印执行状态 & 等待
  ###########################################
  if [ "${EXIT_CODE}" -ne 0 ]; then
    log "WARN: Evaluation script failed (exit code=${EXIT_CODE})"
  else
    log "OK: Evaluation script finished successfully"
  fi

  log "Cooldown: ${COOLDOWN_TIME}s"
  sleep "${COOLDOWN_TIME}"

  idx=$((idx + 1))
done

log "All models finished."
