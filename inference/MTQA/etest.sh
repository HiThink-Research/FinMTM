#!/usr/bin/env bash
set -Eeuo pipefail

###########################################
#               参数配置
###########################################

# 模型路径数组（本地权重目录或hf路径）
MODEL_PATHS=(
  "/model/qwen3vl4B_IS"
  "/model/qwen3vl8B_IS"
  "/model/qwen3vl4B_TK"
  "/model/qwen3vl8B_TK"
)

# 可选：为特定路径自定义 served-model-name（默认取 basename）
# 例如： declare -A NAME_MAP=( ["/model/qwen3vl4B_IS"]="Qwen3-VL-4B-Instruct" )
declare -A NAME_MAP=()

# 评测 Python 脚本（与你的 vLLM 一起用）
PYTHON_SCRIPT="./inference.py"

# vLLM 服务参数
VLLM_HOST="0.0.0.0"
BASE_PORT=8000            # 每个模型自动端口递增
NUM_GPUS=1                # vLLM tensor-parallel-size
MAX_MODEL_LEN=30720

# 等待与超时参数
START_TIMEOUT=240         # 最多等待 vLLM 启动就绪秒数（健康检查）
REST_BETWEEN_MODELS=20    # 每个模型处理完后的休眠
HEALTH_RETRY_INTERVAL=3   # 健康检查轮询间隔

# 日志目录
LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"

###########################################
#               工具函数
###########################################

log() { echo "[$(date +'%F %T')] $*"; }

# 健康检查：轮询 /v1/models 直到成功或超时
wait_ready() {
  local port="$1"
  local timeout="${2:-$START_TIMEOUT}"
  local deadline=$(( $(date +%s) + timeout ))
  local url="http://127.0.0.1:${port}/v1/models"

  log "⏳ 等待 vLLM 在端口 ${port} 就绪（超时 ${timeout}s）..."
  until curl -fsS "${url}" >/dev/null 2>&1; do
    if (( $(date +%s) > deadline )); then
      log "❌ vLLM 健康检查超时：${url}"
      return 1
    fi
    sleep "${HEALTH_RETRY_INTERVAL}"
  done
  log "✅ vLLM 已就绪：${url}"
}

# 启动 vLLM，返回全局变量 VLLM_PID
start_vllm() {
  local model_path="$1"
  local served_name="$2"
  local host="$3"
  local port="$4"
  local num_gpus="$5"
  local max_len="$6"
  local log_file="$7"

  log "🚀 启动 vLLM：${served_name} @ ${model_path} (port=${port})"
  # 使用 setsid 使其成为新进程组，便于整组杀掉
  setsid python3 -m vllm.entrypoints.openai.api_server \
    --model "${model_path}" \
    --trust-remote-code \
    --served-model-name "${served_name}" \
    --host "${host}" \
    --port "${port}" \
    --tensor-parallel-size "${num_gpus}" \
    --max-model-len "${max_len}" \
    > "${log_file}" 2>&1 &

  VLLM_PID=$!
  log "📝 vLLM 日志：${log_file}  (PID=${VLLM_PID})"
}

# 停止 vLLM（优雅 → 强制）
stop_vllm() {
  local port="$1"
  local pid="${2:-}"

  # 优先按 PID 杀
  if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
    log "🔻 优雅停止 vLLM (PID=${pid})"
    # 整个进程组
    kill -TERM "-${pid}" >/dev/null 2>&1 || true
    sleep 2
    if ps -p "${pid}" >/dev/null 2>&1; then
      log "🪓 强制停止 vLLM (PID=${pid})"
      kill -KILL "-${pid}" >/dev/null 2>&1 || true
    fi
    return 0
  fi

  # 退而求其次：按端口查进程
  if command -v lsof >/dev/null 2>&1; then
    local main_pid
    main_pid=$(lsof -i :"${port}" -t || true)
    if [[ -n "${main_pid}" ]]; then
      log "🔻 端口 ${port} 进程 PID=${main_pid}，尝试优雅停止"
      kill -TERM "${main_pid}" >/dev/null 2>&1 || true
      sleep 2
      if ps -p "${main_pid}" >/dev/null 2>&1; then
        log "🪓 强制停止 PID=${main_pid}"
        kill -KILL "${main_pid}" >/dev/null 2>&1 || true
      fi
    else
      log "⚠️ 未找到监听端口 ${port} 的 vLLM 进程，可能已退出"
    fi
  else
    log "⚠️ lsof 不可用，跳过按端口清理。"
  fi
}

# 运行你的 Python 评测任务（与 vLLM 一起）
run_task() {
  local model_path="$1"
  local served_name="$2"
  local port="$3"

  export CURRENT_VLM_MODEL="${model_path}"
  export CURRENT_VLM_PORT="${port}"
  export VLLM_API_BASE="http://127.0.0.1:${port}/v1"
  export VLLM_SERVED_MODEL="${served_name}"

  log "🧠 执行 Python 任务脚本：${PYTHON_SCRIPT}"
  log "   环境：CURRENT_VLM_MODEL=${CURRENT_VLM_MODEL}"
  log "        CURRENT_VLM_PORT=${CURRENT_VLM_PORT}"
  log "        VLLM_API_BASE=${VLLM_API_BASE}"
  log "        VLLM_SERVED_MODEL=${VLLM_SERVED_MODEL}"

  python3 "${PYTHON_SCRIPT}"
}

###########################################
#               主循环
###########################################

idx=0
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  echo
  echo "=================================================="

  # 端口与名称
  VLLM_PORT=$((BASE_PORT + idx))
  DEFAULT_NAME="$(basename "${MODEL_PATH}")"
  SERVED_MODEL_NAME="${NAME_MAP[${MODEL_PATH}]:-${DEFAULT_NAME}}"

  LOG_FILE="${LOG_DIR}/vllm_${SERVED_MODEL_NAME}_${VLLM_PORT}.log"

  log "📂 模型路径：${MODEL_PATH}"
  log "🏷 运行名   ：${SERVED_MODEL_NAME}"
  log "🔗 服务端口：${VLLM_PORT}"

  VLLM_PID=""

  # 1) 启动 vLLM
  start_vllm "${MODEL_PATH}" "${SERVED_MODEL_NAME}" "${VLLM_HOST}" "${VLLM_PORT}" "${NUM_GPUS}" "${MAX_MODEL_LEN}" "${LOG_FILE}"

  # 2) 健康检查
  if ! wait_ready "${VLLM_PORT}" "${START_TIMEOUT}"; then
    log "❌ 启动失败，打印日志尾部（最后 200 行）："
    tail -n 200 "${LOG_FILE}" || true
    stop_vllm "${VLLM_PORT}" "${VLLM_PID}"
    exit 1
  fi

  # 3) 运行 Python 任务
  set +e
  run_task "${MODEL_PATH}" "${SERVED_MODEL_NAME}" "${VLLM_PORT}"
  SCRIPT_EXIT_CODE=$?
  set -e

  # 4) 关闭 vLLM
  stop_vllm "${VLLM_PORT}" "${VLLM_PID}"

  # 5) 检查任务状态 & 休眠
  if [[ ${SCRIPT_EXIT_CODE} -ne 0 ]]; then
    log "⚠️ Python 脚本执行失败（退出码 ${SCRIPT_EXIT_CODE}）"
  else
    log "✅ Python 脚本执行成功"
  fi

  log "🧹 模型 ${SERVED_MODEL_NAME} 已处理完成。休息 ${REST_BETWEEN_MODELS}s..."
  sleep "${REST_BETWEEN_MODELS}"

  idx=$((idx + 1))
done

echo
log "🎉 所有模型已经处理完毕！"
