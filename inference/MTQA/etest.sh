#!/usr/bin/env bash
set -euo pipefail

: "${FINMTM_OPEN_ENDED_INPUT_DIR:?Set FINMTM_OPEN_ENDED_INPUT_DIR}"

python -m inference.MTQA.inference \
  --input-dir "${FINMTM_OPEN_ENDED_INPUT_DIR}" \
  --output-dir "${FINMTM_OPEN_ENDED_OUTPUT_DIR:-outputs/open_ended}" \
  --include "${FINMTM_OPEN_ENDED_PATTERN:-*.jsonl}" \
  --backend "${VLM_BACKEND:-qwen3vl}" \
  --api-base "${VLM_API_BASE:-http://localhost:8000/v1}" \
  --model "${VLM_MODEL:-Qwen3-VL-30B-A3B-Instruct}" \
  --max-retries "${FINMTM_MAX_RETRIES:-2}" \
  --retry-sleep "${FINMTM_RETRY_SLEEP:-1.5}"
