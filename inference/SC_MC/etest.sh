#!/usr/bin/env bash
set -euo pipefail

: "${FINMTM_OBJECTIVE_INPUT:?Set FINMTM_OBJECTIVE_INPUT to the input JSONL path}"

ARGS=(
  --input "${FINMTM_OBJECTIVE_INPUT}"
  --output "${FINMTM_OBJECTIVE_OUTPUT:-outputs/objective_eval_results.jsonl}"
  --summary "${FINMTM_OBJECTIVE_SUMMARY:-outputs/objective_eval_summary.json}"
  --api-base "${VLM_API_BASE:-http://localhost:8000/v1}"
  --model "${VLM_MODEL:-Qwen3-VL-30B-A3B-Instruct}"
)
if [[ -n "${FINMTM_IMAGE_ROOT:-}" ]]; then
  ARGS+=(--image-root "${FINMTM_IMAGE_ROOT}")
fi

python -m inference.SC_MC.etest "${ARGS[@]}"
