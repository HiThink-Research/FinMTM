"""Default command-line settings for open-ended evaluation."""

DEFAULT_MODEL = "Qwen3-VL-235B-A22B-Instruct"
DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_OUT_SUBDIR = "scores"
DEFAULT_PATTERN = "L*_with_id_vlm.jsonl"

# Manuscript Equation (4).
WEIGHT_TURN = 0.5
WEIGHT_SESSION = 0.5
