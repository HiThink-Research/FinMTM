# eval_runner/config.py
# -*- coding: utf-8 -*-

DEFAULT_DIRS = [
    "/cpfs01/HithinkOmniSSD/user_workspace/ganziliang/code/omini/generate_qes_ans/1_cleaned/outputs_qwen3_vl_235B_A32B",
    "/cpfs01/HithinkOmniSSD/user_workspace/ganziliang/code/omini/generate_qes_ans/1_cleaned/outputs_qwen2.57b",
    "/cpfs01/HithinkOmniSSD/user_workspace/ganziliang/code/omini/generate_qes_ans/1_cleaned/Qwen3-VL-32B-Instruct",
    "/cpfs01/HithinkOmniSSD/user_workspace/ganziliang/code/omini/generate_qes_ans/1_cleaned/Qwen3-VL-32B-Thinking",
    "/cpfs01/HithinkOmniSSD/user_workspace/ganziliang/code/omini/generate_qes_ans/1_cleaned/Qwen3-VL-235B-A22B-Thinking",
    "/cpfs01/HithinkOmniSSD/user_workspace/ganziliang/code/omini/generate_qes_ans/1_cleaned/qwen3vl4B_IS",
]

DEFAULT_MODEL = "Qwen3-VL-235B-Instruct"
DEFAULT_API_BASE = "http://localhost:30000/v1"
DEFAULT_OUT_SUBDIR = "val_new"
DEFAULT_PATTERN = "L*_with_id_vlm.jsonl"

# 融合权重（可按你需求调）
WEIGHT_TURN = 0.5
WEIGHT_SESSION = 0.5

# 分支惩罚阈值
CITATION_FAIL_TH = 6
ROBUSTNESS_FAIL_TH = 5
PENALTY_MULT = 0.3
