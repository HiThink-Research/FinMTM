#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import httpx

# === MCP ===
MCP_SERVER_URL = "http://localhost:8081/sse"

# === 默认文件路径 ===
DEFAULT_INPUT_FILE = "/mnt/HithinkOmni/user_workspace/zhangchenxi4/omini/eval/agent.jsonl"
DEFAULT_INTERMEDIATE_FILE = "/mnt/HithinkOmni/user_workspace/zhangchenxi4/omini/eval/agent_val/out/gpt5/agent_result_wVision.jsonl"
DEFAULT_RESULT_FILE = "/mnt/HithinkOmni/user_workspace/zhangchenxi4/omini/eval/agent_val/out/gpt5/agent_eval_scores1.jsonl"

# === Agent 设置 ===
MAX_ITER = 8

# === SSE 超时（SSE read 不应超时）===
SSE_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=None,
    write=30.0,
    pool=30.0,
)
