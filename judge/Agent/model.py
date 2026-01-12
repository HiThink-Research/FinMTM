#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, List

from api2 import GPT4OClient, GEMINIClient, Grok
from qwen3vl import Qwen3VLClient

def build_models() -> List[Dict[str, Any]]:
    """
    按顺序测哪些模型，在这里写。
    key: 用于目录名/筛选；obj: client 实例；display: 日志显示
    """
    return [
        {"key": "grok", "obj": Grok(), "display": "Grok"},
        {"key": "o3", "obj": GPT4OClient(model="o3"), "display": "GPT4OClient(o3)"},
        {"key": "gemini_3_flash_preview", "obj": GEMINIClient(model="gemini-3-flash-preview"), "display": "Gemini 3 Flash Preview"},
        {"key": "gemini", "obj": GEMINIClient(), "display": "Gemini"},
        {"key": "gpt5", "obj": GPT4OClient(model="gpt-5"), "display": "GPT-5"},
        {"key": "qwen3vl_30b_a3b", "obj": Qwen3VLClient(api_base="http://localhost:8000/v1", model="Qwen3-VL-30B-A3B-Instruct"), "display": "Qwen3-VL-30B-A3B-Instruct"},
    ]
