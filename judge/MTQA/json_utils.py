# eval_runner/json_utils.py
# -*- coding: utf-8 -*-

import json

def safe_parse_json(text):
    """鲁棒 JSON 解析：支持 dict/markdown code fence/中英文引号修复"""
    if isinstance(text, dict):
        return text
    if text is None:
        return {}

    try:
        cleaned = str(text).strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned)
    except Exception:
        fixed = str(text).replace("“", '"').replace("”", '"').replace("'", '"')
        try:
            return json.loads(fixed)
        except Exception:
            return {}
