#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
from typing import Any, Dict, List

def safe_parse_json(text: str) -> Dict[str, Any]:
    """
    尽可能鲁棒地从输出中抽取 JSON 对象：
    - 处理 ```json ... ```
    - 处理前后有解释性文本
    - 不做危险的全局引号替换
    """
    if not text:
        return {}

    text = str(text).strip()

    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        candidate = code_block.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            text = candidate

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1].strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    cleaned = text.replace("“", '"').replace("”", '"')
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1].strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return {}

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                continue
    return data

def safe_name(s: str) -> str:
    return (
        s.replace("/", "_")
         .replace(":", "_")
         .replace(" ", "_")
         .replace(".", "_")
         .replace("-", "_")
    )

def normalize_image_path(image_path):
    if isinstance(image_path, list):
        return image_path[0] if image_path else ""
    return image_path or ""


def unique_tool_calls(calls: Any) -> List[Any]:
    """Return exact duplicate-free calls while preserving their first occurrence."""

    unique: List[Any] = []
    seen = set()
    for call in calls or []:
        try:
            canonical = json.dumps(
                call,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except TypeError:
            canonical = str(call)
        if canonical in seen:
            continue
        seen.add(canonical)
        unique.append(call)
    return unique
