# eval_runner/io_utils.py
# -*- coding: utf-8 -*-

import os
import json

def ensure_dir(path: str):
    """确保文件所在目录存在"""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def load_jsonl(path: str):
    samples = []
    with open(path, "rb") as f:
        for raw in f:
            try:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                samples.append(json.loads(line))
            except Exception as e:
                print(f"⚠️ 跳过损坏行: {e}")
    return samples
