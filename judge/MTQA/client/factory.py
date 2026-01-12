# eval_runner/clients/factory.py
# -*- coding: utf-8 -*-

from qwen3vl import Qwen3VLClient
from api2 import GEMINIClient

def build_client(kind: str, api_base: str = None, model: str = None):
    """
    kind: "qwen" or "gemini"
    """
    kind = (kind or "").lower().strip()
    if kind == "gemini":
        return GEMINIClient()

    # 默认 qwen
    return Qwen3VLClient(api_base=api_base, model=model)
