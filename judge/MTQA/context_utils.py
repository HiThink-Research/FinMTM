# eval_runner/context_utils.py
# -*- coding: utf-8 -*-

def build_context(turns, upto_idx: int):
    """拼接对话上下文到当前轮（取 upto_idx 之前的轮次）"""
    ctx_lines = []
    for i in range(upto_idx):
        q = turns[i].get("question", "")
        a = turns[i].get("model_answer", "")
        ctx_lines.append(f"T{i+1}问：{q}\nT{i+1}答：{a}")
    return "\n".join(ctx_lines)
