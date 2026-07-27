#!/usr/bin/env python3
"""Inference and judge prompts for the paper-aligned Agent protocol."""

from __future__ import annotations

import json
from typing import Any


def agent_sys_prompt(query: str) -> str:
    return f"""
你是一名金融多轮分析 Agent，需要逐步规划 MCP 工具调用，并结合图像信息完成任务。
问题：{query}

只能使用以下固定 MCP 工具；系统会执行 ActionTrace 并返回结果。

可用工具：
1. FinQuery：市场价格、估值指标、交易统计和基本面数据。
2. StockNews：实时股票新闻检索。
3. AnalysisLib：结构化金融分析。
4. NoticeSearch：公司公告、披露和备案文件检索。
5. VisitWeb：解析给定网页 URL 的正文。

每轮仅输出一个 JSON 对象：
{{
  "Thought": "高层推理与下一步规划",
  "VisualObservation": ["从图像中获得的证据"],
  "ActionTrace": [
    {{"tool": "FinQuery", "query": "包含公司、指标和时间范围的查询"}}
  ]
}}

不要重复已经成功执行的调用。允许采用任意有效规划顺序。
若任务完成，请输出 <FINISHED> 并总结最终结论。
""".strip()


def build_agent_eval_prompt(sample: dict[str, Any]) -> str:
    """Build the paper-aligned financial-agent judge prompt."""

    question = sample.get("question", "")
    model_answer = sample.get("model_final_answer", "")
    reference_answer = sample.get("reference_gold_answer", "")
    model_visual = sample.get("model_visual_observation", "")
    reference_visual = sample.get("reference_visual_observation", "")
    model_tools = sample.get("model_tool_calls", [])
    reference_tools = sample.get("reference_tool_calls", [])
    model_thought = sample.get("model_thought", "")
    reference_thought = sample.get("reference_thought", "")

    return f"""
你是一名专业的金融 Agent 评测专家。请依据图像证据、实际工具返回信息和标准答案，
对模型的答案、工具规划和推理进行评分。只输出 JSON。

一、答案正确性 answer_score（0～50）
这是分级评分，不是 0/50 二元评分。检查核心实体、数值、方向性结论和任务要求。
语义等价的格式不得扣分，例如 12.45% 与 0.1245、公司名与明确股票代码、等价日期
写法、单位换算和合理舍入。

评分锚点：
- 45～50：核心答案完整正确，仅有不影响结论的轻微遗漏；
- 35～44：主要结论正确，但存在次要遗漏或轻微数值偏差；
- 20～34：部分关键点正确，但答案不完整或含实质性错误；
- 1～19：仅有少量相关信息，核心结论错误或缺乏证据；
- 0：完全错误、无关或没有作答。

二、工具调用匹配
参考轨迹是有效、效率导向的信息需求基线，不是唯一推理路径。你只需进行一对一语义
匹配并报告 true_positives；评分代码将按论文公式计算 F2×25（β=2），并将 EMR
作为诊断指标。

匹配规则：
1. 将轨迹视为无序集合，不比较调用顺序。
2. 同一个参考调用不能被重复计为多个 TP。
3. 工具功能及核心参数（公司/股票、日期范围、指标或查询条件）必须与任务信息需求
   语义一致。
4. 功能等价、满足同一信息需求的调用可以匹配，不要求工具名字符串完全一致。
5. 参数允许语义等价，例如“2024Q4”与“2024年第四季度”。
6. predicted_count 和 reference_count 必须分别等于给定集合的大小。

三、推理质量 reasoning_score（0～25）
评价推理是否连贯、是否由图像与实际工具返回结果支持，以及是否存在幻觉、逻辑断层
或与工具证据矛盾。不得仅因推理顺序与参考轨迹不同而扣分。

输出格式：
{{
  "answer_score": 0～50,
  "tool_metrics": {{
    "true_positives": 0,
    "predicted_count": 0,
    "reference_count": 0,
    "precision": 0.0,
    "recall": 0.0,
    "f2": 0.0,
    "emr": 0
  }},
  "reasoning_score": 0～25,
  "answer_basis": "答案评分依据",
  "tool_basis": "语义匹配、多余及缺失的信息需求",
  "reasoning_basis": "推理评分依据"
}}

题目：
{question}

模型最终回答：
{model_answer}

标准参考答案：
{reference_answer}

模型工具调用（Predicted Set）：
{json.dumps(model_tools, ensure_ascii=False, indent=2)}

参考工具调用（Reference Set）：
{json.dumps(reference_tools, ensure_ascii=False, indent=2)}

模型视觉观察与推理：
Visual: {model_visual}
Thought: {model_thought}

参考视觉观察与推理：
Visual: {reference_visual}
Thought: {reference_thought}
""".strip()
