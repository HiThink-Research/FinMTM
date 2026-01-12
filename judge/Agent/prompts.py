#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from typing import Any, Dict

def agent_sys_prompt(query: str) -> str:
    # 你原来 sys prompt 的内容（我保持原意）
    return f"""
你是一名**金融多轮分析 Agent**，需要逐步规划 MCP 工具调用，并结合图像中的信息完成任务。
问题是：{query}
你先找到这个公司是哪家，
严禁使用你自己的工具查阅信息，下面是我给的工具，你给我说ActionTrace，我会自己执行。
---
### 可使用的工具
1. **FinQuery**: 金融数据查询 (股价、涨跌幅、财务数据等)。
2. **Search**: 通用搜索 (新闻、概念)。
3. **StockNews**: 财经新闻检索。
4. **ReportQuery**: 研报观点查询。

---
### 输出格式要求,tool 不要有 若上条无结果，每次查询的内容不要太复杂
请仅输出一个 JSON 对象，包含以下 4 个字段：
{{
  "Thought": "你对任务的总体思考、推理步骤。",
  "VisualObservation": ["你从图像中观察或识别到的金融关键信息"],
  "ActionTrace": [
    {{"tool": "FinQuery", "query": "查询xxx"}},
    {{"tool": "Search", "query": "查询xxx"}}
  ]
}}

若任务完成，请输出 <FINISHED> 并总结最终结论。
""".strip()

def build_agent_eval_prompt(sample: Dict[str, Any]) -> str:
    """
    你的新版 Judge Prompt（含 Tool F1 / EMR / 0-50 Answer / 0-25 Reasoning）
    """
    question = sample.get("question", "")
    model_answer = sample.get("model_final_answer", "")
    ref_answer = sample.get("reference_gold_answer", "")

    model_visual = sample.get("model_visual_observation", "")
    ref_visual = sample.get("reference_visual_observation", "")

    model_tools = sample.get("model_tool_calls", [])
    ref_tools = sample.get("reference_tool_calls", [])

    model_thought = sample.get("model_thought", "")
    ref_thought = sample.get("reference_thought", "")

    meta_prompt = f"""
你是一名专业的金融 Agent 评测专家。你的任务是对模型的行为进行严格量化打分。

你会得到以下信息：
- 题目（Question）
- 模型与标准的视觉观察（Visual Observation）
- 模型与标准的工具调用轨迹（Tool Calls）
- 模型与标准的思考过程（Thought）
- 最终回答（Final Answer）

你需要输出详细的指标分数，并计算总分（满分 100 分）。

============================
一、答案正确性评分（answer_score，满分 50）
============================
对比 model_final_answer 与 reference_gold_answer：
1. 核心判定：实体识别正确（如公司名） 且 数值/结论准确（容许一定格式差异，如 12.45% vs 0.1245）。
2. 评分规则：
   - 若所有关键要点完全匹配：50 分
   - 否则：0 分

============================
二、工具调用评分（tool_score，满分 25）
============================
请根据以下定义计算指标，基于 Tool F1 计算得分。

【定义】：
- Reference Set (R): 标准答案中的工具调用集合。
- Predicted Set (P): 模型输出的工具调用集合。
- Correct Tool (Hit): 模型调用的工具，其 tool_name 和 核心参数(query/args) 与标准答案语义一致。

【指标计算】：
1. Tool Recall (TR) = |P ∩ R| / |R|
2. Tool Precision (TP) = |P ∩ R| / |P|
3. Tool F1 (TF1) = (2 * TR * TP) / (TR + TP)，若 TR+TP=0，则 TF1=0
4. Exact Match Rate (EMR) (0/1)：工具调用序列组织与参考序列完全一致则为 1 否则 0

【最终工具得分】：
- tool_score = round(TF1 * 25)

注意：参数匹配允许语义等价（例如 "2024Q4" 与 "2024年第四季度"）。

============================
三、推理连贯性评分（reasoning_score，满分 25）
============================
先给出 raw_reason ∈ [0, 1]，再计算 reasoning_score = round(raw_reason * 25)。

扣分项：
- 幻觉/记忆错误：-0.2 ~ -0.3
- 逻辑断层/跳步：-0.1
- 与工具结果矛盾：-0.2

============================
四、输出格式
============================
请直接输出如下 JSON 格式，不要包含Markdown代码块标记：

{{
  "answer_score": 0 或 50,
  "tool_metrics": {{
      "recall": 0.0~1.0,
      "precision": 0.0~1.0,
      "f1": 0.0~1.0,
      "emr": 0 或 1
  }},
  "tool_score": 0~25,
  "reasoning_score": 0~25,
  "total_score": 0~100,
  "answer_basis": "简短说明答案判定理由",
  "tool_basis": "简短说明工具F1及EMR判定理由（指出哪个工具多余或缺失）",
  "reasoning_basis": "简短说明推理打分理由"
}}

============================
待评测样本：
============================

【题目】: {question}

【模型最终回答】: {model_answer}
【标准参考答案】: {ref_answer}

【模型工具调用 (Predicted Set)】:
{json.dumps(model_tools, ensure_ascii=False, indent=2)}

【标准工具调用 (Reference Set)】:
{json.dumps(ref_tools, ensure_ascii=False, indent=2)}

【模型视觉与思考】:
Visual: {model_visual}
Thought: {model_thought}

【标准视觉与思考】:
Visual: {ref_visual}
Thought: {ref_thought}
"""
    return meta_prompt.strip()
