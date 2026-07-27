# eval_runner/judges/turn_judge.py
# -*- coding: utf-8 -*-

import traceback

from finmtm_eval.metrics import CAPABILITIES, clamp, turn_capability_score

from ..context_utils import build_context
from ..json_utils import safe_parse_json

async def judge_financial_turn(client, turn, all_turns, idx, image_paths):
    """
    Turn-Level：严苛金融逐轮评分
    """
    q = turn.get("question", "")
    gold = turn.get("gold_answer", "")
    pred = turn.get("model_answer", "")
    ctx_text = build_context(all_turns, idx)

    meta_prompt = f"""
You are a **Senior Financial Auditor & Quantitative Analyst**.
Your task is to audit an AI agent's performance in analyzing financial documents (Candlestick charts, Balance Sheets, Research Reports, Pie/Bar charts).

You apply **Institutional-Grade Standards**. In finance, a decimal error or a hallucinated trend results in millions of dollars in losses.
**Be extremely harsh. Do not forgive "minor" numerical errors.**

---
### 1. Financial Scoring Metrics (1-10 Scale)

**A. Visual Precision & OCR (视觉精度与数据提取)**
- 1-3: wrong row/col; confuses O/C or H/L; misses units
- 4-6: gets number but misses context
- 7-8: accurate reads and correct color code understanding
- 9-10: pixel-perfect tiny labels and complex interactions

**B. Financial Logic & Calculation (金融逻辑与计算)**
- 1-3: wrong formula; wrong trend description
- 4-6: correct calc but shallow interpretation
- 7-8: computes derived metrics correctly
- 9-10: multi-step synthesis like earnings quality

**C. Data Accuracy (数据准确性)**
- 1-4: fatal: any digit/date wrong
- 5-6: lazy approximation
- 7-8: exact match
- 9-10: precise formatted output

**D. Cross-Modal Verification (多模态互证)**
- 1-3: hallucinates alignment
- 7-8: aligns table/chart with text correctly
- 9-10: finds discrepancies

**E. Temporal Awareness (时序敏感度)**
- 1: time blind
- 5: vague timeline
- 10: pinpoint timing

---
### 2. Evaluation Context
User Query: {q}
Model Answer: {pred}
Ground Truth: {gold}
Conversation History: {ctx_text}

---
### 3. JSON Output
{{
  "Visual_Precision": 1-10,
  "Financial_Logic": 1-10,
  "Data_Accuracy": 1-10,
  "Cross_Modal_Verification": 1-10,
  "Temporal_Awareness": 1-10,
  "Comment": "Specific critique."
}}
""".strip()

    image_uri = image_paths if isinstance(image_paths, list) else [image_paths]

    try:
        resp = client.chat(image=image_uri, text=str(meta_prompt))
        resp_json = safe_parse_json(resp)
        aliases = {
            "visual_precision": "Visual_Precision",
            "financial_logic": "Financial_Logic",
            "data_accuracy": "Data_Accuracy",
            "cross_modal_verification": "Cross_Modal_Verification",
            "temporal_awareness": "Temporal_Awareness",
        }
        capability_scores = {
            name: resp_json.get(field, 0.0) for name, field in aliases.items()
        }
        score = turn_capability_score(capability_scores)
    except Exception as e:
        traceback.print_exc()
        return {"score": 0.0, "comment": f"Error: {e}", "details": {}}

    return {
        "turn_id": turn.get("turn_id", f"T{idx+1}"),
        "question": q,
        "score": score,
        "comment": resp_json.get("Comment", ""),
        "capability_scores": {
            name: clamp(capability_scores[name], 0.0, 10.0)
            for name in CAPABILITIES
        },
        "details": resp_json,
    }
