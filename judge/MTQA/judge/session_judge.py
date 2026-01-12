# eval_runner/judges/session_judge.py
# -*- coding: utf-8 -*-

import json
from json_utils import safe_parse_json

import json

async def judge_session_behavior(gpt, session_data, image_path):
    """
    Session-level evaluator for a 4-turn financial multi-turn dialogue session (T1–T4),
    scored on a 0–100 scale with explicit deduction items.

    Expected gpt.chat(...) returns a JSON string (or text containing JSON only).
    """
    session_str = json.dumps(session_data, ensure_ascii=False, indent=2)

    meta_prompt = f"""
You are a strict **Financial Multi-turn Image-Dialogue Evaluator**.

Your job is to evaluate ONE complete 4-turn session (T1–T4) at the SESSION level.
Do NOT answer the questions. Only judge compliance and quality.

Session to evaluate:
{session_str}

Design intent of the session:
- Exactly 4 turns (T1–T4)
- All turns revolve around the SAME image object (A / R1 / Z1 / S1 / C1 / P1)
- Cognitive progression must follow:
  T1 Localization → T2 Derivation → T3 Mild counterfactual intervention → T4 Observational verification
- At least ONE mild, computable intervention must appear (typically T3), WITHOUT changing the object definition
- All gold_answer values must be directly observable from the image or computable from image-derived values
  (NO external knowledge, speculation, or fabricated numbers)

Scoring rules (0–100):
- Start from 100 and subtract penalties according to issues found.
- If any fatal violation occurs (hallucination / image-ungrounded answers / object drift / missing intervention / invalid structure),
  the score may be reduced into the 0–30 range.

Deduction categories (cumulative):
1) Object Consistency Violations (-15 ~ -35)
   - Object naming changes across turns
   - Object definition changes (inconsistent minimal definition)
   - Questions drift away from the object defined in T1

2) Insufficient Minimal Re-definition (-5 ~ -15)
   - A turn fails to restate the minimal identifying definition (label/color/position/time/keyword)
   - A turn cannot be judged independently without relying on previous answers

3) Unreasonable Multi-turn Progression (-5 ~ -15)
   - Missing/unclear progression (Localization → Derivation → Intervention → Verification)
   - Redundant, circular, or jump-ahead turns

4) Invalid Counterfactual Intervention (-10 ~ -25)
   - No mild intervention in the entire session
   - Intervention is not computable or does not specify how to obtain the original value
   - Intervention effectively changes the object definition

5) Image-Ungrounded Answers / Hallucination (-25 ~ -70) [FATAL]
   - gold_answer cannot be read from the image or computed from image-visible values
   - External knowledge, assumptions, speculation, or fabricated values

6) Structural / Formatting Errors (-5 ~ -30)
   - Not exactly 4 turns, missing T1–T4, or incorrect turn_id
   - Invalid JSON / missing required fields (turn_id, question, gold_answer, vars_out)
   - vars_out missing or variables are unclear/unusable for programmatic evaluation

Output JSON ONLY (no extra text):
{{
  "Score": 0-100,
  "Pass": true/false,
  "Deductions": [
    {{
      "item": "Deduction category name",
      "detail": "Precise explanation (must indicate turn(s) and/or field(s) involved)",
      "penalty": -15
    }}
  ]
}}

Constraints:
- If there are no deductions, output "Deductions": []
- Score = 100 + sum(penalty), minimum 0
- Pass should be false if any fatal violation exists or if Score < 70; otherwise true.
""".strip()

    try:
        resp = gpt.chat(image=image_path, text=meta_prompt)
        return safe_parse_json(resp)
    except Exception as e:
        return {"Score": 0, "Pass": False, "Deductions": [{"item": "Runtime Error", "detail": str(e), "penalty": -100}]}

import json

async def judge_session_logic_reasoning(gpt, session_data, image_path):
    """
    Evaluator for the "4-turn logical reasoning" construction design:
    T1 Fact Extraction -> T2 Direct Reasoning -> T3 Counterfactual Adjustment -> T4 Comparative Validation

    Output: score out of 100 + explicit deduction items.
    """
    session_str = json.dumps(session_data, ensure_ascii=False, indent=2)

    meta_prompt = f"""
You are a strict **Financial Multi-turn Logical-Reasoning Session Evaluator**.

Your job is to evaluate ONE complete 4-turn session (T1–T4) at the SESSION level.
Do NOT answer the questions. Only judge compliance and quality.

Session to evaluate:
{session_str}

Design spec to enforce (must match the generator design):
- Exactly 4 turns: T1, T2, T3, T4.
- All turns must revolve around ONE uniquely-defined object (A/R1/Z1/S1/C1/P1).
- Logical flow must be:
  T1 Fact Extraction -> T2 One-step Calculation -> T3 Mild Counterfactual Adjustment -> T4 Comparative Validation.
- Each turn must restate the object's minimal identifying definition (label/color/position/time/keyword) so it is independently gradable.
- Turns may have implicit dependency, but MUST NOT contain explicit reference phrases such as:
  "previous turn", "last round", "as computed above", "you just", "recalculate", "上一轮", "上轮", "你刚才", "重新计算", "如前所述".
- All gold_answer values must be image-grounded:
  either directly readable from the image, or computable from image-visible numbers.
  No external knowledge, speculation, or fabricated values.

What to check (session-level):
A) Structure & fields:
   - Exactly 4 turns, turn_id must be T1/T2/T3/T4
   - Each turn has question, gold_answer, vars_out
   - vars_out variables are usable (value present; unit for numeric values when applicable)
B) Object consistency:
   - Same object type/name across turns (no drift)
   - Minimal definition is consistent, not redefined or swapped
C) Implicit dependency (no explicit phrasing):
   - Natural carry-over via semantics, but without explicit linking words/phrases
D) T2 correctness type:
   - Must be a clear, one-step calculation using T1 variables/object
E) T3 counterfactual validity:
   - Must introduce a mild, computable adjustment on the same object/metric
   - Must explain how to get the original value and how the adjustment is applied
   - Must NOT change object definition
F) T4 comparative validation:
   - Must perform a verification/comparison tied to the same object definition
G) Image-groundedness:
   - No hallucinated dates/labels/values; answers must be verifiable from the image or direct computation.

Scoring (0–100):
Start from 100 and subtract penalties. Penalties are cumulative.
If any fatal issue exists (hallucination/image-ungrounded, object drift, missing/invalid T3 adjustment, wrong turn count/ids),
final score may drop into 0–30 range.

Deduction guidance:
1) Structural/format errors (-10 ~ -40)
2) Object drift / inconsistent minimal definition (-15 ~ -40) [FATAL if clear drift]
3) Explicit reference phrases present (-5 ~ -20)
4) Weak/incorrect logical flow (T2 not a one-step calc, T4 not comparative) (-10 ~ -25)
5) Invalid counterfactual (missing / not computable / changes object) (-20 ~ -45) [FATAL if missing or changes object]
6) Image-ungrounded answers / hallucination (-30 ~ -90) [FATAL]

Output JSON ONLY (no extra text), in this format:
{{
  "Score": 0-100,
  "Pass": true/false,
  "Deductions": [
    {{
      "item": "Deduction category name",
      "detail": "Precise explanation (must indicate turn(s) and/or field(s) involved)",
      "penalty": -20
    }}
  ]
}}

Pass rule:
- Pass=false if any FATAL issue exists OR Score < 70; else Pass=true.
""".strip()

    try:
        resp = gpt.chat(image=image_path, text=meta_prompt)
        return safe_parse_json(resp)
    except Exception as e:
        return {
            "Score": 0,
            "Pass": False,
            "Deductions": [
                {"item": "Runtime Error", "detail": str(e), "penalty": -100}
            ],
        }


async def judge_session_v2(gpt, session_data, image_path):
    session_str = json.dumps(session_data, ensure_ascii=False, indent=2)

    meta_prompt = f"""
You are a **Strict Logic & Linguistics Auditor** for a financial dataset.
Evaluate a generated 4-turn dialogue session based on "Implicit Reasoning" rules.

Evaluation Target:
{session_str}

Output JSON only:
{{
  "Format_Score": 0-10,
  "Logic_Structure_Score": 0-10,
  "Implicit_Style_Score": 0-10,
  "Object_Consistency_Score": 0-10,
  "Overall_Score": 0-10,
  "Pass": true/false,
  "Critique": "..."
}}
""".strip()

    image_uri = image_path[0] if isinstance(image_path, list) else image_path
    try:
        resp = gpt.chat(image=image_uri, text=str(meta_prompt))
        return safe_parse_json(resp)
    except Exception as e:
        return {"Overall_Score": 0, "Pass": False, "Critique": f"Error: {e}"}


async def judge_session_multiview(gpt, session_data, image_paths):
    session_str = json.dumps(session_data, ensure_ascii=False, indent=2)
    num_pages = len(image_paths) if isinstance(image_paths, list) else 1
    doc_context = f"The document contains {num_pages} page(s)."

    meta_prompt = f"""
You are a **Senior Financial Document Auditor**.
Evaluate a 4-turn dialogue session generated from a multi-page financial report.

Evaluation Target:
{session_str}

Document Info:
{doc_context}

Output JSON only:
{{
  "Format_Citation_Score": 0-10,
  "Citation_Accuracy_Score": 0-10,
  "Progressive_Logic_Score": 0-10,
  "MultiView_Synthesis_Score": 0-10,
  "Overall_Score": 0-10,
  "Pass": true/false,
  "Critique": "..."
}}
""".strip()

    try:
        resp = gpt.chat(image=image_paths, text=meta_prompt)
        return safe_parse_json(resp)
    except Exception as e:
        return {"Overall_Score": 0, "Pass": False, "Critique": f"Error: {e}"}
