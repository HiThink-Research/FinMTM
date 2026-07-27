"""Task-specific session judges for FinMTM open-ended dialogues."""

from __future__ import annotations

import json
import math
from typing import Any

from ..json_utils import safe_parse_json


LEVEL_CHECKLISTS = {
    "L1": {
        "name": "Comprehension",
        "items": (
            "Entity_Recognition",
            "Spatial_Awareness",
        ),
    },
    "L2": {
        "name": "Calculation",
        "items": (
            "Multi_Step_Numerical_Calculation",
            "Chart_Numerical_Estimation",
        ),
    },
    "L3": {
        "name": "Self-correction",
        "items": (
            "Adversarial_Robustness",
            "Logical_Consistency",
        ),
    },
    "L4": {
        "name": "Memory",
        "items": (
            "Cross_Page_Entity_Linking",
            "Long_Context_Memory",
            "Multi_Source_Knowledge_Fusion",
        ),
    },
}


def _clamp_score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    if math.isnan(number):
        number = 0.0
    return min(max(number, 0.0), 10.0)


async def judge_session(
    client: Any,
    session_data: dict[str, Any],
    image_paths: Any,
    level: str,
) -> dict[str, Any]:
    """Judge a complete session with its paper-defined task checklist."""

    level = str(level).upper()
    if level not in LEVEL_CHECKLISTS:
        raise ValueError(f"unsupported task level: {level}")

    specification = LEVEL_CHECKLISTS[level]
    checklist_template = ",\n    ".join(
        f'"{item}": 0-10' for item in specification["items"]
    )
    session_str = json.dumps(session_data, ensure_ascii=False, indent=2)
    prompt = f"""
You are a senior financial multimodal evaluator.
Evaluate the complete dialogue at the SESSION level. Do not answer the task.

Task label: {level} ({specification["name"]})
Dialogue:
{session_str}

Score every checklist item from 0 to 10. Judge factual correctness, financial
validity, grounding in the supplied visual evidence, and consistency across
turns. For L4, citation checks assess whether claims are traceable to the
retrieved page/document; formatting alone must not override factual quality.

Return JSON only:
{{
  "Checklist_Scores": {{
    {checklist_template}
  }},
  "Overall_Score": 0-10,
  "Pass": true,
  "Critique": "brief evidence-based explanation"
}}

Overall_Score must be the arithmetic mean of the checklist scores.
""".strip()

    try:
        response = client.chat(image=image_paths, text=prompt)
        result = safe_parse_json(response)
    except Exception as exc:
        return {
            "Checklist_Scores": {},
            "Overall_Score": 0.0,
            "Pass": False,
            "Critique": f"Runtime error: {exc}",
            "judge_status": "error",
        }

    raw_scores = result.get("Checklist_Scores", {}) or {}
    missing_items = [
        item for item in specification["items"] if item not in raw_scores
    ]
    if missing_items:
        return {
            "Checklist_Scores": {},
            "Overall_Score": 0.0,
            "Pass": False,
            "Critique": (
                "Invalid judge response; missing checklist items: "
                + ", ".join(missing_items)
            ),
            "judge_status": "invalid_response",
        }
    checklist_scores = {
        item: _clamp_score(raw_scores.get(item, 0.0))
        for item in specification["items"]
    }
    overall = sum(checklist_scores.values()) / len(checklist_scores)
    return {
        "Checklist_Scores": checklist_scores,
        "Overall_Score": overall,
        "Pass": bool(result.get("Pass", overall >= 5.0)),
        "Critique": str(result.get("Critique", "")),
        "judge_status": "ok",
    }


async def judge_session_L1(client, session_data, image_paths):
    return await judge_session(client, session_data, image_paths, "L1")


async def judge_session_logic_reasoning(client, session_data, image_paths):
    return await judge_session(client, session_data, image_paths, "L2")


async def judge_session_behavior(client, session_data, image_paths):
    return await judge_session(client, session_data, image_paths, "L3")


async def judge_session_multiview(client, session_data, image_paths):
    return await judge_session(client, session_data, image_paths, "L4")


async def judge_session_v2(client, session_data, image_paths):
    """Backward-compatible alias for the L1 comprehension checklist."""

    return await judge_session_L1(client, session_data, image_paths)
