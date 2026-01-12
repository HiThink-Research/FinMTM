# eval_runner/evaluator.py
# -*- coding: utf-8 -*-

import json
import traceback
from tqdm import tqdm

from io_utils import load_jsonl, ensure_dir
from judge.turn_judge import judge_financial_turn
from judge.session_judge import (
    judge_session_behavior,
    judge_session_v2,
    judge_session_multiview,
    judge_session_logic_reasoning,
    judge_session_L1
)
import config


def detect_multiview(task_type_hint: str, image_paths):
    # 不要把“单页”当 multiview；multiview 以多图/研报/Multi 为准
    return (
        (len(image_paths) > 1)
        or ("研报" in (task_type_hint or ""))
        or ("Multi" in (task_type_hint or ""))
    )


async def evaluate_sample(sample, eval_client,task):
    turns = sample.get("turns", [])
    image_paths = sample.get("image_paths") or sample.get("image_path")
    image_paths = image_paths if isinstance(image_paths, list) else [image_paths]

    # --- 1) Turn-Level ---
    turn_results = []
    for idx, turn in enumerate(turns):
        res = await judge_financial_turn(eval_client, turn, turns, idx, image_paths)
        turn_results.append(res)

    avg_turn_score = (
        sum(x["score"] for x in turn_results) / len(turn_results)
        if turn_results else 0.0
    )

    # --- 2) Session-Level (branch) ---
    task_type_hint = str(sample.get("task_type", "")) + str(turns[0].get("task_type", "") if turns else "")
    num_turns = len(turns)
    is_multiview = detect_multiview(task_type_hint, image_paths)

    session_eval_res = {}
    session_score = 0.0
    final_score = None

    if is_multiview:
        session_eval_res = await judge_session_multiview(eval_client, sample, image_paths)
        session_score = float(session_eval_res.get("Overall_Score", 0) or 0)
        citation_score = float(session_eval_res.get("Format_Citation_Score", 0) or 0)

        if citation_score < config.CITATION_FAIL_TH:
            final_score = avg_turn_score * config.PENALTY_MULT
        else:
            final_score = (avg_turn_score * 0.4) + (session_score * 0.6)

    elif num_turns == 5:
        session_eval_res = await judge_session_behavior(eval_client, sample, image_paths[0] if image_paths else None)
        session_score = float(session_eval_res.get("Overall_Score", 0) or 0)
        robustness = float(session_eval_res.get("T3_Robustness_Score", 0) or 0)

        if robustness < config.ROBUSTNESS_FAIL_TH:
            final_score = avg_turn_score * config.PENALTY_MULT
        else:
            final_score = (avg_turn_score * 0.4) + (session_score * 0.6)

    elif task=="L1":
        session_eval_res = await judge_session_L1(eval_client, sample, image_paths)
        session_score = float(session_eval_res.get("Overall_Score", 0) or 0)
        final_score = (avg_turn_score * 0.5) + (session_score * 0.5)
    elif task=="L2":
        session_eval_res = await judge_session_logic_reasoning(eval_client, sample, image_paths)
        session_score = float(session_eval_res.get("Overall_Score", 0) or 0)
        final_score = (avg_turn_score * 0.5) + (session_score * 0.5)
        
    # --- 3) fallback（理论上不会走到） ---
    if final_score is None:
        final_score = (avg_turn_score * config.WEIGHT_TURN) + (session_score * config.WEIGHT_SESSION)

    return {
        "image_path": sample.get("image_path"),
        "final_composite_score": round(float(final_score or 0), 2),
        "avg_turn_score": round(float(avg_turn_score or 0), 2),
        "session_structure_score": float(session_score or 0),
        "is_pass": bool(session_eval_res.get("Pass", False)),
        "session_critique": session_eval_res.get("Critique", ""),
        "turn_details": turn_results,
        "session_details": session_eval_res,
    }

import os
import re

def level_from_input_path(input_path: str) -> str:
    """
    从输入文件名提取 L1-L4。
    例如：.../L1_with_id_vlm.jsonl -> L1
         .../abc_L2__with_id_vlm.jsonl -> L2
    """
    fname = os.path.basename(input_path).upper()
    m = re.search(r"\bL[1-4]\b", fname)
    if not m:
        raise ValueError(f"无法从文件名解析 L1-L4: {fname}")
    return m.group(0)

async def run_file(input_path, output_path, eval_client):
    samples = load_jsonl(input_path)
    task=level_from_input_path(input_path)
    print(f"📂 Load {len(samples)} samples from {input_path}")
    ensure_dir(output_path)

    with open(output_path, "a", encoding="utf-8") as fout:
        for sample in tqdm(samples, desc=f"🚀 Evaluating {input_path}"):
            try:
                result = await evaluate_sample(sample, eval_client,task)
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
            except Exception as e:
                print(f"⚠️ Sample failed: {e}")
                traceback.print_exc()

    print(f"✅ Done. Saved to {output_path}")
