#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from typing import Any, Dict

from tqdm import tqdm

from .utils import load_jsonl, safe_parse_json, normalize_image_path
from .prompts import build_agent_eval_prompt

async def eval_one_sample(judge_client: Any, sample: Dict[str, Any], logger) -> Dict[str, Any]:
    image_path = normalize_image_path(sample.get("image_path"))
    prompt = build_agent_eval_prompt(sample)

    resp_text = ""
    parsed_json = {}
    try:
        resp = await judge_client.image2text(
            instruction=prompt,
            image=image_path if image_path else None,
            temperature=0.0,
            pbar=None,
            output_file=None,
        )
        if isinstance(resp, dict):
            resp_text = json.dumps(resp, ensure_ascii=False)
        else:
            resp_text = str(resp)

        parsed_json = safe_parse_json(resp_text)
    except Exception as e:
        logger.error(f"❌ 评分调用失败: {e}")
        parsed_json = {}

    a_score = int(parsed_json.get("answer_score", 0))
    t_score = int(parsed_json.get("tool_score", 0))
    r_score = int(parsed_json.get("reasoning_score", 0))
    total = int(parsed_json.get("total_score", a_score + t_score + r_score))

    tool_metrics = parsed_json.get("tool_metrics", {}) or {}
    recall = tool_metrics.get("recall", 0.0)
    precision = tool_metrics.get("precision", 0.0)
    f1 = tool_metrics.get("f1", 0.0)
    emr = tool_metrics.get("emr", 0)

    return {
        "sample_id": sample.get("sample_id"),
        "question": sample.get("question2") or sample.get("question"),
        "scores": {"answer": a_score, "tool": t_score, "reasoning": r_score, "total": total},
        "metrics": {"tool_recall": recall, "tool_precision": precision, "tool_f1": f1, "tool_emr": emr},
        "basis": {
            "answer": parsed_json.get("answer_basis", ""),
            "tool": parsed_json.get("tool_basis", ""),
            "reasoning": parsed_json.get("reasoning_basis", ""),
        },
        "judge_raw_response": resp_text,
    }

async def run_evaluation_pipeline(input_file: str, output_file: str, judge_client: Any, logger):
    logger.info(f"⚖️  开始 Stage 2: 评分 (Input: {input_file})")

    samples = load_jsonl(input_file)
    if not samples:
        logger.error("❌ 待评分文件为空或不存在。")
        return

    results = []
    total_metrics = {
        "score_total": 0,
        "tool_recall": 0.0,
        "tool_precision": 0.0,
        "tool_f1": 0.0,
        "tool_emr": 0.0,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in tqdm(samples, desc="Evaluating"):
            res = await eval_one_sample(judge_client, sample, logger)
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()

            results.append(res)
            total_metrics["score_total"] += res["scores"]["total"]
            total_metrics["tool_recall"] += float(res["metrics"]["tool_recall"] or 0.0)
            total_metrics["tool_precision"] += float(res["metrics"]["tool_precision"] or 0.0)
            total_metrics["tool_f1"] += float(res["metrics"]["tool_f1"] or 0.0)
            total_metrics["tool_emr"] += float(res["metrics"]["tool_emr"] or 0.0)

    if results:
        count = len(results)
        logger.info(f"\n✅ 评测完成！共 {count} 条样本")
        logger.info("-" * 30)
        logger.info(f"📊 Avg Total Score:    {total_metrics['score_total'] / count:.2f}")
        logger.info(f"📈 Avg Tool Recall:    {total_metrics['tool_recall'] / count:.2%}")
        logger.info(f"🎯 Avg Tool Precision: {total_metrics['tool_precision'] / count:.2%}")
        logger.info(f"⚖️  Avg Tool F1:        {total_metrics['tool_f1'] / count:.2%}")
        logger.info(f"💎 Avg Tool EMR:       {total_metrics['tool_emr'] / count:.2%}")
        logger.info("-" * 30)
        logger.info(f"📄 详细结果已写入: {output_file}")
