#!/usr/bin/env python3
"""Paper-aligned financial-agent judge aggregation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal installs
    def tqdm(iterable, **_kwargs):
        return iterable

from finmtm_eval.metrics import agent_score, f_beta_score

from .prompts import build_agent_eval_prompt
from .utils import (
    load_jsonl,
    normalize_image_path,
    safe_parse_json,
    unique_tool_calls,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else number


def _unique_call_count(calls: Any) -> int:
    """Count exact duplicate-free invocations, consistent with set notation."""

    return len(unique_tool_calls(calls))


def _validated_match_pairs(
    raw_pairs: Any,
    predicted_count: int,
    reference_count: int,
) -> list[dict[str, Any]]:
    """Validate judge-reported one-to-one semantic matches by set index."""

    validated = []
    used_predicted = set()
    used_reference = set()
    for pair in raw_pairs or []:
        if not isinstance(pair, dict):
            continue
        try:
            predicted_index = int(pair.get("predicted_index"))
            reference_index = int(pair.get("reference_index"))
        except (TypeError, ValueError):
            continue
        if not 0 <= predicted_index < predicted_count:
            continue
        if not 0 <= reference_index < reference_count:
            continue
        if predicted_index in used_predicted or reference_index in used_reference:
            continue
        used_predicted.add(predicted_index)
        used_reference.add(reference_index)
        validated.append(
            {
                "predicted_index": predicted_index,
                "reference_index": reference_index,
                "basis": str(pair.get("basis", "")),
            }
        )
    return validated


def resolve_tool_metrics(
    parsed_json: dict[str, Any],
    sample: dict[str, Any],
) -> dict[str, float | int]:
    """Resolve semantic-match counts and deterministically calculate P/R/F2/EMR."""

    metrics = parsed_json.get("tool_metrics", {}) or {}
    predicted_count = _unique_call_count(sample.get("model_tool_calls", []))
    reference_count = _unique_call_count(sample.get("reference_tool_calls", []))
    raw_match_pairs = metrics.get("matched_pairs")
    raw_true_positives = metrics.get("true_positives")

    if isinstance(raw_match_pairs, list):
        matched_pairs = _validated_match_pairs(
            raw_match_pairs,
            predicted_count,
            reference_count,
        )
        true_positives = len(matched_pairs)
        precision = (
            true_positives / predicted_count if predicted_count else 0.0
        )
        recall = true_positives / reference_count if reference_count else 0.0
    elif raw_true_positives is not None:
        matched_pairs = []
        true_positives = int(
            min(
                max(_safe_float(raw_true_positives), 0.0),
                float(min(predicted_count, reference_count)),
            )
        )
        precision = (
            true_positives / predicted_count if predicted_count else 0.0
        )
        recall = true_positives / reference_count if reference_count else 0.0
    else:
        # Compatibility with judge outputs generated before TP counts were added.
        matched_pairs = []
        true_positives = -1
        precision = min(max(_safe_float(metrics.get("precision")), 0.0), 1.0)
        recall = min(max(_safe_float(metrics.get("recall")), 0.0), 1.0)

    if true_positives >= 0:
        emr = int(
            predicted_count == reference_count
            and true_positives == predicted_count
        )
    else:
        emr = int(
            _safe_float(metrics.get("emr"), default=-1.0) == 1.0
            or (precision == 1.0 and recall == 1.0)
        )

    return {
        "true_positives": true_positives,
        "matched_pairs": matched_pairs,
        "predicted_count": predicted_count,
        "reference_count": reference_count,
        "precision": precision,
        "recall": recall,
        "f2": f_beta_score(precision, recall, beta=2.0),
        "emr": emr,
    }


async def eval_one_sample(
    judge_client: Any,
    sample: dict[str, Any],
    logger: Any,
) -> dict[str, Any]:
    image_path = normalize_image_path(sample.get("image_path"))
    prompt = build_agent_eval_prompt(sample)

    response_text = ""
    judge_status = "ok"
    judge_error = ""
    try:
        response = await judge_client.image2text(
            instruction=prompt,
            image=image_path if image_path else None,
            temperature=0.0,
            pbar=None,
            output_file=None,
        )
        response_text = (
            json.dumps(response, ensure_ascii=False)
            if isinstance(response, dict)
            else str(response)
        )
        parsed_json = safe_parse_json(response_text)
        required_fields = {"answer_score", "reasoning_score", "tool_metrics"}
        if not required_fields.issubset(parsed_json):
            judge_status = "invalid_response"
            judge_error = "judge response is missing required scoring fields"
    except Exception as exc:
        logger.error(f"Judge call failed: {exc}")
        parsed_json = {}
        judge_status = "error"
        judge_error = str(exc)

    metrics = resolve_tool_metrics(parsed_json, sample)
    scores = agent_score(
        parsed_json.get("answer_score", 0.0),
        parsed_json.get("reasoning_score", 0.0),
        metrics["precision"],
        metrics["recall"],
    )

    return {
        "sample_id": sample.get("sample_id"),
        "question": sample.get("question2") or sample.get("question"),
        "evaluation_status": judge_status,
        "evaluation_error": judge_error,
        "scores": {key: round(value, 4) for key, value in scores.items()},
        "metrics": {
            "tool_true_positives": metrics["true_positives"],
            "tool_matched_pairs": metrics["matched_pairs"],
            "tool_predicted_count": metrics["predicted_count"],
            "tool_reference_count": metrics["reference_count"],
            "tool_precision": round(float(metrics["precision"]), 6),
            "tool_recall": round(float(metrics["recall"]), 6),
            "tool_f2": round(float(metrics["f2"]), 6),
            "tool_emr": int(metrics["emr"]),
        },
        "basis": {
            "answer": parsed_json.get("answer_basis", ""),
            "tool": parsed_json.get("tool_basis", ""),
            "reasoning": parsed_json.get("reasoning_basis", ""),
        },
        "judge_raw_response": response_text,
    }


async def run_evaluation_pipeline(
    input_file: str,
    output_file: str,
    judge_client: Any,
    logger: Any,
) -> None:
    logger.info(f"Starting evaluation: {input_file}")
    samples = load_jsonl(input_file)
    if not samples:
        logger.error("Evaluation input is empty or missing.")
        return

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    score_total = 0.0
    precision_total = 0.0
    recall_total = 0.0
    f2_total = 0.0
    emr_total = 0.0
    valid_count = 0
    failed_count = 0
    with open(output_file, "w", encoding="utf-8") as output:
        for sample in tqdm(samples, desc="Evaluating"):
            result = await eval_one_sample(
                judge_client,
                sample,
                logger,
            )
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            if result["evaluation_status"] != "ok":
                failed_count += 1
                continue
            valid_count += 1
            score_total += result["scores"]["total"]
            precision_total += result["metrics"]["tool_precision"]
            recall_total += result["metrics"]["tool_recall"]
            f2_total += result["metrics"]["tool_f2"]
            emr_total += result["metrics"]["tool_emr"]

    logger.info(
        f"Evaluation complete: {len(samples)} samples "
        f"({valid_count} valid, {failed_count} failed)"
    )
    if not valid_count:
        logger.error("No valid judge outputs; aggregate scores were not computed.")
        return
    logger.info(f"Average total score: {score_total / valid_count:.2f}")
    logger.info(f"Average tool precision: {precision_total / valid_count:.2%}")
    logger.info(f"Average tool recall: {recall_total / valid_count:.2%}")
    logger.info(f"Average tool F2: {f2_total / valid_count:.2%}")
    logger.info(f"Average tool EMR: {emr_total / valid_count:.2%}")
