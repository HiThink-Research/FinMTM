"""Paper-aligned open-ended dialogue evaluator."""

from __future__ import annotations

import json
import os
import re
import traceback
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal installs
    def tqdm(iterable, **_kwargs):
        return iterable

from finmtm_eval.metrics import CAPABILITIES, dialogue_score

from .io_utils import ensure_dir, load_jsonl
from .judge.session_judge import judge_session
from .judge.turn_judge import judge_financial_turn


LEVEL_ALIASES = {
    "L1": "L1",
    "COM": "L1",
    "COMPREHENSION": "L1",
    "L2": "L2",
    "CAL": "L2",
    "CALCULATION": "L2",
    "L3": "L3",
    "SELFCORR": "L3",
    "SELF-CORRECTION": "L3",
    "SELF_CORRECTION": "L3",
    "L4": "L4",
    "MEM": "L4",
    "MEMORY": "L4",
}


def _normalise_images(sample: dict[str, Any]) -> list[str]:
    image_paths = sample.get("image_paths") or sample.get("image_path") or []
    if not isinstance(image_paths, list):
        image_paths = [image_paths]
    return [path for path in image_paths if path]


def level_from_sample(sample: dict[str, Any], fallback: str | None = None) -> str:
    """Read the session-level task label, falling back to the file label."""

    candidates = [
        sample.get("level"),
        sample.get("task_type"),
        sample.get("task"),
        fallback,
    ]
    for candidate in candidates:
        key = str(candidate or "").strip().upper()
        if key in LEVEL_ALIASES:
            return LEVEL_ALIASES[key]
        match = re.search(r"(?:^|[^A-Z0-9])(L[1-4])(?:[^A-Z0-9]|$)", key)
        if match:
            return match.group(1)
    raise ValueError("sample is missing a valid L1-L4 session label")


async def evaluate_sample(
    sample: dict[str, Any],
    eval_client: Any,
    task: str | None = None,
) -> dict[str, Any]:
    turns = sample.get("turns", [])
    if not isinstance(turns, list) or not turns:
        raise ValueError("open-ended sample must contain at least one turn")
    image_paths = _normalise_images(sample)
    level = level_from_sample(sample, fallback=task)

    turn_results = []
    for index, turn in enumerate(turns):
        result = await judge_financial_turn(
            eval_client,
            turn,
            turns,
            index,
            image_paths,
        )
        turn_results.append(result)

    turn_score = (
        sum(item["score"] for item in turn_results) / len(turn_results)
        if turn_results
        else 0.0
    )
    session_result = await judge_session(
        eval_client,
        sample,
        image_paths,
        level,
    )
    session_score = float(session_result.get("Overall_Score", 0.0) or 0.0)

    capability_means = {}
    for capability in CAPABILITIES:
        values = [
            item.get("capability_scores", {}).get(capability, 0.0)
            for item in turn_results
        ]
        capability_means[capability] = (
            sum(values) / len(values) if values else 0.0
        )

    final_score = dialogue_score(
        turn_score,
        session_score,
        alpha=0.5,
        report_scale=100.0,
    )
    judge_statuses = [
        item.get("judge_status", "ok") for item in turn_results
    ] + [session_result.get("judge_status", "ok")]
    evaluation_status = (
        "ok" if all(status == "ok" for status in judge_statuses) else "error"
    )
    return {
        "sample_id": sample.get("sample_id") or sample.get("session_id"),
        "image_path": sample.get("image_path"),
        "task_level": level,
        "score_scale": "0-100",
        "final_composite_score": round(final_score, 2),
        "avg_turn_score_0_10": round(turn_score, 4),
        "session_score_0_10": round(session_score, 4),
        "capability_scores_0_100": {
            name: round(value * 10.0, 2)
            for name, value in capability_means.items()
        },
        "is_pass": bool(session_result.get("Pass", False)),
        "session_critique": session_result.get("Critique", ""),
        "turn_details": turn_results,
        "session_details": session_result,
        "evaluation_status": evaluation_status,
    }


def level_from_input_path(input_path: str) -> str:
    filename = os.path.basename(input_path).upper()
    match = re.search(r"(?:^|[^A-Z0-9])(L[1-4])(?:[^A-Z0-9]|$)", filename)
    if not match:
        raise ValueError(f"cannot infer L1-L4 from filename: {filename}")
    return match.group(1)


async def run_file(input_path: str, output_path: str, eval_client: Any):
    samples = load_jsonl(input_path)
    fallback_level = level_from_input_path(input_path)
    print(f"Load {len(samples)} samples from {input_path}")
    ensure_dir(output_path)

    with open(output_path, "w", encoding="utf-8") as output:
        for sample in tqdm(samples, desc=f"Evaluating {input_path}"):
            try:
                result = await evaluate_sample(
                    sample,
                    eval_client,
                    fallback_level,
                )
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
            except Exception as exc:
                print(f"Sample failed: {exc}")
                traceback.print_exc()
                failure = {
                    "sample_id": sample.get("sample_id")
                    or sample.get("session_id"),
                    "evaluation_status": "error",
                    "error": str(exc),
                }
                output.write(json.dumps(failure, ensure_ascii=False) + "\n")
                output.flush()

    print(f"Done. Saved to {output_path}")
