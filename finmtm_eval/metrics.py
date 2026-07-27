"""Paper-aligned deterministic metrics for FinMTM.

The formulas in this module implement Equations (1), (3), (4), (5), and (6)
from the submitted manuscript. LLM judges may identify semantic matches or
produce rubric scores, but all arithmetic aggregation is performed here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


CAPABILITIES = (
    "visual_precision",
    "financial_logic",
    "data_accuracy",
    "cross_modal_verification",
    "temporal_awareness",
)


def clamp(value: Any, lower: float, upper: float) -> float:
    """Convert *value* to float and clamp it to an inclusive interval."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = lower
    return min(max(number, lower), upper)


def _normalise_option(option: Any) -> str:
    return str(option).strip().upper()


def objective_set_score(
    predicted: Iterable[Any],
    ground_truth: Iterable[Any],
) -> float:
    """Score an objective item using manuscript Equation (1).

    Any over-selected option makes the score zero. Otherwise the score is the
    fraction of ground-truth options selected. Single-choice is the special
    case in which the ground-truth set contains one option.
    """

    predicted_set = {_normalise_option(item) for item in predicted if str(item).strip()}
    ground_truth_set = {
        _normalise_option(item) for item in ground_truth if str(item).strip()
    }
    if not ground_truth_set:
        raise ValueError("ground_truth must contain at least one option")
    if predicted_set - ground_truth_set:
        return 0.0
    return len(predicted_set & ground_truth_set) / len(ground_truth_set)


def f_beta_score(precision: Any, recall: Any, beta: float = 2.0) -> float:
    """Return F-beta for precision and recall in [0, 1]."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    p = clamp(precision, 0.0, 1.0)
    r = clamp(recall, 0.0, 1.0)
    beta_squared = beta * beta
    denominator = beta_squared * p + r
    if denominator == 0:
        return 0.0
    return (1.0 + beta_squared) * p * r / denominator


def tool_score(
    precision: Any,
    recall: Any,
    *,
    beta: float = 2.0,
    weight: float = 25.0,
) -> float:
    """Return the paper's recall-oriented planning score Qt (Equation 5)."""

    return clamp(weight, 0.0, float("inf")) * f_beta_score(
        precision,
        recall,
        beta=beta,
    )


def turn_capability_score(
    capability_scores: Mapping[str, Any] | Iterable[Any],
) -> float:
    """Uniformly average the five 0-10 capability scores (Equations 2-3)."""

    if isinstance(capability_scores, Mapping):
        values = [capability_scores.get(name, 0.0) for name in CAPABILITIES]
    else:
        values = list(capability_scores)
        if len(values) != len(CAPABILITIES):
            raise ValueError(f"expected {len(CAPABILITIES)} capability scores")
    return sum(clamp(value, 0.0, 10.0) for value in values) / len(CAPABILITIES)


def dialogue_score(
    turn_score: Any,
    session_score: Any,
    *,
    alpha: float = 0.5,
    report_scale: float = 100.0,
) -> float:
    """Combine 0-10 turn/session scores and convert to the reporting scale.

    Equation (4) fixes alpha=0.5. The manuscript tables report 0-100 values,
    so the default output multiplies the internal 0-10 score by ten.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    turn = clamp(turn_score, 0.0, 10.0)
    session = clamp(session_score, 0.0, 10.0)
    internal = alpha * turn + (1.0 - alpha) * session
    return internal * (report_scale / 10.0)


def agent_score(
    answer_score: Any,
    reasoning_score: Any,
    precision: Any,
    recall: Any,
) -> dict[str, float]:
    """Return Qa, Qr, Qt, and Qfinal using Equations (5-6)."""

    answer = clamp(answer_score, 0.0, 50.0)
    reasoning = clamp(reasoning_score, 0.0, 25.0)
    planning = tool_score(precision, recall, beta=2.0, weight=25.0)
    return {
        "answer": answer,
        "reasoning": reasoning,
        "tool": planning,
        "total": answer + reasoning + planning,
    }
