from typing import Any

from src.state import ExperimentState


SUPPORTED_TUNING_TASKS = {
    "classification",
    "regression",
}


def _find_candidate(
    leaderboard: list[dict[str, Any]],
    model_id: str | None,
) -> dict[str, Any]:
    """
    Find a model in the leaderboard by model ID.
    """

    if not model_id:
        return {}

    for item in leaderboard:
        if item.get("model_id") == model_id:
            return item

    return {}


def assess_tuning_eligibility(
    state: ExperimentState,
    max_trials: int = 20,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """
    Decide whether limited hyperparameter tuning is justified.

    This function does not tune a model. It only evaluates whether the
    experiment has enough evidence to permit tuning.

    Tuning is allowed only when:
    - the dataset is large enough;
    - a final holdout was reserved;
    - a useful non-dummy model was selected;
    - that model beats the dummy baseline;
    - the candidate does not have low reliability;
    - there are no high-severity leakage warnings;
    - the task is supported.

    The untouched holdout is never part of the tuning budget or tuning data.
    """

    profile = state.profile or {}
    leaderboard = state.leaderboard or []
    comparison_summary = state.comparison_summary or {}
    validation_summary = state.validation_summary or {}
    leakage_warnings = getattr(
        state,
        "leakage_warnings",
        [],
    ) or []

    rows = int(profile.get("rows", 0) or 0)
    base_task = state.base_task

    candidate_model_id = comparison_summary.get(
        "best_useful_model_id"
    )

    candidate = _find_candidate(
        leaderboard=leaderboard,
        model_id=candidate_model_id,
    )

    blocking_reasons: list[str] = []

    if rows < 200:
        blocking_reasons.append(
            "The dataset has fewer than 200 rows."
        )

    if not validation_summary.get("holdout_used"):
        blocking_reasons.append(
            "No final holdout was reserved. Limited tuning requires a "
            "separate untouched holdout for final evaluation."
        )

    if not candidate_model_id:
        blocking_reasons.append(
            "No useful non-dummy candidate model was selected."
        )

    recommendation_status = comparison_summary.get(
        "recommendation_status"
    )

    if recommendation_status == "no_model_clearly_beats_dummy":
        blocking_reasons.append(
            "No non-dummy candidate clearly beats the dummy baseline."
        )

    if candidate:
        improvement = candidate.get(
            "improvement_over_dummy",
            {},
        )

        if improvement.get("beats_dummy") is not True:
            blocking_reasons.append(
                "The selected candidate does not clearly beat the dummy baseline."
            )

        if candidate.get("reliability_level") == "low":
            blocking_reasons.append(
                "The selected candidate has low reliability."
            )

    elif candidate_model_id:
        blocking_reasons.append(
            "The selected candidate could not be found in the leaderboard."
        )

    high_leakage_warnings = [
        warning
        for warning in leakage_warnings
        if warning.get("severity") == "high"
    ]

    if high_leakage_warnings:
        blocking_reasons.append(
            "High-severity target leakage warnings must be resolved before tuning."
        )

    if base_task not in SUPPORTED_TUNING_TASKS:
        blocking_reasons.append(
            f"Tuning is not implemented for base task '{base_task}'."
        )

    if max_trials < 1:
        blocking_reasons.append(
            "The tuning trial budget must be at least 1."
        )

    if timeout_seconds < 1:
        blocking_reasons.append(
            "The tuning timeout must be at least 1 second."
        )

    eligible = len(blocking_reasons) == 0

    if eligible:
        reason = (
            "Limited tuning is allowed because a useful candidate beats the "
            "dummy baseline, the candidate is not low reliability, the dataset "
            "has a reserved untouched holdout, and no high-severity leakage "
            "warning blocks the experiment."
        )
    else:
        reason = (
            "Limited tuning is not allowed because one or more tuning "
            "guardrails were not satisfied."
        )

    return {
        "eligible": eligible,
        "reason": reason,
        "blocking_reasons": blocking_reasons,
        "candidate_model_id": candidate_model_id,
        "candidate_display_name": candidate.get(
            "display_name"
        ),
        "candidate_reliability": candidate.get(
            "reliability_level"
        ),
        "candidate_primary_metric": candidate.get(
            "primary_metric"
        ),
        "candidate_primary_score": candidate.get(
            "primary_score"
        ),
        "rows": rows,
        "holdout_reserved": bool(
            validation_summary.get("holdout_used")
        ),
        "base_task": base_task,
        "budget": {
            "max_trials": int(max_trials),
            "timeout_seconds": int(timeout_seconds),
        },
        "holdout_usage_rule": (
            "The final holdout must not be used during hyperparameter tuning. "
            "Tuning must use cross-validation on the training portion only."
        ),
    }