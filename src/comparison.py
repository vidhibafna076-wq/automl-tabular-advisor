from typing import Any


def _safe_round(value: Any, digits: int = 4) -> float | None:
    """
    Round numeric values safely for reporting.
    """

    if value is None:
        return None

    try:
        return round(float(value), digits)
    except Exception:
        return None


def _get_metric_summary(
    result: dict[str, Any],
    metric_name: str,
) -> dict[str, Any]:
    """
    Get one metric summary from a model result.
    """

    return result.get("metrics", {}).get(
        metric_name,
        {
            "cv_mean": None,
            "cv_std": None,
            "train_mean": None,
            "train_std": None,
        },
    )


def _calculate_improvement_against_dummy(
    model_score: float | None,
    dummy_score: float | None,
    direction: str,
) -> dict[str, Any]:
    """
    Compare a model's primary score against the dummy baseline.

    For higher-is-better metrics:
        improvement = model_score - dummy_score

    For lower-is-better metrics:
        improvement = dummy_score - model_score
    """

    if model_score is None or dummy_score is None:
        return {
            "raw_improvement": None,
            "percentage_improvement": None,
            "beats_dummy": False,
            "ties_dummy": False,
        }

    if direction == "lower_is_better":
        raw_improvement = dummy_score - model_score
    else:
        raw_improvement = model_score - dummy_score

    ties_dummy = abs(raw_improvement) < 1e-9
    beats_dummy = raw_improvement > 0

    if abs(dummy_score) > 1e-12:
        percentage_improvement = (raw_improvement / abs(dummy_score)) * 100
    else:
        percentage_improvement = None

    return {
        "raw_improvement": _safe_round(raw_improvement),
        "percentage_improvement": _safe_round(percentage_improvement),
        "beats_dummy": beats_dummy,
        "ties_dummy": ties_dummy,
    }


def _calculate_overfit_gap(
    cv_mean: float | None,
    train_mean: float | None,
    direction: str,
) -> float | None:
    """
    Calculate the train-vs-validation gap.

    For higher-is-better metrics:
        gap = train_mean - cv_mean

    For lower-is-better metrics:
        gap = cv_mean - train_mean

    A large positive gap suggests possible overfitting.
    """

    if cv_mean is None or train_mean is None:
        return None

    if direction == "lower_is_better":
        gap = cv_mean - train_mean
    else:
        gap = train_mean - cv_mean

    return _safe_round(gap)


def _add_flag(
    flags: list[dict[str, Any]],
    severity: str,
    issue: str,
    finding: str,
    recommendation: str,
) -> None:
    """
    Add a reliability flag.
    """

    flags.append(
        {
            "severity": severity,
            "issue": issue,
            "finding": finding,
            "recommendation": recommendation,
        }
    )


def _create_reliability_flags(
    result: dict[str, Any],
    primary_metric: str,
    direction: str,
    dummy_score: float | None,
    profile: dict[str, Any],
    training_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Create model-level reliability flags.
    """

    flags: list[dict[str, Any]] = []

    rows = profile.get("rows", 0)
    base_task = training_summary.get("base_task")

    if result["status"] != "completed":
        _add_flag(
            flags=flags,
            severity="high",
            issue="Model failed",
            finding=f"{result['display_name']} did not complete training.",
            recommendation="Review the error message before using this model.",
        )
        return flags

    if rows < 100:
        _add_flag(
            flags=flags,
            severity="high",
            issue="Very small dataset",
            finding=f"The model was evaluated on only {rows} rows.",
            recommendation=(
                "Treat this score as a pipeline test only. Do not make a real "
                "model selection decision from such a small dataset."
            ),
        )

    primary_summary = _get_metric_summary(result, primary_metric)
    cv_mean = primary_summary.get("cv_mean")
    cv_std = primary_summary.get("cv_std")
    train_mean = primary_summary.get("train_mean")

    if cv_mean is None:
        _add_flag(
            flags=flags,
            severity="high",
            issue="Missing primary score",
            finding=f"No valid cross-validation score was produced for {primary_metric}.",
            recommendation="Do not rank this model until the metric issue is resolved.",
        )

    if cv_std is not None and cv_std >= 0.15:
        _add_flag(
            flags=flags,
            severity="medium",
            issue="High cross-validation variability",
            finding=(
                f"The {primary_metric} CV standard deviation is "
                f"{_safe_round(cv_std)}."
            ),
            recommendation=(
                "The model score changes noticeably across folds. Use caution, "
                "especially with small datasets."
            ),
        )

    overfit_gap = _calculate_overfit_gap(
        cv_mean=cv_mean,
        train_mean=train_mean,
        direction=direction,
    )

    if overfit_gap is not None and overfit_gap >= 0.20:
        _add_flag(
            flags=flags,
            severity="medium",
            issue="Possible overfitting",
            finding=(
                f"The train-vs-validation gap for {primary_metric} is "
                f"{overfit_gap}."
            ),
            recommendation=(
                "Prefer simpler models or collect more data before trusting this result."
            ),
        )

    improvement = _calculate_improvement_against_dummy(
        model_score=result.get("primary_score"),
        dummy_score=dummy_score,
        direction=direction,
    )

    if not result.get("is_dummy_baseline") and not improvement["beats_dummy"]:
        if improvement["ties_dummy"]:
            finding = "This model ties the dummy baseline on the primary metric."
        else:
            finding = "This model does not beat the dummy baseline on the primary metric."

        _add_flag(
            flags=flags,
            severity="high",
            issue="Does not improve over dummy baseline",
            finding=finding,
            recommendation=(
                "Do not treat this model as useful yet. It must clearly outperform "
                "the dummy baseline on a larger dataset."
            ),
        )

    if result.get("is_dummy_baseline"):
        _add_flag(
            flags=flags,
            severity="info",
            issue="Dummy baseline",
            finding="This model is a naive baseline, not a useful final model.",
            recommendation=(
                "Use this result only as a reference point for judging real models."
            ),
        )

    if base_task == "classification":
        balanced_accuracy = _get_metric_summary(
            result,
            "balanced_accuracy",
        ).get("cv_mean")

        if balanced_accuracy is not None and balanced_accuracy <= 0.55:
            _add_flag(
                flags=flags,
                severity="medium",
                issue="Weak balanced accuracy",
                finding=(
                    f"Balanced accuracy is only {_safe_round(balanced_accuracy)}."
                ),
                recommendation=(
                    "The model may not be learning both classes well. Review confusion "
                    "matrix and class-level recall later."
                ),
            )

    return flags


def _reliability_level(flags: list[dict[str, Any]]) -> str:
    """
    Summarise reliability flags into a simple level.
    """

    severities = [flag["severity"] for flag in flags]

    if "high" in severities:
        return "low"

    if "medium" in severities:
        return "medium"

    return "high"


def create_model_leaderboard(
    model_results: list[dict[str, Any]],
    training_summary: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Create a ranked model leaderboard and comparison summary.
    """

    primary_metric = training_summary["primary_metric"]
    direction = training_summary["primary_metric_direction"]

    completed_results = [
        result
        for result in model_results
        if result["status"] == "completed"
    ]

    dummy_result = next(
        (
            result
            for result in completed_results
            if result.get("is_dummy_baseline")
        ),
        None,
    )

    dummy_score = dummy_result.get("primary_score") if dummy_result else None

    def sort_key(result: dict[str, Any]) -> float:
        score = result.get("primary_score")

        if score is None:
            return float("inf") if direction == "lower_is_better" else float("-inf")

        return float(score)

    reverse = direction == "higher_is_better"

    ranked_results = sorted(
        completed_results,
        key=sort_key,
        reverse=reverse,
    )

    leaderboard = []

    for rank, result in enumerate(ranked_results, start=1):
        primary_summary = _get_metric_summary(
            result,
            primary_metric,
        )

        improvement = _calculate_improvement_against_dummy(
            model_score=result.get("primary_score"),
            dummy_score=dummy_score,
            direction=direction,
        )

        overfit_gap = _calculate_overfit_gap(
            cv_mean=primary_summary.get("cv_mean"),
            train_mean=primary_summary.get("train_mean"),
            direction=direction,
        )

        flags = _create_reliability_flags(
            result=result,
            primary_metric=primary_metric,
            direction=direction,
            dummy_score=dummy_score,
            profile=profile,
            training_summary=training_summary,
        )

        leaderboard.append(
            {
                "rank": rank,
                "model_id": result["model_id"],
                "display_name": result["display_name"],
                "family": result["family"],
                "complexity": result["complexity"],
                "interpretability": result["interpretability"],
                "is_dummy_baseline": result["is_dummy_baseline"],
                "primary_metric": primary_metric,
                "primary_score": _safe_round(result.get("primary_score")),
                "primary_cv_std": _safe_round(primary_summary.get("cv_std")),
                "primary_train_score": _safe_round(primary_summary.get("train_mean")),
                "overfit_gap": overfit_gap,
                "dummy_baseline_score": _safe_round(dummy_score),
                "improvement_over_dummy": improvement,
                "reliability_level": _reliability_level(flags),
                "reliability_flags": flags,
            }
        )

    best_overall = leaderboard[0] if leaderboard else None

    useful_candidates = [
        item
        for item in leaderboard
        if not item["is_dummy_baseline"]
        and item["improvement_over_dummy"]["beats_dummy"]
    ]

    best_useful_model = useful_candidates[0] if useful_candidates else None

    important_notes = []

    if profile.get("rows", 0) < 100:
        important_notes.append(
            "The dataset is very small, so leaderboard ranking is not reliable for real model selection."
        )

    if dummy_score is not None and not best_useful_model:
        important_notes.append(
            "No non-dummy model clearly beats the dummy baseline on the primary metric."
        )

    if training_summary.get("unsupported_metrics_in_first_trainer"):
        important_notes.append(
            "Some planned metrics were not implemented in the first trainer: "
            + ", ".join(training_summary["unsupported_metrics_in_first_trainer"])
            + "."
        )

    comparison_summary = {
        "primary_metric": primary_metric,
        "primary_metric_direction": direction,
        "dummy_baseline_score": _safe_round(dummy_score),
        "models_ranked": len(leaderboard),
        "best_overall_model_id": best_overall["model_id"] if best_overall else None,
        "best_overall_display_name": best_overall["display_name"] if best_overall else None,
        "best_overall_score": best_overall["primary_score"] if best_overall else None,
        "best_useful_model_id": best_useful_model["model_id"] if best_useful_model else None,
        "best_useful_display_name": best_useful_model["display_name"] if best_useful_model else None,
        "recommendation_status": (
            "candidate_found"
            if best_useful_model
            else "no_model_clearly_beats_dummy"
        ),
        "important_notes": important_notes,
    }

    return leaderboard, comparison_summary