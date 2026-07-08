from datetime import datetime
from typing import Any

from src.state import ExperimentState


def _add_tool_event(
    state: ExperimentState,
    tool_name: str,
    status: str,
    message: str,
) -> None:
    """
    Add a factual event to the experiment tool history.
    """

    state.tool_history.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tool_name": tool_name,
            "status": status,
            "message": message,
        }
    )


def _add_finding(
    findings: list[dict[str, Any]],
    severity: str,
    issue: str,
    evidence: str,
    recommendation: str,
) -> None:
    """
    Add one critic finding.
    """

    findings.append(
        {
            "severity": severity,
            "issue": issue,
            "evidence": evidence,
            "recommendation": recommendation,
        }
    )


def _get_non_dummy_models(leaderboard: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Return leaderboard entries that are not dummy baselines.
    """

    return [
        item
        for item in leaderboard
        if not item.get("is_dummy_baseline", False)
    ]


def _get_best_useful_model(leaderboard: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Return the first non-dummy model that beats the dummy baseline.
    """

    for item in leaderboard:
        if item.get("is_dummy_baseline"):
            continue

        improvement = item.get("improvement_over_dummy", {})

        if improvement.get("beats_dummy"):
            return item

    return None


def _count_findings_by_severity(
    findings: list[dict[str, Any]],
    severity: str,
) -> int:
    """
    Count critic findings with a specific severity.
    """

    return sum(
        1
        for finding in findings
        if finding["severity"] == severity
    )


def create_critic_report(state: ExperimentState) -> dict[str, Any]:
    """
    Create a reliability critique of the AutoML experiment.

    The critic does not train models.
    The critic does not tune models.
    The critic decides whether the current experiment is trustworthy enough
    to support a recommendation.
    """

    profile = state.profile
    leaderboard = state.leaderboard
    comparison_summary = state.comparison_summary
    training_summary = state.training_summary

    rows = profile.get("rows", 0)
    findings: list[dict[str, Any]] = []

    # 1. Dataset size check
    if rows < 100:
        _add_finding(
            findings=findings,
            severity="high",
            issue="Dataset too small for reliable model selection",
            evidence=(
                f"The experiment used only {rows} rows. This is enough to test "
                "the pipeline, but not enough to confidently select a model."
            ),
            recommendation=(
                "Use a larger dataset before making a real modelling decision. "
                "Treat current results as a technical pipeline test only."
            ),
        )
    elif rows < 500:
        _add_finding(
            findings=findings,
            severity="medium",
            issue="Dataset is small",
            evidence=f"The experiment used {rows} rows.",
            recommendation=(
                "Use cross-validation and interpret model differences carefully. "
                "Collect more data if possible."
            ),
        )

    # 2. Dummy baseline comparison
    recommendation_status = comparison_summary.get("recommendation_status")

    if recommendation_status == "no_model_clearly_beats_dummy":
        _add_finding(
            findings=findings,
            severity="high",
            issue="No useful model clearly beats dummy baseline",
            evidence=(
                "The comparison summary says no non-dummy model clearly beats the "
                "dummy baseline on the primary metric."
            ),
            recommendation=(
                "Do not recommend a final model yet. Review data quality, collect "
                "more data, and rerun the experiment."
            ),
        )

    # 3. Best overall is dummy baseline
    best_overall_model_id = comparison_summary.get("best_overall_model_id")

    if best_overall_model_id and "dummy" in best_overall_model_id:
        _add_finding(
            findings=findings,
            severity="high",
            issue="Best ranked model is the dummy baseline",
            evidence=(
                f"The best overall model is listed as "
                f"{comparison_summary.get('best_overall_display_name')}."
            ),
            recommendation=(
                "A dummy baseline should not be treated as a useful final model. "
                "The real models must beat it clearly."
            ),
        )

    # 4. Unsupported metrics check
    unsupported_metrics = training_summary.get(
        "unsupported_metrics_in_first_trainer",
        [],
    )

    if unsupported_metrics:
        _add_finding(
            findings=findings,
            severity="medium",
            issue="Some planned metrics are not yet implemented",
            evidence=(
                "The first trainer did not implement: "
                + ", ".join(unsupported_metrics)
                + "."
            ),
            recommendation=(
                "Add these metrics before making a stronger classification "
                "recommendation, especially ROC-AUC or PR-AUC when useful."
            ),
        )

    # 5. Reliability level of top non-dummy models
    non_dummy_models = _get_non_dummy_models(leaderboard)

    if non_dummy_models:
        top_non_dummy = non_dummy_models[0]

        if top_non_dummy.get("reliability_level") == "low":
            _add_finding(
                findings=findings,
                severity="high",
                issue="Top non-dummy model has low reliability",
                evidence=(
                    f"The top non-dummy model is {top_non_dummy['display_name']}, "
                    "but its reliability level is low."
                ),
                recommendation=(
                    "Do not tune or recommend this model yet. First improve the "
                    "experiment reliability."
                ),
            )

    # 6. Overfitting and weak balanced accuracy summary
    overfit_models = []
    weak_balanced_accuracy_models = []

    for item in non_dummy_models:
        for flag in item.get("reliability_flags", []):
            if flag["issue"] == "Possible overfitting":
                overfit_models.append(item["display_name"])

            if flag["issue"] == "Weak balanced accuracy":
                weak_balanced_accuracy_models.append(item["display_name"])

    if overfit_models:
        _add_finding(
            findings=findings,
            severity="medium",
            issue="Possible overfitting detected",
            evidence=(
                "Possible overfitting was flagged for: "
                + ", ".join(sorted(set(overfit_models)))
                + "."
            ),
            recommendation=(
                "Prefer simpler models, use more data, or add regularisation before "
                "trusting these results."
            ),
        )

    if weak_balanced_accuracy_models:
        _add_finding(
            findings=findings,
            severity="medium",
            issue="Weak balanced accuracy detected",
            evidence=(
                "Weak balanced accuracy was flagged for: "
                + ", ".join(sorted(set(weak_balanced_accuracy_models)))
                + "."
            ),
            recommendation=(
                "Review class-level performance later using a confusion matrix, "
                "recall per class, ROC-AUC, and PR-AUC where appropriate."
            ),
        )

    # 7. Training completion check
    models_failed = training_summary.get("models_failed", 0)

    if models_failed > 0:
        _add_finding(
            findings=findings,
            severity="medium",
            issue="Some models failed during training",
            evidence=f"{models_failed} model(s) failed during baseline training.",
            recommendation="Inspect model errors before trusting the leaderboard.",
        )

    # 8. Decide overall reliability
    high_count = _count_findings_by_severity(findings, "high")
    medium_count = _count_findings_by_severity(findings, "medium")

    best_useful_model = _get_best_useful_model(leaderboard)

    if high_count > 0:
        overall_reliability = "low"
        recommendation_decision = "do_not_recommend_model"
        can_proceed_to_tuning = False
    elif medium_count > 0:
        overall_reliability = "medium"
        recommendation_decision = "recommend_with_caution"
        can_proceed_to_tuning = True
    else:
        overall_reliability = "high"
        recommendation_decision = "recommend_candidate_model"
        can_proceed_to_tuning = True

    if best_useful_model is None:
        selected_candidate = None
        can_proceed_to_tuning = False

        if recommendation_decision != "do_not_recommend_model":
            recommendation_decision = "do_not_recommend_model"
            overall_reliability = "low"

    else:
        selected_candidate = {
            "model_id": best_useful_model["model_id"],
            "display_name": best_useful_model["display_name"],
            "primary_metric": best_useful_model["primary_metric"],
            "primary_score": best_useful_model["primary_score"],
            "reliability_level": best_useful_model["reliability_level"],
        }

    # 9. Next recommended actions
    if recommendation_decision == "do_not_recommend_model":
        next_actions = [
            "Do not select a final model from this run.",
            "Use a larger dataset before trusting model ranking.",
            "Review whether the selected features contain enough predictive signal.",
            "Rerun the workflow after improving the dataset or evaluation evidence.",
        ]

        if unsupported_metrics:
            next_actions.insert(
                -1,
                "Add planned metrics such as ROC-AUC or PR-AUC where appropriate.",
            )

    elif recommendation_decision == "recommend_with_caution":
        next_actions = [
            "Treat the best useful model as a tentative candidate only.",
            "Run additional evaluation on a larger holdout set if available.",
            "Review reliability flags before tuning.",
            "Proceed to limited tuning only if the model clearly beats the dummy baseline.",
        ]

    else:
        next_actions = [
            "Proceed to model recommendation.",
            "Consider tuning the selected candidate model.",
            "Generate a final report explaining performance and reliability.",
        ]

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "overall_reliability": overall_reliability,
        "recommendation_decision": recommendation_decision,
        "can_proceed_to_tuning": can_proceed_to_tuning,
        "selected_candidate": selected_candidate,
        "finding_counts": {
            "high": high_count,
            "medium": medium_count,
            "low": _count_findings_by_severity(findings, "low"),
            "info": _count_findings_by_severity(findings, "info"),
        },
        "findings": findings,
        "next_actions": next_actions,
    }


def critic_agent(state: ExperimentState) -> dict[str, Any]:
    """
    Reliability critic agent.

    This agent reviews the completed leaderboard and decides whether the
    experiment is trustworthy enough for recommendation or tuning.
    """

    tool_name = "critic_agent"
    state.iteration += 1

    _add_tool_event(
        state=state,
        tool_name=tool_name,
        status="started",
        message="Started reliability critique.",
    )

    try:
        if state.status != "model_comparison_completed":
            raise ValueError(
                "Cannot run critic before model comparison is completed."
            )

        if not state.leaderboard:
            raise ValueError("No leaderboard found in experiment state.")

        if not state.comparison_summary:
            raise ValueError("No comparison summary found in experiment state.")

        critic_report = create_critic_report(state)

        state.critic_report = critic_report
        state.status = "reliability_critique_completed"
        state.completed_steps.append("created_critic_report")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="completed",
            message=(
                "Reliability critique completed with decision: "
                f"{critic_report['recommendation_decision']}."
            ),
        )

        return {
            "success": True,
            "state": state,
            "critic_report": critic_report,
            "error": None,
        }

    except Exception as error:
        state.status = "reliability_critique_failed"
        state.warnings.append(str(error))
        state.completed_steps.append("failed_critic_report")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="failed",
            message=str(error),
        )

        return {
            "success": False,
            "state": state,
            "critic_report": {},
            "error": str(error),
        }